r"""The fast path: a fixed LangGraph workflow that every invoice goes through first.

    START -> ingest -> extract --(output)--> validate -> finalize -> END
                             \--(failed)----------------/

One LLM call (two if the primary model is unavailable and the fallback is used), then
the deterministic validation gate decides: accepted, or needs review. This is a workflow,
not an agent - the code fixes the path; the agent handles invoices that fail here.
"""

import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict

import structlog
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from PIL import Image

from invoice_agent.config import Settings, get_settings
from invoice_agent.extraction.messages import build_messages
from invoice_agent.extraction.prompts import PROMPT_VERSION
from invoice_agent.extraction.validate import GateReport, validate
from invoice_agent.ingest.document import Document
from invoice_agent.ingest.light import LightIngestor
from invoice_agent.ingest.pages import page_images
from invoice_agent.llm import get_chat_model
from invoice_agent.llm_cache import ResponseCache
from invoice_agent.models_catalog import output_mode_for
from invoice_agent.normalize import to_invoice
from invoice_agent.schemas import ExtractionResult, Invoice, RawInvoice, ValidationIssue
from invoice_agent.structured import invoke_structured

log = structlog.get_logger()
IMAGE_DPI = 150  # same as the bake-off, so its cached answers are reused


class FastPathState(TypedDict, total=False):
    """Everything the fast-path nodes read and write.

    Attributes:
        path: The document being processed.
        run_id: Correlates all log lines of one run.
        doc: Ingested document (layout text + word positions).
        images: Page images sent to the vision model.
        raw: The model's transcription, if it produced one.
        model: The model that produced `raw`.
        usage: Token usage of the extraction call(s).
        errors: Why extraction failed, per model tried.
        invoice: Normalized invoice.
        gate: The validation gate's report.
        result: The final ExtractionResult.
    """

    path: str
    run_id: str
    doc: Document
    images: list[Image.Image]
    raw: RawInvoice | None
    model: str
    usage: dict[str, int]
    errors: list[str]
    invoice: Invoice
    gate: GateReport
    result: ExtractionResult


class FastPath:
    """The fast-path workflow: configuration in __init__, one method per graph node.

    Usage:
        graph = FastPath(settings, cache=cache).build()
        final_state = graph.invoke({"path": "invoice.pdf", "run_id": "abc"})
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm_factory: Callable[[str, Settings], BaseChatModel] = get_chat_model,
        cache: ResponseCache | None = None,
    ) -> None:
        """Store what the nodes need: settings, model factory, cache and the ingestor.

        Args:
            settings: App settings (models, threshold); defaults to the cached settings.
            llm_factory: Creates the chat model for a slug; tests pass a fake here.
            cache: Response cache for LLM replies, or None to always call the API.
        """
        self.settings = settings or get_settings()
        self.llm_factory = llm_factory
        self.cache = cache
        self.ingestor = LightIngestor()

    # ---------- nodes: each receives the state and returns the keys it changes ----------

    def ingest(self, state: FastPathState) -> dict[str, Any]:
        """Read the document: text + word positions, and page images for the LLM.

        Args:
            state: Must contain 'path'.

        Returns:
            {'doc': Document, 'images': page images}.
        """
        path = Path(state["path"])
        return {"doc": self.ingestor.ingest(path), "images": page_images(path, dpi=IMAGE_DPI)}

    def extract(self, state: FastPathState) -> dict[str, Any]:
        """One LLM call with the primary model; the fallback model only if that fails.

        Args:
            state: Must contain 'doc' and 'images' (from `ingest`).

        Returns:
            On success {'raw', 'model', 'usage', 'errors'}; if every model fails
            {'raw': None, 'errors': [...]}.
        """
        messages = build_messages(images=state["images"], text=state["doc"].text)
        errors: list[str] = []
        models = dict.fromkeys(
            [self.settings.extractor_model, self.settings.extractor_fallback_model]
        )
        for model in models:
            try:
                result = invoke_structured(
                    self.llm_factory(model, self.settings),
                    RawInvoice,
                    messages,
                    model=model,
                    mode=output_mode_for(model, self.settings.cache_dir),
                    cache=self.cache,
                )
            except Exception as e:  # rate limits, provider errors, invalid output
                errors.append(f"{model}: {type(e).__name__}: {str(e)[:200]}")
                log.warning("extract.failed", run_id=state["run_id"], model=model, error=errors[-1])
                continue
            log.info(
                "extract.ok",
                run_id=state["run_id"],
                model=model,
                attempts=result.attempts,
                cached=result.cached,
                usage=result.usage,
            )
            return {"raw": result.value, "model": model, "usage": result.usage, "errors": errors}
        return {"raw": None, "errors": errors}

    def validate(self, state: FastPathState) -> dict[str, Any]:
        """Normalize the transcription and run the validation gate on it.

        Args:
            state: Must contain 'raw' (not None) and 'doc'.

        Returns:
            {'invoice': normalized Invoice, 'gate': GateReport}.
        """
        raw = state["raw"]
        assert raw is not None
        invoice, issues = to_invoice(raw)
        gate = validate(
            invoice, state["doc"], normalize_issues=issues, threshold=self.settings.gate_threshold
        )
        return {"invoice": invoice, "gate": gate}

    def finalize(self, state: FastPathState) -> dict[str, Any]:
        """Package the outcome: accepted only if the gate passed.

        Args:
            state: Final state; 'gate' is missing when extraction itself failed.

        Returns:
            {'result': ExtractionResult}.
        """
        gate = state.get("gate")
        if gate is None:  # extraction itself failed
            issue = ValidationIssue(
                field="*",
                code="extract.failed",
                message="; ".join(state.get("errors", [])) or "no output",
            )
            result = ExtractionResult(
                status="needs_human_review",
                route="fast_path",
                invoice=Invoice(),
                issues=[issue],
                model=self.settings.extractor_model,
                prompt_version=PROMPT_VERSION,
                run_id=state["run_id"],
            )
        else:
            result = ExtractionResult(
                status="accepted" if gate.passed else "needs_human_review",
                route="fast_path",
                invoice=state["invoice"],
                raw=state["raw"],
                issues=gate.issues,
                field_confidence=gate.field_confidence,
                score=gate.score,
                model=state["model"],
                prompt_version=PROMPT_VERSION,
                run_id=state["run_id"],
            )
        log.info(
            "fast_path.done",
            run_id=state["run_id"],
            status=result.status,
            score=round(result.score, 3),
            errors=[i.code for i in result.issues],
        )
        return {"result": result}

    # ---------- routing ----------

    def route_after_extract(self, state: FastPathState) -> Literal["validate", "finalize"]:
        """Validate a transcription if there is one; otherwise go straight to finalize.

        Args:
            state: State after `extract`.

        Returns:
            'validate' or 'finalize'.
        """
        return "validate" if state.get("raw") else "finalize"

    # ---------- wiring ----------

    def build(self) -> Any:
        """Wire the nodes into a graph and compile it.

        Returns:
            A compiled LangGraph runnable: `.invoke({"path": ..., "run_id": ...})`.
        """
        graph = StateGraph(FastPathState)
        graph.add_node("ingest", self.ingest)
        graph.add_node("extract", self.extract)
        graph.add_node("validate", self.validate)
        graph.add_node("finalize", self.finalize)

        graph.add_edge(START, "ingest")
        graph.add_edge("ingest", "extract")
        graph.add_conditional_edges("extract", self.route_after_extract, ["validate", "finalize"])
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()


def build_fast_path(
    settings: Settings | None = None,
    *,
    llm_factory: Callable[[str, Settings], BaseChatModel] = get_chat_model,
    cache: ResponseCache | None = None,
) -> Any:
    """Compile the fast-path graph (shortcut for FastPath(...).build()).

    Args:
        settings: App settings (models, threshold); defaults to the cached settings.
        llm_factory: Creates the chat model for a slug; tests pass a fake here.
        cache: Response cache for LLM replies, or None to always call the API.

    Returns:
        A compiled LangGraph runnable: `.invoke({"path": "..."})` returns the final state.
    """
    return FastPath(settings, llm_factory=llm_factory, cache=cache).build()


def extract_document(
    path: Path, settings: Settings | None = None, *, cache: ResponseCache | None = None
) -> tuple[ExtractionResult, float]:
    """Run the fast path on one file.

    Args:
        path: A PDF or image.
        settings: App settings; defaults to the cached settings.
        cache: Response cache for LLM replies, or None.

    Returns:
        (result, seconds): the ExtractionResult and the wall-clock duration.
    """
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    state = build_fast_path(settings, cache=cache).invoke({"path": str(path), "run_id": run_id})
    return state["result"], time.perf_counter() - started
