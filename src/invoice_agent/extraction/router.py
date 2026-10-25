"""The router: cheap fast path first, the agent only for invoices the gate rejects.

invoice -> fast path -> gate passed? -> accepted                  (most invoices)
                     -> gate failed  -> agent -> gate passed? -> accepted
                                              -> still failing -> needs_human_review
                     -> no output at all (provider down)      -> needs_human_review
"""

import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from invoice_agent.config import Settings, get_settings
from invoice_agent.extraction.agent import workspace
from invoice_agent.extraction.agent.graph import build_extraction_agent
from invoice_agent.extraction.agent.prompts import initial_task
from invoice_agent.extraction.fast_path import build_fast_path
from invoice_agent.extraction.messages import image_to_data_url
from invoice_agent.llm import get_chat_model
from invoice_agent.llm_cache import ResponseCache
from invoice_agent.models_catalog import load_catalog
from invoice_agent.schemas import ExtractionResult, ValidationIssue

log = structlog.get_logger()


def _model_sees_images(model: str, settings: Settings) -> bool:
    """Check whether the agent model accepts images (then it also gets the page images).

    Args:
        model: OpenRouter slug.
        settings: App settings (catalog cache location).

    Returns:
        True if the catalog lists image input for the model; False if unknown/offline.
    """
    try:
        info = load_catalog(settings.cache_dir).get(model)
    except Exception:
        return False
    return bool(info and info.vision)


def process_document(
    path: Path,
    settings: Settings | None = None,
    *,
    mode: str = "router",
    cache: ResponseCache | None = None,
    llm_factory: Callable[[str, Settings], BaseChatModel] = get_chat_model,
) -> tuple[ExtractionResult, float]:
    """Extract one invoice: fast path, then the agent if the validation gate fails.

    Args:
        path: A PDF or image.
        settings: App settings; defaults to the cached settings.
        mode: 'router' (fast path + agent) or 'fast' (fast path only).
        cache: Response cache for the fast path's LLM call.
        llm_factory: Creates chat models; tests pass fakes.

    Returns:
        (result, seconds). `result.route` says which path produced the answer; agent
        results carry the agent's audit notes. If the gate rejected the fast path but no
        agent model could run (AGENT_MODEL, then AGENT_FALLBACK_MODEL), the fast-path
        result is returned with an `agent.unavailable` warning.
    """
    settings = settings or get_settings()
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    fast = build_fast_path(settings, llm_factory=llm_factory, cache=cache).invoke(
        {"path": str(path), "run_id": run_id}
    )
    result: ExtractionResult = fast["result"]
    if mode == "fast" or result.status == "accepted" or fast.get("raw") is None:
        return result, time.perf_counter() - started

    log.info("router.escalate", run_id=run_id, issues=[i.code for i in result.issues])
    doc_id = workspace.register(workspace.Workspace(Path(path), fast["doc"], fast["images"]))
    errors: list[str] = []
    try:
        for model in dict.fromkeys([settings.agent_model, settings.agent_fallback_model]):
            try:
                agent_result = _run_agent(
                    model, fast, result, doc_id, run_id, settings, llm_factory
                )
            except Exception as e:  # provider down, model removed, ...: try the next model
                errors.append(f"{model.split('/')[-1]}: {type(e).__name__}")
                log.warning("router.agent_failed", run_id=run_id, model=model,
                            error=f"{type(e).__name__}: {e}"[:300])  # fmt: skip
                continue
            return agent_result, time.perf_counter() - started
    finally:
        workspace.release(doc_id)

    # No agent model could run: keep the fast-path verdict, but say the agent was tried.
    unavailable = ValidationIssue(
        field="*",
        code="agent.unavailable",
        message="the agent was tried but its model provider failed ("
        + "; ".join(errors)
        + "); showing the fast-path result",
        severity="warning",
    )
    result = result.model_copy(update={"issues": [*result.issues, unavailable]})
    return result, time.perf_counter() - started


def _run_agent(
    model: str,
    fast: dict[str, Any],
    result: ExtractionResult,
    doc_id: str,
    run_id: str,
    settings: Settings,
    llm_factory: Callable[[str, Settings], BaseChatModel],
) -> ExtractionResult:
    """Run the extraction agent with one model on a document the gate rejected.

    Args:
        model: OpenRouter slug of the agent model.
        fast: The fast path's final state (draft, document, page images).
        result: The fast path's result (its issues become the agent's task).
        doc_id: Workspace id of the registered document.
        run_id: Correlates log lines with the fast-path run.
        settings: App settings (step limit, gate threshold).
        llm_factory: Creates the chat model.

    Returns:
        The agent's ExtractionResult (route='agent'), already re-checked by the gate.

    Raises:
        Exception: Whatever the model provider raises (rate limit, 404, timeout...); the
            caller then tries the next model.
    """
    content: list[str | dict[Any, Any]] = [
        {"type": "text", "text": initial_task(fast["raw"], result.issues, fast["doc"].text)}
    ]
    if _model_sees_images(model, settings):
        content += [
            {"type": "image_url", "image_url": {"url": image_to_data_url(img)}}
            for img in fast["images"]
        ]
    agent = build_extraction_agent(model, settings, llm_factory=llm_factory)
    state = agent.invoke(
        {
            "messages": [HumanMessage(content=content)],
            "doc_id": doc_id,
            "run_id": run_id,
            "steps": 0,
        },
        config={"recursion_limit": 4 * settings.agent_max_steps + 10},
    )
    agent_result: ExtractionResult = state["result"]
    return agent_result
