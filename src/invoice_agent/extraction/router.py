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
from invoice_agent.schemas import ExtractionResult

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
        results carry the agent's audit notes.
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
    try:
        model = settings.agent_model
        content: list[str | dict[Any, Any]] = [
            {"type": "text", "text": initial_task(fast["raw"], result.issues, fast["doc"].text)}
        ]
        if _model_sees_images(model, settings):
            content += [
                {"type": "image_url", "image_url": {"url": image_to_data_url(img)}}
                for img in fast["images"]
            ]
        agent = build_extraction_agent(model, settings, llm_factory=llm_factory)
        try:
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
        except Exception as e:  # provider down mid-investigation: keep the fast-path verdict
            log.warning(
                "router.agent_failed", run_id=run_id, error=f"{type(e).__name__}: {e}"[:200]
            )
            return result, time.perf_counter() - started
    finally:
        workspace.release(doc_id)
    return agent_result, time.perf_counter() - started
