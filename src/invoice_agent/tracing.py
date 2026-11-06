"""Optional Langfuse tracing: one trace per invoice, with every LLM call and agent step.

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set. Tracing never breaks an
extraction: if Langfuse can't start, the run goes ahead untraced and the log says why.
Traces are sent in the background, so a slow or unreachable Langfuse doesn't slow uploads.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from invoice_agent.config import Settings
from invoice_agent.schemas import ExtractionResult

log = structlog.get_logger()

TRACE_NAME = "extract-invoice"


@dataclass
class Trace:
    """The current run's trace: callbacks to pass to LangGraph, and its root span."""

    callbacks: list[BaseCallbackHandler] = field(default_factory=list)
    span: Any = None  # LangfuseChain while tracing; None when tracing is off

    @property
    def trace_id(self) -> str | None:
        """The Langfuse trace id (to attach scores later), or None when tracing is off."""
        return None if self.span is None else str(self.span.trace_id)

    def finish(self, result: ExtractionResult) -> None:
        """Record the run's outcome on the trace (a no-op when tracing is off).

        Args:
            result: The final extraction result.
        """
        if self.span is not None:
            self.span.update(
                output={
                    "status": result.status,
                    "route": result.route,
                    "score": result.score,
                    "issues": [i.code for i in result.issues],
                }
            )


def enabled(settings: Settings) -> bool:
    """Check whether Langfuse keys are configured.

    Args:
        settings: App settings.

    Returns:
        True if both the public and the secret key are set.
    """
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key.get_secret_value())


@lru_cache(maxsize=1)
def _client(public_key: str, secret_key: str, host: str) -> Langfuse:
    """Create the Langfuse client once per process (it owns the background sender).

    Args:
        public_key: Langfuse project public key (pk-lf-...).
        secret_key: Langfuse project secret key (sk-lf-...).
        host: Langfuse server, e.g. https://cloud.langfuse.com.

    Returns:
        The shared Langfuse client.
    """
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


@contextmanager
def trace_run(
    settings: Settings,
    run_id: str,
    *,
    mode: str,
    user_id: str | None = None,
    filename: str | None = None,
) -> Iterator[Trace]:
    """Open one Langfuse trace for an invoice run; LLM calls inside it nest under it.

    Args:
        settings: App settings (Langfuse keys and host).
        run_id: The run's id, stored on the trace to match it with the app's logs.
        mode: 'router' or 'fast'; added as a tag.
        user_id: Who uploaded the invoice (the account id), if known.
        filename: The uploaded file's name, shown as the trace's input.

    Yields:
        A Trace: pass `trace.callbacks` to each graph run, then call `trace.finish()`.
        With tracing off (or Langfuse failing to start) the callbacks list is empty.
    """
    if not enabled(settings):
        yield Trace()
        return
    try:
        client = _client(
            settings.langfuse_public_key,
            settings.langfuse_secret_key.get_secret_value(),
            settings.langfuse_host,
        )
        handler = CallbackHandler(public_key=settings.langfuse_public_key)
    except Exception as e:
        log.warning("tracing.unavailable", error=f"{type(e).__name__}: {e}"[:300])
        yield Trace()
        return
    with (
        client.start_as_current_observation(
            name=TRACE_NAME, as_type="chain", input={"file": filename, "mode": mode}
        ) as span,
        propagate_attributes(
            user_id=user_id, trace_name=TRACE_NAME, tags=[mode], metadata={"run_id": run_id}
        ),
    ):
        yield Trace([handler], span)
