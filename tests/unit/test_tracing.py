import json
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr
from test_agent import (
    DATA,
    DOC,
    ModelPicker,
    ScriptedModel,
    call,
    raw_from_truth,
)

from invoice_agent import tracing
from invoice_agent.config import Settings
from invoice_agent.extraction.router import process_document

KEYS = {"langfuse_public_key": "pk-lf-test", "langfuse_secret_key": SecretStr("sk-lf-test")}


def settings(**extra: object) -> Settings:
    """Fake models; Langfuse keys only if passed in."""
    return Settings(_env_file=None, extractor_model="fake/x", extractor_fallback_model="fake/x",
                    agent_model="fake/agent", agent_max_steps=5, **extra)  # type: ignore[arg-type]  # fmt: skip


class LocalLangfuse(Langfuse):
    """A Langfuse client that keeps spans in memory instead of sending them."""

    exporter = InMemorySpanExporter()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, span_exporter=self.exporter)


class BrokenLangfuse(Langfuse):
    """A Langfuse client that fails to start."""

    def __init__(self, **kwargs: Any) -> None:
        raise RuntimeError("no network")


def app_client() -> Langfuse:
    """The app's (cached) Langfuse client for the test keys."""
    s = settings(**KEYS)
    return tracing._client(s.langfuse_public_key, "sk-lf-test", s.langfuse_host)


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Make the app's Langfuse client record spans in memory for this test."""
    monkeypatch.setattr(tracing, "Langfuse", LocalLangfuse)
    tracing._client.cache_clear()
    LocalLangfuse.exporter.clear()
    yield LocalLangfuse.exporter
    app_client().shutdown()
    tracing._client.cache_clear()


def test_tracing_is_off_without_keys() -> None:
    """No keys: no callbacks, nothing traced, extraction unchanged."""
    s = settings()
    assert not tracing.enabled(s)
    with tracing.trace_run(s, "r1", mode="router") as trace:
        assert trace.callbacks == [] and trace.trace_id is None


def test_one_trace_holds_fast_path_and_agent(exporter: InMemorySpanExporter) -> None:
    """A run that escalates to the agent is one trace: root span, LLM calls, user, outcome."""
    fixed = raw_from_truth()
    model = ScriptedModel(messages=iter([
        AIMessage(json.dumps(raw_from_truth(currency=None))),
        call("submit_invoice", {"draft": fixed, "notes": ["currency: '$'"]}, 1),
    ]))  # fmt: skip
    result, _ = process_document(
        DATA / f"{DOC}.pdf", settings(**KEYS), llm_factory=ModelPicker({}, model),
        user_id="42", filename="acme.pdf",
    )  # fmt: skip
    assert result.route == "agent" and result.status == "accepted"
    app_client().flush()

    spans = exporter.get_finished_spans()
    assert {f"{s.context.trace_id:032x}" for s in spans} == {result.trace_id}  # one trace
    root = next(dict(s.attributes or {}) for s in spans if s.name == tracing.TRACE_NAME)
    assert root["user.id"] == "42"
    assert "acme.pdf" in str(root["langfuse.observation.input"])
    assert '"status": "accepted"' in str(root["langfuse.observation.output"])
    kinds = [(s.attributes or {}).get("langfuse.observation.type") for s in spans]
    generations = [k for k in kinds if k == "generation"]
    assert len(generations) == 2  # the fast path's call and the agent's call


def test_broken_langfuse_does_not_break_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the Langfuse client can't be created, the run goes ahead untraced."""
    monkeypatch.setattr(tracing, "Langfuse", BrokenLangfuse)
    tracing._client.cache_clear()
    with tracing.trace_run(settings(**KEYS), "r1", mode="fast") as trace:
        assert trace.callbacks == []
