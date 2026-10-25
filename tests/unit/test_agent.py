import json
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from invoice_agent.config import Settings
from invoice_agent.extraction.router import process_document

DATA = Path("data/samples")
DOC = "inv-034-minimal-US"  # real calibration failure: the model dropped the currency


class ScriptedModel(GenericFakeChatModel):
    """A fake chat model that supports bind_tools and replies from a script."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":  # type: ignore[override]
        """Tools are ignored: the script already contains the tool calls."""
        return self


def raw_from_truth(**overrides: Any) -> dict[str, Any]:
    """A transcription of DOC as a perfect model would produce it, with edits."""
    t = json.loads((DATA / f"{DOC}.truth.json").read_text(encoding="utf-8"))["invoice"]
    raw: dict[str, Any] = {
        "document_type": t["document_type"], "invoice_number": t["invoice_number"],
        "invoice_date": t["invoice_date"], "due_date": t["due_date"],
        "po_number": t["po_number"], "currency": "USD",  # bare "$" is ambiguous
        "vendor": {"name": t["vendor"]["name"], "address": None, "tax_id": t["vendor"]["tax_id"]},
        "buyer": {"name": t["buyer"]["name"], "address": None, "tax_id": t["buyer"]["tax_id"]},
        "line_items": [{k: li[k] for k in ("description", "quantity", "unit_price", "amount")}
                       | {"discount": None, "tax_rate": None, "tax_amount": None}
                       for li in t["line_items"]],
        "summary_lines": summary_from_truth(t), "payment_terms": t["payment_terms"],
    }  # fmt: skip
    raw.update(overrides)
    return raw


def summary_from_truth(t: dict[str, Any], total: str | None = None) -> list[dict[str, Any]]:
    """Totals-block rows a perfect model would transcribe for truth `t`."""
    rows: list[dict[str, Any]] = [
        {"label": "Subtotal", "amount": t["subtotal"], "kind": "subtotal"}
    ]
    if t["discount"]:
        rows.append({"label": "Discount", "amount": t["discount"], "kind": "discount"})
    if t["tax"]:
        rows.append({"label": "Tax", "amount": t["tax"], "kind": "tax"})
    rows.append({"label": "Total", "amount": total or t["total"], "kind": "total"})
    return rows


def call(name: str, args: dict[str, Any], i: int) -> AIMessage:
    """An AI turn that calls one tool."""
    return AIMessage(
        "", tool_calls=[{"name": name, "args": args, "id": f"c{i}", "type": "tool_call"}]
    )


def run(*script: AIMessage) -> Any:
    """Route DOC through fast path + agent with a scripted model."""
    settings = Settings(_env_file=None, extractor_model="fake/x", extractor_fallback_model="fake/x",
                        agent_model="fake/agent", agent_max_steps=5)  # type: ignore[call-arg]  # fmt: skip
    model = ScriptedModel(messages=iter(script))
    return process_document(DATA / f"{DOC}.pdf", settings, llm_factory=lambda m, s: model)[0]


def test_agent_fixes_missing_currency_and_is_accepted() -> None:
    """Fast path drops the currency -> gate fails -> agent finds '$' -> re-validated -> accepted."""
    fixed = raw_from_truth()
    result = run(
        AIMessage(json.dumps(raw_from_truth(currency=None))),  # fast path's flawed answer
        call("search_text", {"query": "$"}, 1),
        call("check_draft", {"draft": fixed}, 2),
        call("submit_invoice", {"draft": fixed, "notes": ["currency: '$' printed on totals"]}, 3),
    )
    assert result.status == "accepted", result.issues
    assert result.route == "agent" and result.invoice.currency == "USD"
    assert result.notes == ["currency: '$' printed on totals"]


def test_agent_cannot_talk_its_way_past_the_gate() -> None:
    """A submission that is still wrong is re-validated and sent to human review."""
    still_wrong = raw_from_truth(currency=None)
    result = run(
        AIMessage(json.dumps(still_wrong)),
        call("submit_invoice", {"draft": still_wrong, "notes": ["looks fine to me"]}, 1),
    )
    assert result.route == "agent" and result.status == "needs_human_review"
    assert any(i.code == "required.missing" for i in result.issues)


def test_agent_without_submission_ends_in_review() -> None:
    """An agent that never submits hits the step limit and goes to human review."""
    result = run(
        AIMessage(json.dumps(raw_from_truth(currency=None))),
        *[call("search_text", {"query": "total"}, i) for i in range(5)],
    )
    assert result.status == "needs_human_review"
    assert result.issues[0].code == "agent.no_submission"


def test_clean_invoice_never_reaches_the_agent() -> None:
    """A fast-path answer that passes the gate is accepted without any agent turn."""
    result = run(AIMessage(json.dumps(raw_from_truth())))
    assert result.route == "fast_path" and result.status == "accepted"


def test_notes_sent_as_one_string_stay_one_note() -> None:
    """Nemotron Super sent notes as a plain string; it must not be split into characters."""
    fixed = raw_from_truth()
    result = run(
        AIMessage(json.dumps(raw_from_truth(currency=None))),
        call("submit_invoice", {"draft": fixed, "notes": "added currency '$'"}, 1),
    )
    assert result.notes == ["added currency '$'"]


def test_invalid_submission_loops_back_then_valid_one_finishes() -> None:
    """submission='invalid' sends the run back to decide; the next valid draft finishes."""
    fixed = raw_from_truth()
    result = run(
        AIMessage(json.dumps(raw_from_truth(currency=None))),
        call("submit_invoice", {"draft": {"total": "not a full invoice"}, "notes": []}, 1),
        call("submit_invoice", {"draft": fixed, "notes": ["second try"]}, 2),
    )
    assert result.status == "accepted", result.issues
    assert result.notes == ["second try"]


def test_router_reads_submission_flag() -> None:
    """The single router follows the explicit state facts, in order."""
    from invoice_agent.extraction.agent.graph import ExtractionAgent

    settings = Settings(_env_file=None, agent_max_steps=8)  # type: ignore[call-arg]
    agent = ExtractionAgent("fake/agent", settings, llm_factory=lambda m, s: ScriptedModel(
        messages=iter([])))  # fmt: skip
    tool = call("search_text", {"query": "$"}, 1)
    submit = call("submit_invoice", {"draft": {}}, 2)
    rejected = ToolMessage("Submission rejected", tool_call_id="c2")
    assert agent.route({"messages": [tool], "steps": 1, "submission": "none"}) == "tools"
    assert agent.route({"messages": [submit], "steps": 1, "submission": "none"}) == "check_submit"
    assert agent.route({"messages": [submit], "steps": 1, "submission": "valid"}) == "finalize"
    assert agent.route({"messages": [rejected], "steps": 3, "submission": "invalid"}) == "decide"
    assert agent.route({"messages": [rejected], "steps": 8, "submission": "invalid"}) == "finalize"
    assert agent.route({"messages": [tool], "steps": 8, "submission": "none"}) == "finalize"


class BrokenModel(ScriptedModel):
    """A model whose provider is down: every agent call fails like NVIDIA's 404."""

    def invoke(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Fail the way the real provider did."""
        raise RuntimeError("404 Function not found for account")


class ModelPicker:
    """llm_factory stand-in: returns a specific fake model per slug, else a default."""

    def __init__(self, models: dict[str, Any], default: Any) -> None:
        """Remember which fake to return for which model slug."""
        self.models, self.default = models, default

    def __call__(self, model: str, settings: Settings) -> Any:
        """Return the fake model for `model`."""
        return self.models.get(model, self.default)


def test_agent_falls_back_to_the_second_model() -> None:
    """Primary agent model down -> the fallback model runs the investigation."""
    fixed = raw_from_truth()
    good = ScriptedModel(messages=iter([
        AIMessage(json.dumps(raw_from_truth(currency=None))),
        call("submit_invoice", {"draft": fixed, "notes": ["added currency"]}, 1),
    ]))  # fmt: skip
    settings = Settings(_env_file=None, extractor_model="fake/x", extractor_fallback_model="fake/x",
                        agent_model="fake/broken", agent_fallback_model="fake/backup")  # type: ignore[call-arg]  # fmt: skip
    pick = {"fake/broken": BrokenModel(messages=iter([]))}
    result, _ = process_document(DATA / f"{DOC}.pdf", settings, llm_factory=ModelPicker(pick, good))
    assert result.route == "agent" and result.status == "accepted"
    assert result.model == "fake/backup"


def test_unavailable_agent_is_reported_not_hidden() -> None:
    """Both agent models down -> fast-path verdict plus an agent.unavailable warning."""
    settings = Settings(_env_file=None, extractor_model="fake/x", extractor_fallback_model="fake/x",
                        agent_model="fake/a", agent_fallback_model="fake/b")  # type: ignore[call-arg]  # fmt: skip
    fast = ScriptedModel(messages=iter([AIMessage(json.dumps(raw_from_truth(currency=None)))]))
    broken = BrokenModel(messages=iter([]))
    factory = ModelPicker({"fake/x": fast}, broken)
    result, _ = process_document(DATA / f"{DOC}.pdf", settings, llm_factory=factory)
    assert result.route == "fast_path" and result.status == "needs_human_review"
    unavailable = [i for i in result.issues if i.code == "agent.unavailable"]
    assert unavailable and "a: RuntimeError" in unavailable[0].message
