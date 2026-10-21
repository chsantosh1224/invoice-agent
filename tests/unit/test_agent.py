import json
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

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
        "po_number": t["po_number"], "currency": "$",
        "vendor": {"name": t["vendor"]["name"], "address": None, "tax_id": t["vendor"]["tax_id"]},
        "buyer": {"name": t["buyer"]["name"], "address": None, "tax_id": t["buyer"]["tax_id"]},
        "line_items": [{k: li[k] for k in ("description", "quantity", "unit_price", "amount")}
                       for li in t["line_items"]],
        "subtotal": t["subtotal"], "discount": t["discount"],
        "tax_lines": [{"label": "Sales Tax", "amount": t["tax"]}] if t["tax"] else [],
        "total": t["total"], "payment_terms": t["payment_terms"],
    }  # fmt: skip
    raw.update(overrides)
    return raw


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
