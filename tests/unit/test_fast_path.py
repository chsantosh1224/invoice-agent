import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from invoice_agent.config import Settings
from invoice_agent.extraction.fast_path import build_fast_path

DATA = Path("data/samples")
DOC = "inv-004-minimal-IN"  # small digital invoice from the committed dataset


def truth() -> dict[str, Any]:
    """The truth invoice of DOC."""
    return json.loads((DATA / f"{DOC}.truth.json").read_text(encoding="utf-8"))["invoice"]


def transcription(t: dict[str, Any], **overrides: Any) -> str:
    """A RawInvoice JSON a perfect model would return for truth `t` (with optional edits)."""
    raw: dict[str, Any] = {
        "document_type": t["document_type"], "invoice_number": t["invoice_number"],
        "invoice_date": t["invoice_date"], "due_date": t["due_date"],
        "po_number": t["po_number"], "currency": t["currency"],
        "vendor": {"name": t["vendor"]["name"], "address": None, "tax_id": t["vendor"]["tax_id"]},
        "buyer": {"name": t["buyer"]["name"], "address": None, "tax_id": t["buyer"]["tax_id"]},
        "line_items": [{"description": li["description"], "quantity": li["quantity"],
                        "unit_price": li["unit_price"], "amount": li["amount"]}
                       for li in t["line_items"]],
        "subtotal": t["subtotal"], "discount": t["discount"],
        "tax_lines": [{"label": "Tax", "amount": t["tax"]}] if t["tax"] else [],
        "total": t["total"], "payment_terms": t["payment_terms"],
    }  # fmt: skip
    raw.update(overrides)
    return json.dumps(raw)


def run(*replies: str) -> Any:
    """Run the fast path on DOC with a fake model that returns `replies` in order."""
    settings = Settings(_env_file=None, extractor_model="fake/a", extractor_fallback_model="fake/b")  # type: ignore[call-arg]
    fake = GenericFakeChatModel(messages=iter(AIMessage(r) for r in replies))
    graph = build_fast_path(settings, llm_factory=lambda model, s: fake)
    return graph.invoke({"path": str(DATA / f"{DOC}.pdf"), "run_id": "test"})["result"]


def test_correct_extraction_is_accepted() -> None:
    """A perfect transcription passes the gate and is auto-accepted."""
    result = run(transcription(truth()))
    assert result.status == "accepted", result.issues
    assert result.route == "fast_path" and result.score >= 0.8
    assert result.invoice.total == Decimal(truth()["total"])


def test_wrong_total_is_sent_to_review() -> None:
    """A misread total fails the arithmetic and grounding checks."""
    t = truth()
    result = run(transcription(t, total=str(Decimal(t["total"]) + 500)))
    assert result.status == "needs_human_review"
    assert {"math.total", "grounding.missing"} <= {i.code for i in result.issues}


def test_unusable_output_ends_in_review_not_a_crash() -> None:
    """If neither model returns valid JSON, the invoice goes to review with the reason."""
    result = run(*["not json"] * 6)
    assert result.status == "needs_human_review"
    assert result.issues[0].code == "extract.failed"
