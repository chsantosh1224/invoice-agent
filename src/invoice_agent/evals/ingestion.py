"""Score ingestion engines against the synthetic truth, without any LLM.

Two questions, per document:
1. Value recall: does each truth value appear somewhere in the engine's text?
   (If it doesn't, no extractor downstream can get it right from the text.)
2. Row integrity: does each line item's qty, unit price and amount appear together on one
   line? (If not, a text-only model has to guess which numbers belong together.)

Matching is done on normalized values (amounts parsed to Decimal, dates tried in every
known print format), so engines are not penalized for spacing or symbol differences.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from invoice_agent.textmatch import compact as _compact
from invoice_agent.textmatch import date_present as _date_present
from invoice_agent.textmatch import numbers_in

__all__ = ["DocScore", "numbers_in", "score_document"]


@dataclass
class DocScore:
    """Ingestion scores for one document and engine.

    Attributes:
        doc_id: Dataset document id.
        found: Field name -> whether its truth value was found in the text.
        rows_total: Number of truth line items.
        rows_intact: Line items whose qty, unit price and amount share one text line.
    """

    doc_id: str
    found: dict[str, bool] = field(default_factory=dict)
    rows_total: int = 0
    rows_intact: int = 0

    @property
    def recall(self) -> float:
        """Share of truth values found in the text.

        Returns:
            found / checked, or 1.0 if nothing was checked.
        """
        return sum(self.found.values()) / len(self.found) if self.found else 1.0


def _check_amount(score: DocScore, nums: set[Decimal], name: str, value: str | None) -> None:
    """Record whether an amount is present in the text (skipped when the truth is null).

    Args:
        score: The score being filled in.
        nums: Every number found in the engine's text.
        name: Field path, e.g. 'total' or 'line_items[0].amount'.
        value: The truth value as a string, or None.

    Returns:
        None. Sets score.found[name].
    """
    if value is not None:
        score.found[name] = abs(Decimal(value)) in nums


def _check_text(score: DocScore, flat: str, name: str, value: str | None) -> None:
    """Record whether a string value is present in the text (spacing/case-insensitive).

    Args:
        score: The score being filled in.
        flat: The engine's text, compacted (no whitespace, lowercase).
        name: Field path, e.g. 'vendor.name'.
        value: The truth value, or None/empty to skip.

    Returns:
        None. Sets score.found[name].
    """
    if value:
        score.found[name] = _compact(value) in flat


def score_document(doc_id: str, text: str, truth: dict[str, Any]) -> DocScore:
    """Score one engine's text for one document against its truth invoice.

    Args:
        doc_id: Dataset document id.
        text: The engine's full layout text / Markdown for the document.
        truth: The 'invoice' object from the document's .truth.json.

    Returns:
        A DocScore with per-field presence and line-item row integrity.
    """
    score = DocScore(doc_id)
    flat = _compact(text)
    nums = numbers_in(text)

    for name in ("invoice_number", "po_number"):
        _check_text(score, flat, name, truth[name])
    for party in ("vendor", "buyer"):
        _check_text(score, flat, f"{party}.name", truth[party]["name"])
        _check_text(score, flat, f"{party}.tax_id", truth[party]["tax_id"])
    for name in ("invoice_date", "due_date"):
        if truth[name]:
            score.found[name] = _date_present(date.fromisoformat(truth[name]), flat)
    for name in ("subtotal", "discount", "tax", "total"):
        if name == "tax" and truth["tax"] is not None:
            continue  # tax is printed as separate CGST/SGST lines; the sum may not appear
        _check_amount(score, nums, name, truth[name])
    for i, item in enumerate(truth["line_items"]):
        _check_amount(score, nums, f"line_items[{i}].amount", item["amount"])
        _check_amount(score, nums, f"line_items[{i}].unit_price", item["unit_price"])

    lines = [numbers_in(line) for line in text.splitlines()]
    for item in truth["line_items"]:
        wanted = {abs(Decimal(item[k])) for k in ("quantity", "unit_price", "amount")}
        score.rows_total += 1
        score.rows_intact += any(wanted <= line for line in lines)
    return score
