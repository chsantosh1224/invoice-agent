"""Score ingestion engines against the synthetic truth, without any LLM.

Two questions, per document:
1. Value recall: does each truth value appear somewhere in the engine's text?
   (If it doesn't, no extractor downstream can get it right from the text.)
2. Row integrity: does each line item's qty, unit price and amount appear together on one
   line? (If not, a text-only model has to guess which numbers belong together.)

Matching is done on normalized values (amounts parsed to Decimal, dates tried in every
known print format), so engines are not penalized for spacing or symbol differences.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from invoice_agent.normalize import parse_amount
from invoice_agent.synth.formatting import DATE_FORMATS

_NUMBER = re.compile(r"\d[\d.,]*\d|\d")
_SPACES = re.compile(r"\s+")


def numbers_in(text: str) -> set[Decimal]:
    """Find every number-like token in a text and parse it as an amount.

    Args:
        text: Any text (a whole document or a single line).

    Returns:
        The absolute values of every token that parses (unparseable tokens are skipped).
    """
    found = set()
    for token in _NUMBER.findall(text):
        try:
            found.add(abs(parse_amount(token)))
        except ValueError:
            continue
    return found


def _compact(text: str) -> str:
    """Lowercase and remove all whitespace, for spacing-insensitive containment checks.

    Args:
        text: Any text.

    Returns:
        The text lowercased with all whitespace removed.
    """
    return _SPACES.sub("", text).lower()


def _date_present(value: date, text: str) -> bool:
    """Check whether a date appears in the text in any supported print format.

    Args:
        value: The truth date.
        text: Compacted document text (see `_compact`).

    Returns:
        True if any format of the date (e.g. '03/04/2026', '3 Apr 2026') is present.
    """
    return any(_compact(fmt(value)) in text for fmt in DATE_FORMATS.values())


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

    def check_amount(name: str, value: str | None) -> None:
        """Record whether an amount is present (skipped when the truth is null)."""
        if value is not None:
            score.found[name] = abs(Decimal(value)) in nums

    def check_text(name: str, value: str | None) -> None:
        """Record whether a string value is present (spacing/case-insensitive)."""
        if value:
            score.found[name] = _compact(value) in flat

    for name in ("invoice_number", "po_number"):
        check_text(name, truth[name])
    for party in ("vendor", "buyer"):
        check_text(f"{party}.name", truth[party]["name"])
        check_text(f"{party}.tax_id", truth[party]["tax_id"])
    for name in ("invoice_date", "due_date"):
        if truth[name]:
            score.found[name] = _date_present(date.fromisoformat(truth[name]), flat)
    for name in ("subtotal", "discount", "tax", "total"):
        if name == "tax" and truth["tax"] is not None:
            continue  # tax is printed as separate CGST/SGST lines; the sum may not appear
        check_amount(name, truth[name])
    for i, item in enumerate(truth["line_items"]):
        check_amount(f"line_items[{i}].amount", item["amount"])
        check_amount(f"line_items[{i}].unit_price", item["unit_price"])

    lines = [numbers_in(line) for line in text.splitlines()]
    for item in truth["line_items"]:
        wanted = {abs(Decimal(item[k])) for k in ("quantity", "unit_price", "amount")}
        score.rows_total += 1
        score.rows_intact += any(wanted <= line for line in lines)
    return score
