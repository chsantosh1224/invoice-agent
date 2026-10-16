"""Score an extracted Invoice against ground truth, field by field.

Comparison happens on normalized values (Decimal, date, ISO currency), so a model is never
penalized for formatting - only for getting the meaning wrong. Missing (null) vs present is
tracked separately: a value where the truth is null is a *hallucination*, the most
dangerous error in accounts payable.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any

from invoice_agent.schemas import Invoice

SCALAR_FIELDS = (
    "document_type", "invoice_number", "invoice_date", "due_date", "po_number", "currency",
    "subtotal", "discount", "tax", "total", "payment_terms",
)  # fmt: skip
PARTY_FIELDS = ("name", "tax_id")
FUZZY_FIELDS = {"vendor.name", "buyer.name", "payment_terms"}


def _norm_text(value: str) -> str:
    """Normalize free text for comparison: lowercase, collapse spaces, drop punctuation.

    Args:
        value: Any string.

    Returns:
        Lowercased alphanumerics separated by single spaces.
    """
    return " ".join("".join(c if c.isalnum() else " " for c in value.lower()).split())


def _compact(value: str) -> str:
    """Normalize identifiers: uppercase with all non-alphanumerics removed.

    Args:
        value: An ID such as 'INV-2026/0042' or 'DE 123 456 789'.

    Returns:
        e.g. 'INV20260042' / 'DE123456789'.
    """
    return "".join(c for c in value.upper() if c.isalnum())


def similar(a: str, b: str, *, threshold: float = 0.9) -> bool:
    """Fuzzy text match, tolerant of small OCR or punctuation differences.

    Args:
        a: First text.
        b: Second text.
        threshold: Minimum similarity ratio (0..1) after normalization.

    Returns:
        True if the normalized texts are equal, one contains the other, or their
        similarity ratio reaches the threshold.
    """
    x, y = _norm_text(a), _norm_text(b)
    if not x or not y:
        return x == y
    return x == y or x in y or y in x or SequenceMatcher(None, x, y).ratio() >= threshold


def values_match(name: str, pred: Any, truth: Any) -> bool:
    """Decide whether one predicted field value equals the truth.

    Args:
        name: Field path, e.g. 'total' or 'vendor.name' (decides the comparison rule).
        pred: The extracted, normalized value (or None).
        truth: The truth value from .truth.json (JSON types: str/None).

    Returns:
        True if both are null, or both present and equal under the field's rule:
        amounts and dates exact, IDs ignoring punctuation, names/terms fuzzy.
    """
    if truth is None or pred is None:
        return truth is None and pred is None
    if isinstance(pred, Decimal):
        return bool(pred == Decimal(str(truth)))
    if isinstance(pred, date):
        return pred.isoformat() == str(truth)
    if name in FUZZY_FIELDS:
        return similar(str(pred), str(truth))
    return _compact(str(pred)) == _compact(str(truth))


@dataclass
class ExtractionScore:
    """How one extraction compares to its truth.

    Attributes:
        fields: Field path -> correct?
        hallucinated: Fields where the model returned a value but the truth is null.
        missed: Fields where the truth has a value but the model returned null.
        items_truth: Number of truth line items.
        items_pred: Number of predicted line items.
        items_matched: Predicted items whose qty, unit price and amount all match a truth
            item (one-to-one).
        descriptions_ok: Matched items whose description also matches.
    """

    fields: dict[str, bool] = field(default_factory=dict)
    hallucinated: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    items_truth: int = 0
    items_pred: int = 0
    items_matched: int = 0
    descriptions_ok: int = 0

    @property
    def field_accuracy(self) -> float:
        """Share of header fields extracted correctly.

        Returns:
            correct / scored, or 0.0 if nothing was scored.
        """
        return sum(self.fields.values()) / len(self.fields) if self.fields else 0.0

    @property
    def items_f1(self) -> float:
        """F1 of line items (precision: predicted items that are real; recall: real items found).

        Returns:
            2 * matched / (truth + predicted), or 1.0 when both are empty.
        """
        denom = self.items_truth + self.items_pred
        return 2 * self.items_matched / denom if denom else 1.0


def score_extraction(pred: Invoice, truth: dict[str, Any]) -> ExtractionScore:
    """Compare an extracted Invoice with the truth invoice.

    Args:
        pred: The normalized extraction result.
        truth: The 'invoice' object from a .truth.json file.

    Returns:
        An ExtractionScore with per-field correctness, hallucinations, misses and
        line-item matching.
    """
    score = ExtractionScore()
    pairs: list[tuple[str, Any, Any]] = [(f, getattr(pred, f), truth[f]) for f in SCALAR_FIELDS]
    for party in ("vendor", "buyer"):
        for f in PARTY_FIELDS:
            pairs.append((f"{party}.{f}", getattr(getattr(pred, party), f), truth[party][f]))
    for name, p, t in pairs:
        score.fields[name] = values_match(name, p, t)
        if t is None and p is not None:
            score.hallucinated.append(name)
        elif t is not None and p is None:
            score.missed.append(name)

    truth_items = list(truth["line_items"])
    score.items_truth, score.items_pred = len(truth_items), len(pred.line_items)
    for item in pred.line_items:
        for i, t in enumerate(truth_items):
            if all(
                values_match(k, getattr(item, k), t[k])
                for k in ("quantity", "unit_price", "amount")
            ):
                score.items_matched += 1
                score.descriptions_ok += bool(
                    item.description and similar(item.description, t["description"], threshold=0.8)
                )
                truth_items.pop(i)
                break
    return score
