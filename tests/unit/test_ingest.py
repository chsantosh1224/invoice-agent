import math
from decimal import Decimal

from invoice_agent.evals.ingestion import numbers_in, score_document
from invoice_agent.ingest.document import Span
from invoice_agent.ingest.layout import deskew_quad, estimate_skew, group_rows, layout_text

TRUTH = {
    "invoice_number": "INV-2026-0042",
    "po_number": None,
    "invoice_date": "2026-04-03",
    "due_date": None,
    "vendor": {"name": "Acme Supplies", "tax_id": "29ABCDE1234F1Z5"},
    "buyer": {"name": "Globex", "tax_id": None},
    "subtotal": "10900.00",
    "discount": None,
    "tax": None,
    "total": "10900.00",
    "line_items": [
        {"quantity": "10", "unit_price": "450.00", "amount": "4500.00"},
        {"quantity": "2", "unit_price": "3200.00", "amount": "6400.00"},
    ],
}


def span(text: str, x0: float, y0: float, x1: float | None = None, h: float = 0.01) -> Span:
    """A span with sensible defaults for layout tests."""
    return Span(text, x0, y0, x1 if x1 is not None else x0 + 0.01 * len(text), y0 + h)


def test_rows_group_by_vertical_position_and_sort_left_to_right() -> None:
    """Spans on the same printed line form one row, ordered by x."""
    spans = [span("Amount", 0.8, 0.100), span("Qty", 0.5, 0.102), span("Item", 0.1, 0.099),
             span("Paper", 0.1, 0.130)]  # fmt: skip
    rows = group_rows(spans)
    assert [[s.text for s in r] for r in rows] == [["Item", "Qty", "Amount"], ["Paper"]]


def test_layout_text_keeps_columns_aligned() -> None:
    """Values in the same column start at the same character position."""
    spans = [span("Qty", 0.5, 0.10), span("10", 0.5, 0.12), span("2", 0.5, 0.14)]
    lines = layout_text(spans, columns=100).splitlines()
    assert {line.index(line.strip()) for line in lines} == {50}


def test_skew_is_estimated_and_removed() -> None:
    """A text line rotated by 2 degrees is detected and straightened."""
    angle = math.radians(2)
    x0, y0, w, h = 100.0, 500.0, 600.0, 20.0
    dx, dy = w * math.cos(angle), w * math.sin(angle)
    quad = [(x0, y0), (x0 + dx, y0 + dy), (x0 + dx, y0 + dy + h), (x0, y0 + h)]
    est = estimate_skew([quad])
    assert abs(est - angle) < 1e-9
    _bx0, by0, _bx1, by1 = deskew_quad(quad, est, 400, 400)
    assert by1 - by0 < h + 2  # flat again: height ~= text height, not text + 21 px slope


def test_numbers_in_parses_regional_formats() -> None:
    """Indian, EU and plain numbers in text are all recovered as exact Decimals."""
    assert {Decimal("166460.00"), Decimal("1319.20"), Decimal(7)} <= numbers_in(
        "Total 1,66,460.00 | Netto 1.319,20 € | Net 7"
    )


def test_score_counts_found_values_and_intact_rows() -> None:
    """Values present anywhere count as found; a row is intact only if on one line."""
    text = (
        "ACME SUPPLIES  GSTIN 29ABCDE1234F1Z5\nInvoice No INV-2026-0042  Date 03/04/2026\n"
        "Bill to Globex\nPaper 10 450.00 4,500.00\nToner 2 3,200.00\n6,400.00\nTotal 10,900.00"
    )
    s = score_document("d1", text, TRUTH)
    assert s.recall == 1.0
    assert (s.rows_intact, s.rows_total) == (1, 2)  # toner amount wrapped to the next line


def test_missing_value_lowers_recall() -> None:
    """A value absent from the text (e.g. OCR dropped it) is reported as not found."""
    s = score_document("d1", "Invoice INV-2026-0042", TRUTH)
    assert s.found["invoice_number"] and not s.found["total"]
    assert s.recall < 0.5
