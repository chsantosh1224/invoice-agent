from datetime import date
from decimal import Decimal

from invoice_agent.evals.extraction import score_extraction, similar, values_match
from invoice_agent.extraction.messages import build_messages
from invoice_agent.schemas import Invoice, LineItem, Party

TRUTH = {
    "document_type": "invoice",
    "invoice_number": "INV-2026-0042",
    "invoice_date": "2026-04-03",
    "due_date": "2026-05-03",
    "po_number": None,
    "currency": "INR",
    "vendor": {"name": "Acme Supplies Pvt Ltd", "address": "x", "tax_id": "29ABCDE1234F1Z5"},
    "buyer": {"name": "Globex India", "address": "y", "tax_id": None},
    "line_items": [
        {"description": "A4 paper (box)", "quantity": "10", "unit_price": "450.00",
         "amount": "4500.00"},
        {"description": "Toner cartridge", "quantity": "2", "unit_price": "3200.00",
         "amount": "6400.00"},
    ],
    "subtotal": "10900.00",
    "discount": None,
    "tax": "1962.00",
    "total": "12862.00",
    "payment_terms": "Net 30",
}  # fmt: skip


def perfect() -> Invoice:
    """An Invoice equal to TRUTH."""
    return Invoice(
        invoice_number="INV-2026-0042", invoice_date=date(2026, 4, 3),
        due_date=date(2026, 5, 3), currency="INR",
        vendor=Party(name="ACME SUPPLIES PVT LTD", tax_id="29ABCDE1234F1Z5"),
        buyer=Party(name="Globex India"),
        line_items=[
            LineItem(description="A4 paper (box)", quantity=Decimal(10),
                     unit_price=Decimal("450.00"), amount=Decimal("4500.00")),
            LineItem(description="Toner cartridge", quantity=Decimal(2),
                     unit_price=Decimal("3200"), amount=Decimal("6400")),
        ],
        subtotal=Decimal("10900"), tax=Decimal("1962"), total=Decimal("12862.00"),
        payment_terms="Net 30",
    )  # fmt: skip


def test_perfect_extraction_scores_100() -> None:
    """Formatting differences (case, trailing zeros) don't count as errors."""
    s = score_extraction(perfect(), TRUTH)
    assert s.field_accuracy == 1.0 and s.items_f1 == 1.0
    assert s.descriptions_ok == 2 and not s.hallucinated and not s.missed


def test_the_smoke_test_errors_are_caught() -> None:
    """The real dots-3 mistakes: wrong due date, invented discount, copied GSTIN."""
    p = perfect()
    p.due_date = date(2026, 3, 5)
    p.discount = Decimal("0.00")
    p.buyer.tax_id = "29ABCDE1234F1Z5"
    s = score_extraction(p, TRUTH)
    assert not s.fields["due_date"]
    assert set(s.hallucinated) == {"discount", "buyer.tax_id"}


def test_line_items_need_all_three_numbers() -> None:
    """A row with a misread amount is not a match; F1 reflects it."""
    p = perfect()
    p.line_items[1].amount = Decimal("6400.01")
    s = score_extraction(p, TRUTH)
    assert s.items_matched == 1 and s.items_f1 == 0.5


def test_value_rules() -> None:
    """IDs ignore punctuation, amounts are exact, names are fuzzy."""
    assert values_match("invoice_number", "inv 2026/0042", "INV-2026-0042")
    assert not values_match("total", Decimal("12862.01"), "12862.00")
    assert similar("Acme Supplies Pvt. Ltd.", "ACME SUPPLIES PVT LTD")
    assert not similar("Acme Supplies", "Globex India")


def test_messages_include_requested_inputs_only() -> None:
    """Text-only requests carry no image; image+text requests carry both."""
    from PIL import Image

    text_only = build_messages(text="INVOICE 42")
    assert all(b["type"] == "text" for b in text_only[1].content)  # type: ignore[union-attr]
    both = build_messages(images=[Image.new("RGB", (3000, 4000), "white")], text="INVOICE 42")
    kinds = [b["type"] for b in both[1].content]  # type: ignore[union-attr]
    assert kinds == ["text", "image_url", "text"]
