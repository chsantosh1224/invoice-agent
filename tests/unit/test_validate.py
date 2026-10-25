from datetime import date
from decimal import Decimal

from invoice_agent.extraction.validate import validate
from invoice_agent.ingest.document import Document, Page, Span
from invoice_agent.ingest.layout import layout_text
from invoice_agent.schemas import Invoice, LineItem, Party

H = 0.012  # typical text height (normalized)


def s(text: str, x0: float, y: float) -> Span:
    """A span of `text` at (x0, y), roughly 0.008 wide per character."""
    return Span(text, x0, y, x0 + 0.008 * len(text), y + H)


def doc(spans: list[Span], kind: str = "digital") -> Document:
    """A one-page document made of `spans`."""
    page = Page(1, 595, 842, spans, layout_text(spans))
    return Document("test.pdf", "light", kind, [page])  # type: ignore[arg-type]


SPANS = [
    s("Acme Supplies", 0.08, 0.05),
    s("GSTIN: 27AAPFU0939F1ZV", 0.08, 0.07),
    s("Invoice No.", 0.60, 0.10), s("INV-42", 0.80, 0.10),
    s("Date", 0.60, 0.12), s("03/04/2026", 0.80, 0.12),
    s("Due Date", 0.60, 0.14), s("03/05/2026", 0.80, 0.14),
    s("Bill To", 0.08, 0.20), s("Globex India", 0.08, 0.215),
    s("Paper", 0.08, 0.30), s("10", 0.50, 0.30), s("450.00", 0.65, 0.30), s("4,500.00", 0.85, 0.30),
    s("Subtotal", 0.65, 0.40), s("₹4,500.00", 0.85, 0.40),
    s("IGST @ 18%", 0.65, 0.42), s("₹810.00", 0.85, 0.42),
    s("Total", 0.65, 0.44), s("₹5,310.00", 0.85, 0.44),
]  # fmt: skip


def invoice(**overrides: object) -> Invoice:
    """The correct extraction of SPANS, with optional field overrides."""
    inv = Invoice(
        invoice_number="INV-42", invoice_date=date(2026, 4, 3), due_date=date(2026, 5, 3),
        currency="INR", vendor=Party(name="Acme Supplies", tax_id="27AAPFU0939F1ZV"),
        buyer=Party(name="Globex India"),
        line_items=[LineItem(description="Paper", quantity=Decimal(10),
                             unit_price=Decimal("450.00"), amount=Decimal("4500.00"))],
        subtotal=Decimal("4500.00"), tax=Decimal("810.00"), total=Decimal("5310.00"),
    )  # fmt: skip
    return inv.model_copy(update=overrides)


def codes(inv: Invoice, kind: str = "digital") -> set[str]:
    """Issue codes the gate raises for `inv` on the SPANS document."""
    return {i.code for i in validate(inv, doc(SPANS, kind)).issues}


def test_correct_extraction_passes_with_high_score() -> None:
    """Everything grounded, reconciled and labelled correctly -> accepted."""
    report = validate(invoice(), doc(SPANS))
    assert report.passed and not report.issues and report.score >= 0.95


def test_same_row_label_beats_label_above() -> None:
    """The due date sits right of 'Due Date' - not under the 'Date' label above it."""
    assert "label.mismatch" not in codes(invoice())


def test_swapped_dates_are_caught_by_labels() -> None:
    """Dates exchanged: each value sits next to the other field's label."""
    swapped = invoice(invoice_date=date(2026, 5, 3), due_date=date(2026, 4, 3))
    assert {"label.mismatch", "format.due_before_invoice"} <= codes(swapped)


def test_total_reading_the_subtotal_is_caught() -> None:
    """Total = 4,500 (the subtotal's value): labels can't tell equal numbers apart, so the
    label check steps aside - and the arithmetic check catches it instead."""
    found = codes(invoice(total=Decimal("4500.00")))
    assert "math.total" in found and "label.mismatch" not in found


def test_vendor_read_from_bill_to_block_is_caught() -> None:
    """A 'vendor' that only appears under 'Bill To' is the buyer."""
    wrong = invoice(vendor=Party(name="Globex India", tax_id="27AAPFU0939F1ZV"),
                    buyer=Party(name="Acme Supplies"))  # fmt: skip
    assert "party.vendor_under_bill_to" in codes(wrong)


def test_gstin_checksum_and_copied_tax_id() -> None:
    """A one-character GSTIN error fails the checksum; a copied tax ID is flagged."""
    assert "format.gstin_checksum" in codes(
        invoice(vendor=Party(name="Acme Supplies", tax_id="27AAPFU0939F1ZW"))
    )
    assert "format.same_tax_id" in codes(
        invoice(buyer=Party(name="Globex India", tax_id="27AAPFU0939F1ZV"))
    )


def test_ungrounded_value_is_an_error_on_digital_but_a_warning_on_scans() -> None:
    """A PO not on the page: digital text is exact (error); OCR may be wrong (warning)."""
    report = validate(invoice(po_number="PO-9999"), doc(SPANS))
    assert any(i.code == "grounding.missing" and i.severity == "error" for i in report.issues)
    scan = validate(invoice(po_number="PO-9999"), doc(SPANS, "scanned"))
    issue = next(i for i in scan.issues if i.field == "po_number")
    assert issue.code == "grounding.ocr_mismatch" and issue.severity == "warning"
    assert scan.field_confidence["po_number"] <= 0.5 and not scan.passed  # low score -> review


def test_assumed_currency_is_sent_to_review() -> None:
    """An ambiguous '$' warning from normalization caps currency confidence below threshold."""
    from invoice_agent.schemas import ValidationIssue

    warning = ValidationIssue(field="currency", code="currency.ambiguous_symbol",
                              message="'$' could be USD, AUD", severity="warning")  # fmt: skip
    report = validate(invoice(currency="USD"), doc(SPANS), normalize_issues=[warning])
    assert report.field_confidence["currency"] == 0.6
    assert not report.passed and not report.errors  # reviewed because of low confidence


def test_shipping_is_part_of_the_total_check() -> None:
    """subtotal - discount + tax + shipping = total passes; without shipping it would fail."""
    from invoice_agent.schemas import Charge

    shipped = invoice(
        other_charges=[Charge(label="Shipping", kind="charge", amount=Decimal("25.00"))],
        total=Decimal("5335.00"),
    )
    assert "math.total" not in codes(shipped)
    assert "math.total" in codes(invoice(total=Decimal("5335.00")))


def test_quotes_and_receipts_are_not_payable() -> None:
    """A quotation is never accepted for payment, however clean the extraction."""
    assert "document.not_payable" in codes(invoice(document_type="quote"))
    assert "document.not_payable" not in codes(invoice(document_type="credit_note"))


def test_line_math_allows_row_discount_and_tax() -> None:
    """10 x 100 less 10% plus 18% tax = 1062.00 is a correct row, not a math.line error."""
    row = LineItem(
        description="Paper",
        quantity=Decimal(10),
        unit_price=Decimal(100),
        amount=Decimal("1062.00"),
        discount_pct=Decimal(10),
        tax_rate=Decimal(18),
    )
    assert "math.line" not in codes(invoice(line_items=[row]))
    wrong = row.model_copy(update={"amount": Decimal("1070.00")})
    assert "math.line" in codes(invoice(line_items=[wrong]))


def test_short_month_date_format_is_grounded() -> None:
    """'Dec 11, 2020' (month abbreviation first) counts as printed - no false warning."""
    from invoice_agent.textmatch import compact, date_present

    assert date_present(date(2020, 12, 11), compact("ORDER DATE Dec 11, 2020, 10:56:14"))


def test_issue_messages_show_readable_values() -> None:
    """Messages show '2026-12-31', never 'datetime.date(2026, 12, 31)'."""
    report = validate(invoice(due_date=date(2026, 12, 31)), doc(SPANS, "scanned"))
    messages = " ".join(i.message for i in report.issues)
    assert "2026-12-31" in messages and "datetime" not in messages
