from datetime import date
from decimal import Decimal

import pytest

from invoice_agent.normalize import (
    AmbiguousDateError,
    parse_amount,
    parse_currency,
    parse_date,
    to_invoice,
)
from invoice_agent.schemas import RawInvoice, RawLineItem, RawParty, RawTaxLine


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234.56", "1234.56"),
        ("₹1,23,456.00", "123456.00"),  # Indian lakh grouping
        ("$ 12,862.00", "12862.00"),
        ("1.234,56 €", "1234.56"),  # EU
        ("1.234.567", "1234567"),
        ("10.476.38", "10476.38"),  # OCR misread comma as dot (seen on a real scan)
        ("1 234,56", "1234.56"),
        ("12,5", "12.5"),
        ("(500.00)", "-500.00"),  # accounting negative
        ("-1,200.00", "-1200.00"),
        ("250.00 CR", "-250.00"),
        ("10", "10"),
        ("Rs. 4,500/-", "4500"),
    ],
)
def test_parse_amount(text: str, expected: str) -> None:
    """Each printed amount format (lakh, EU comma, negatives, '/-') parses exactly."""
    assert parse_amount(text) == Decimal(expected)


@pytest.mark.parametrize("text", ["", "N/A", "--"])
def test_parse_amount_rejects_non_numbers(text: str) -> None:
    """Text with no digits raises ValueError instead of becoming 0."""
    with pytest.raises(ValueError):
        parse_amount(text)


@pytest.mark.parametrize(
    ("text", "dayfirst", "expected"),
    [
        ("2026-04-03", None, date(2026, 4, 3)),
        ("03/04/2026", True, date(2026, 4, 3)),
        ("03/04/2026", False, date(2026, 3, 4)),
        ("25/04/2026", None, date(2026, 4, 25)),  # unambiguous: 25 can't be a month
        ("04/25/2026", None, date(2026, 4, 25)),
        ("03.04.26", True, date(2026, 4, 3)),
        ("3 Apr 2026", None, date(2026, 4, 3)),
        ("April 3rd, 2026", None, date(2026, 4, 3)),
        ("03-Apr-2026", None, date(2026, 4, 3)),
        ("12 Sept 2026", None, date(2026, 9, 12)),
        ("September 13, 2026", None, date(2026, 9, 13)),  # regression: "Sept" fix
    ],
)
def test_parse_date(text: str, dayfirst: bool | None, expected: date) -> None:
    """Supported date formats parse correctly, using `dayfirst` only when the date is ambiguous."""
    assert parse_date(text, dayfirst=dayfirst) == expected


def test_ambiguous_date_without_hint_is_an_error_not_a_guess() -> None:
    """'03/04/2026' with no hint must raise, never silently pick DD/MM or MM/DD."""
    with pytest.raises(AmbiguousDateError):
        parse_date("03/04/2026")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("₹", "INR"), ("Rs.", "INR"), ("INR", "INR"), ("€", "EUR"), ("usd", "USD"), ("CAD $", "CAD")],
)
def test_parse_currency(text: str, expected: str) -> None:
    """Symbols and codes map to ISO 4217; an explicit code beats a symbol ('CAD $' -> CAD)."""
    assert parse_currency(text) == expected


def _raw(**overrides: object) -> RawInvoice:
    """Build a valid RawInvoice for tests.

    Args:
        **overrides: Field values to replace in the default invoice.

    Returns:
        A RawInvoice with the defaults below, updated with `overrides`.
    """
    party = RawParty(name=None, address=None, tax_id=None)
    base: dict[str, object] = dict(
        document_type="invoice",
        invoice_number="INV-1",
        invoice_date="03/04/2026",
        due_date="03/05/2026",
        po_number=None,
        currency="₹",
        vendor=party,
        buyer=party,
        line_items=[
            RawLineItem(description="Paper", quantity="10", unit_price="450.00", amount="4,500.00")
        ],
        subtotal="4,500.00",
        discount=None,
        tax_lines=[
            RawTaxLine(label="CGST @ 9%", amount="405.00"),
            RawTaxLine(label="SGST @ 9%", amount="₹405.00"),
        ],
        total="₹5,310.00",
        payment_terms=None,
    )
    base.update(overrides)
    return RawInvoice.model_validate(base)


def test_to_invoice_uses_currency_to_resolve_date_order() -> None:
    """₹ means day-first, so '03/04/2026' becomes 3 April; amounts become Decimals."""
    invoice, issues = to_invoice(_raw())
    assert issues == []
    assert invoice.currency == "INR"
    assert invoice.invoice_date == date(2026, 4, 3)  # INR -> day first
    assert invoice.total == Decimal("5310.00")
    assert invoice.tax == Decimal("810.00")  # CGST + SGST summed in code, not by the LLM
    assert invoice.line_items[0].amount == Decimal("4500.00")


def test_to_invoice_reports_issues_instead_of_guessing() -> None:
    """Unparseable values become None plus a ValidationIssue, never a guess."""
    invoice, issues = to_invoice(_raw(currency=None, total="TBD"))
    codes = {(i.field, i.code) for i in issues}
    assert ("invoice_date", "normalize.date_ambiguous") in codes
    assert ("total", "normalize.amount") in codes
    assert invoice.invoice_date is None and invoice.total is None


def test_digit_free_tax_id_becomes_null() -> None:
    """'GSTIN: Unregistered' must not become a tax ID (the Day 3 bake-off's only error)."""
    unregistered = RawParty(name="Taneja Inc", address=None, tax_id="Unregistered")
    invoice, _ = to_invoice(_raw(buyer=unregistered))
    assert invoice.buyer.tax_id is None


def test_zero_or_negative_discount_is_normalized() -> None:
    """'-500.00' and '(500.00)' are a 500 discount; an invented '0.00' is no discount."""
    assert to_invoice(_raw(discount="(500.00)"))[0].discount == Decimal("500.00")
    assert to_invoice(_raw(discount="0.00"))[0].discount is None


def test_currency_is_taken_from_amounts_when_the_model_left_it_empty() -> None:
    """The Sliced Invoices sample: no currency line, but '$93.50' is printed on the totals."""
    invoice, issues = to_invoice(_raw(currency=None, total="₹5,310.00"))
    assert invoice.currency == "INR"
    assert [i.code for i in issues] == ["currency.inferred"]


def test_bare_dollar_is_assumed_usd_but_flagged() -> None:
    """'$' could be USD, AUD, CAD...: USD is assumed and reported, never silently."""
    raw = _raw(currency=None, total="$93.50", subtotal="$85.00", tax_lines=[], line_items=[])
    invoice, issues = to_invoice(raw)
    assert invoice.currency == "USD"
    codes = {i.code: i for i in issues}
    assert (
        "currency.ambiguous_symbol" in codes and "AUD" in codes["currency.ambiguous_symbol"].message
    )


@pytest.mark.parametrize(
    ("printed", "code"),
    [
        ("A$ 93.50", "AUD"),
        ("AU$93.50", "AUD"),
        ("CA$ 12", "CAD"),
        ("US$ 5", "USD"),
        ("S$ 9", "SGD"),
    ],
)
def test_prefixed_dollars_are_unambiguous(printed: str, code: str) -> None:
    """Country-prefixed dollars resolve to the right currency with no ambiguity warning."""
    invoice, issues = to_invoice(_raw(currency=printed))
    assert invoice.currency == code
    assert not [i for i in issues if i.code == "currency.ambiguous_symbol"]


def test_conflicting_currency_symbols_are_an_error() -> None:
    """Amounts printed in two currencies cannot be resolved; it is an error, not a pick."""
    _, issues = to_invoice(_raw(currency=None, total="$100.00", subtotal="€90.00"))
    assert any(i.code == "currency.conflicting_symbols" and i.severity == "error" for i in issues)
