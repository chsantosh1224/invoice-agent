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
from invoice_agent.schemas import RawInvoice, RawLineItem, RawParty


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234.56", "1234.56"),
        ("₹1,23,456.00", "123456.00"),  # Indian lakh grouping
        ("$ 12,862.00", "12862.00"),
        ("1.234,56 €", "1234.56"),  # EU
        ("1.234.567", "1234567"),
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
    assert parse_amount(text) == Decimal(expected)


@pytest.mark.parametrize("text", ["", "N/A", "--"])
def test_parse_amount_rejects_non_numbers(text: str) -> None:
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
    ],
)
def test_parse_date(text: str, dayfirst: bool | None, expected: date) -> None:
    assert parse_date(text, dayfirst=dayfirst) == expected


def test_ambiguous_date_without_hint_is_an_error_not_a_guess() -> None:
    with pytest.raises(AmbiguousDateError):
        parse_date("03/04/2026")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("₹", "INR"), ("Rs.", "INR"), ("INR", "INR"), ("€", "EUR"), ("usd", "USD"), ("CAD $", "CAD")],
)
def test_parse_currency(text: str, expected: str) -> None:
    assert parse_currency(text) == expected


def _raw(**overrides: object) -> RawInvoice:
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
        tax="810.00",
        total="₹5,310.00",
        payment_terms=None,
    )
    base.update(overrides)
    return RawInvoice.model_validate(base)


def test_to_invoice_uses_currency_to_resolve_date_order() -> None:
    invoice, issues = to_invoice(_raw())
    assert issues == []
    assert invoice.currency == "INR"
    assert invoice.invoice_date == date(2026, 4, 3)  # INR -> day first
    assert invoice.total == Decimal("5310.00")
    assert invoice.line_items[0].amount == Decimal("4500.00")


def test_to_invoice_reports_issues_instead_of_guessing() -> None:
    invoice, issues = to_invoice(_raw(currency=None, total="TBD"))
    codes = {(i.field, i.code) for i in issues}
    assert ("invoice_date", "normalize.date_ambiguous") in codes
    assert ("total", "normalize.amount") in codes
    assert invoice.invoice_date is None and invoice.total is None
