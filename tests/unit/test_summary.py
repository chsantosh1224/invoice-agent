from decimal import Decimal

from invoice_agent.schemas import SummaryKind
from invoice_agent.summary import SummaryRow, Totals, resolve_summary


def rows(*spec: tuple[str, SummaryKind, str]) -> list[SummaryRow]:
    """Build summary rows from (label, kind, amount) triples, top to bottom."""
    return [
        SummaryRow(i, label, kind, Decimal(amount)) for i, (label, kind, amount) in enumerate(spec)
    ]


def codes(t: Totals) -> list[str]:
    """Issue codes of a resolved block."""
    return [i.code for i in t.issues]


def test_magento_invoice_with_shipping_reconciles() -> None:
    """The real sample: 141.00 - 14.10 + 10.47 + 25.00 shipping = 162.37."""
    t = resolve_summary(rows(
        ("SUBTOTAL", "subtotal", "141.00"),
        ("DISCOUNT (EYHPAOMMT9O9FXDH)", "discount", "-14.10"),
        ("TAX", "tax", "10.47"),
        ("SHIPPING & HANDLING", "charge", "25.00"),
        ("GRAND TOTAL", "total", "162.37"),
    ), lines_sum=Decimal("141.00"))  # fmt: skip
    assert (t.subtotal, t.discount, t.tax, t.total) == (
        Decimal("141.00"), Decimal("14.10"), Decimal("10.47"), Decimal("162.37"))  # fmt: skip
    assert [(c.label, c.kind, c.amount) for c in t.charges] == [
        ("SHIPPING & HANDLING", "charge", Decimal("25.00"))]  # fmt: skip
    assert codes(t) == []


def test_unknown_row_is_classified_by_the_arithmetic() -> None:
    """A label nobody planned for: 'other' + only '+' adds up -> charge, flagged."""
    t = resolve_summary(rows(
        ("Subtotal", "subtotal", "100.00"),
        ("Environmental recovery fee", "other", "3.00"),
        ("Total", "total", "103.00"),
    ), lines_sum=None)  # fmt: skip
    assert t.charges[0].kind == "charge" and t.charges[0].amount == Decimal("3.00")
    assert codes(t) == ["summary.kind_inferred"]


def test_unknown_rows_that_fit_two_ways_stay_unclassified() -> None:
    """+5 -5 and -5 +5 both add up: the code refuses to pick, a person decides."""
    t = resolve_summary(rows(
        ("Subtotal", "subtotal", "100.00"),
        ("Item A", "other", "5.00"),
        ("Item B", "other", "5.00"),
        ("Total", "total", "100.00"),
    ), lines_sum=None)  # fmt: skip
    assert codes(t) == ["summary.unclassified", "summary.unclassified"]
    assert all(i.severity == "error" for i in t.issues)


def test_deduction_round_off_and_included_tax() -> None:
    """Advance paid reduces; round-off keeps its sign; 'incl. VAT' is not added."""
    t = resolve_summary(rows(
        ("Net amount", "subtotal", "1000.00"),
        ("incl. VAT 19%", "tax_included", "159.66"),
        ("Less: advance paid", "deduction", "200.00"),
        ("Round off", "adjustment", "-0.40"),
        ("Net payable", "total", "799.60"),
    ), lines_sum=None)  # fmt: skip
    assert t.tax is None and t.tax_included == Decimal("159.66")
    assert [c.amount for c in t.charges] == [Decimal("-200.00"), Decimal("-0.40")]
    assert codes(t) == []


def test_one_wrong_sign_is_corrected_and_flagged() -> None:
    """Model called a credit a 'charge'; only the opposite sign adds up -> fixed + flagged."""
    t = resolve_summary(rows(
        ("Subtotal", "subtotal", "100.00"),
        ("Loyalty credit", "charge", "10.00"),
        ("Total", "total", "90.00"),
    ), lines_sum=None)  # fmt: skip
    assert t.charges[0].amount == Decimal("-10.00")
    assert codes(t) == ["summary.sign_corrected"]


def test_without_subtotal_the_line_items_are_the_base() -> None:
    """No subtotal row printed: the line-item sum is used to check the block."""
    t = resolve_summary(rows(
        ("Freight", "charge", "15.00"),
        ("Amount due", "total", "115.00"),
    ), lines_sum=Decimal("100.00"))  # fmt: skip
    assert t.subtotal is None and codes(t) == []


def test_last_total_row_is_the_amount_payable() -> None:
    """'Total' then 'Less deposit' then 'Balance due': the balance is what gets paid."""
    t = resolve_summary(rows(
        ("Total", "total", "166.00"),
        ("Less deposit", "deduction", "50.00"),
        ("Balance due", "total", "116.00"),
    ), lines_sum=Decimal("166.00"))  # fmt: skip
    assert t.total == Decimal("116.00") and codes(t) == []
