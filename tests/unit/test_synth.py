import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import pytest
from faker import Faker

from invoice_agent.normalize import DAYFIRST_CURRENCIES, parse_amount, parse_date, to_invoice
from invoice_agent.schemas import RawInvoice, RawLineItem, RawParty, RawSummaryLine
from invoice_agent.synth.build import PROFILES, Country, InvoiceSpec, build_invoice
from invoice_agent.synth.formatting import (
    format_date,
    format_money,
    format_number,
    format_quantity,
)
from invoice_agent.synth.generate import plan_documents
from invoice_agent.synth.render import TEMPLATES, render_html
from invoice_agent.tax_ids import gstin_check_char, is_valid_gstin

COUNTRIES: list[Country] = ["IN", "US", "EU", "UK"]


def spec_for(country: Country, seed: int, multi_page: bool = False) -> InvoiceSpec:
    """Build a reproducible synthetic invoice for tests."""
    fake = Faker(PROFILES[country].faker_locale)
    fake.seed_instance(seed)
    return build_invoice(random.Random(seed), fake, country=country, multi_page=multi_page)


# ---------- tax IDs ----------


def test_gstin_checksum_matches_a_real_gstin() -> None:
    """The mod-36 checksum reproduces the check character of a published GSTIN."""
    assert gstin_check_char("27AAPFU0939F1Z") == "V"
    assert is_valid_gstin("27AAPFU0939F1ZV")


def test_gstin_single_char_typo_is_detected() -> None:
    """One misread character (e.g. OCR 0 -> O) invalidates the GSTIN."""
    assert not is_valid_gstin("27AAPFU0939F1ZW")
    assert not is_valid_gstin("27AAPFUO939F1ZV")


# ---------- formatting round trips through normalize.py ----------


@pytest.mark.parametrize(
    ("value", "style", "expected"),
    [
        ("12345678.90", "indian", "1,23,45,678.90"),
        ("12345678.90", "us", "12,345,678.90"),
        ("12345678.90", "eu", "12.345.678,90"),
        ("1234.5", "space", "1 234,50"),
        ("999.00", "indian", "999.00"),
    ],
)
def test_format_number(value: str, style: str, expected: str) -> None:
    """Each regional style prints the separators real invoices use."""
    assert format_number(Decimal(value), style) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("style", ["us", "indian", "eu", "space"])
def test_every_printed_amount_parses_back_exactly(style: str) -> None:
    """Whatever the generator prints, normalize.parse_amount reads back to the exact value."""
    rng = random.Random(7)
    for _ in range(300):
        value = Decimal(rng.randint(0, 10**9)) / 100
        for symbol, suffix in (("₹", False), ("Rs.", False), ("$", False), ("€", True)):
            printed = format_money(value, style, symbol, suffix=suffix)  # type: ignore[arg-type]
            assert parse_amount(printed) == value, printed
    for qty in ("1", "7.5", "12.25", "40"):
        assert parse_amount(format_quantity(Decimal(qty), style)) == Decimal(qty)  # type: ignore[arg-type]


@pytest.mark.parametrize("country", COUNTRIES)
def test_every_printed_date_parses_back_with_currency_hint(country: Country) -> None:
    """Including ambiguous days (<= 12): the currency decides DD/MM vs MM/DD correctly."""
    p = PROFILES[country]
    dayfirst = p.currency in DAYFIRST_CURRENCIES
    for d in (date(2026, 1, 1) + timedelta(days=n) for n in range(0, 365, 3)):
        for fmt in p.date_formats:
            assert parse_date(format_date(d, fmt), dayfirst=dayfirst) == d, (fmt, d)


# ---------- invoice arithmetic ----------


@pytest.mark.parametrize("country", COUNTRIES)
def test_generated_invoices_always_reconcile(country: Country) -> None:
    """Line math, subtotal and total = subtotal - discount + tax hold for every invoice."""
    for seed in range(60):
        inv = spec_for(country, seed, multi_page=seed % 10 == 0).truth
        for li in inv.line_items:
            assert li.amount == (li.quantity * li.unit_price).quantize(
                Decimal("0.01"), ROUND_HALF_UP
            )  # type: ignore[operator]
        assert inv.subtotal == sum(li.amount for li in inv.line_items)  # type: ignore[misc]
        assert inv.total == inv.subtotal - (inv.discount or 0) + (inv.tax or 0)  # type: ignore[operator]
        assert inv.currency == PROFILES[country].currency
        if inv.due_date:
            assert inv.due_date > inv.invoice_date  # type: ignore[operator]
        if country == "IN":
            assert inv.vendor.tax_id and is_valid_gstin(inv.vendor.tax_id)


def test_edge_cases_all_occur() -> None:
    """Across a modest sample, every edge case the eval relies on is generated."""
    seen = {e for seed in range(80) for e in spec_for("IN", seed).edge_cases}
    assert {"missing_po", "discount", "freight_line", "credit_note", "due_on_receipt",
            "buyer_without_tax_id"} <= seen  # fmt: skip


def summary_rows(v: dict[str, Any]) -> list[RawSummaryLine]:
    """Totals-block rows a perfect model would transcribe from a generated view."""
    rows = [RawSummaryLine(label="Subtotal", amount=v["subtotal"], kind="subtotal")]
    if v["discount"]:
        rows.append(RawSummaryLine(label="Discount", amount=v["discount"], kind="discount"))
    rows += [RawSummaryLine(label=t["label"], amount=t["amount"], kind="tax") for t in v["taxes"]]
    rows.append(RawSummaryLine(label="Total", amount=v["total"], kind="total"))
    return rows


@pytest.mark.parametrize("country", COUNTRIES)
def test_perfect_transcription_normalizes_to_truth(country: Country) -> None:
    """If an LLM copied every printed value perfectly, normalize.py yields the exact truth.

    This proves the truth files are reachable: any eval error is the extractor's, not the
    dataset's.
    """
    for seed in range(25):
        spec = spec_for(country, seed)
        v, truth = spec.view, spec.truth
        raw = RawInvoice(
            document_type=truth.document_type,
            invoice_number=v["invoice_number"],
            invoice_date=v["invoice_date"],
            due_date=v["due_date"],
            po_number=v["po_number"],
            currency=v["total"],  # symbol as printed next to the total
            vendor=RawParty(
                name=v["vendor"]["name"],
                address=", ".join(v["vendor"]["address_lines"]),
                tax_id=v["vendor"]["tax_id"],
            ),
            buyer=RawParty(
                name=v["buyer"]["name"],
                address=", ".join(v["buyer"]["address_lines"]),
                tax_id=v["buyer"]["tax_id"],
            ),
            line_items=[
                RawLineItem(
                    description=i["description"],
                    quantity=i["qty"],
                    unit_price=i["unit_price"],
                    amount=i["amount"],
                    discount=None,
                    tax_rate=None,
                    tax_amount=None,
                )
                for i in v["items"]
            ],
            summary_lines=summary_rows(v),
            payment_terms=v["payment_terms"],
        )
        invoice, issues = to_invoice(raw)
        # A bare "$" is reported as an assumption (USD); nothing else may be flagged.
        assert {i.code for i in issues} <= {"currency.ambiguous_symbol"}, issues
        assert invoice == truth


# ---------- templates and dataset plan ----------


@pytest.mark.parametrize("template", TEMPLATES)
def test_every_template_renders_every_value(template: str) -> None:
    """Each template prints the invoice number, dates, line amounts and total."""
    spec = spec_for("IN", 3)
    html = render_html(template, spec.view)
    for needle in (spec.view["invoice_number"], spec.view["invoice_date"], spec.view["total"]):
        assert needle in html
    assert all(item["amount"] in html for item in spec.view["items"])


def test_dataset_plan_is_reproducible_and_balanced() -> None:
    """Same seed -> same plan; all templates used; split and scan counts as requested."""
    a = plan_documents(36, 24, seed=42, test_fraction=0.4)
    assert a == plan_documents(36, 24, seed=42, test_fraction=0.4)
    assert {p["template"] for p in a} == set(TEMPLATES)
    assert sum(p["split"] == "test" for p in a) == 14
    assert sum(p["scan"] is not None for p in a) == 24
    assert not any(p["scan"] == "png" and p["multi_page"] for p in a)


def test_eu_vat_rate_follows_vendor_member_state() -> None:
    """A DE vendor charges 19%, FR 20%, NL 21% - each country has its own VAT rate."""
    expected = {"DE": "19", "FR": "20", "NL": "21"}
    for seed in range(40):
        spec = spec_for("EU", seed)
        prefix = spec.truth.vendor.tax_id[:2]  # type: ignore[index]
        assert all(t["label"] == f"VAT {expected[prefix]}%" for t in spec.view["taxes"])
