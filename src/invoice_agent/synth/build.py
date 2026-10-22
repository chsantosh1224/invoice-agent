"""Build one random but internally consistent invoice: exact truth + printable view.

The truth is a normalized `Invoice` (what a perfect extractor should return). The view is
the same invoice as printed strings for a template. Both come from the same numbers, so
the arithmetic always reconciles: line amounts = qty x price, subtotal = sum of lines,
total = subtotal - discount + tax.
"""

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from faker import Faker

from invoice_agent.schemas import Invoice, LineItem, Party
from invoice_agent.synth.formatting import (
    NumberStyle,
    format_date,
    format_money,
    format_number,
    format_quantity,
)
from invoice_agent.tax_ids import gstin_check_char

Country = Literal["IN", "US", "EU", "UK"]
CENT = Decimal("0.01")


@dataclass(frozen=True)
class Profile:
    """How invoices from one region look: currency, number/date style, tax regime.

    Attributes:
        country: Region code used by the generator.
        currency: ISO 4217 code (the truth value).
        symbols: Currency symbols the region prints; one is picked per invoice.
        number_style: Thousands/decimal separator style.
        money_suffix: True if the symbol goes after the number ('1.234,56 €').
        date_formats: Date formats in use; all must be unambiguous given the currency.
        faker_locale: Locale for fake company names and addresses.
        tax_rates: Tax/VAT rates in percent.
    """

    country: Country
    currency: str
    symbols: tuple[str, ...]
    number_style: NumberStyle
    money_suffix: bool
    date_formats: tuple[str, ...]
    faker_locale: str
    tax_rates: tuple[Decimal, ...]


PROFILES: dict[Country, Profile] = {
    "IN": Profile(
        "IN", "INR", ("₹", "Rs.", "INR"), "indian", False,
        ("dd/mm/yyyy", "dd-mm-yyyy", "dd Mon yyyy", "dd-Mon-yyyy", "yyyy-mm-dd"),
        "en_IN", (Decimal(5), Decimal(12), Decimal(18), Decimal(28)),
    ),
    "US": Profile(
        "US", "USD", ("$", "USD"), "us", False,
        ("mm/dd/yyyy", "Month dd, yyyy", "yyyy-mm-dd"),
        "en_US", (Decimal(0), Decimal("6.25"), Decimal("7.25"), Decimal("8.875")),
    ),
    "EU": Profile(
        "EU", "EUR", ("€", "EUR"), "eu", True,
        ("dd.mm.yyyy", "dd Mon yyyy"),
        "de_DE", (Decimal(19), Decimal(20), Decimal(21)),
    ),
    "UK": Profile(
        "UK", "GBP", ("£", "GBP"), "us", False,
        ("dd/mm/yyyy", "dd Mon yyyy"),
        "en_GB", (Decimal(20),),
    ),
}  # fmt: skip

# description, unit, (min price, max price), fractional quantities allowed
_CATALOG: list[tuple[str, str, tuple[int, int], bool]] = [
    ("A4 copier paper (box of 5 reams)", "box", (350, 900), False),
    ("Toner cartridge, black", "pcs", (2200, 6500), False),
    ("Laptop, 14in, 16GB RAM", "pcs", (45000, 98000), False),
    ("27in IPS monitor", "pcs", (11000, 26000), False),
    ("Wireless keyboard and mouse set", "set", (900, 3200), False),
    ("Ergonomic office chair", "pcs", (6500, 19000), False),
    ("Cat6 network cable, 305m", "roll", (4200, 9800), False),
    ("Cloud hosting - monthly plan", "month", (1500, 12000), False),
    ("Software licence, annual, per seat", "seat", (2400, 18000), False),
    ("Implementation consulting", "hrs", (1800, 6500), True),
    ("On-site support engineer", "hrs", (1200, 4000), True),
    ("Stainless steel sheet 2mm", "kg", (180, 420), True),
    ("Industrial lubricant", "ltr", (240, 780), True),
    ("Corrugated shipping boxes", "pcs", (25, 140), False),
    ("Safety gloves (pair)", "pair", (60, 320), False),
    ("Training workshop, half day", "session", (8000, 25000), False),
]

_IN_STATES = ["27", "29", "07", "33", "36", "24", "09"]  # MH, KA, DL, TN, TS, GJ, UP
_EU_VAT = {"DE": Decimal(19), "FR": Decimal(20), "NL": Decimal(21)}  # standard VAT rates


_SERVICE_UNITS = {"hrs", "month", "seat", "session"}


def _section(unit: str) -> str:
    """Group heading a line item is printed under on dense invoices.

    Args:
        unit: The item's unit of measure.

    Returns:
        'Services', 'Other charges' (freight) or 'Goods'.
    """
    if unit in _SERVICE_UNITS:
        return "Services"
    return "Other charges" if unit == "lot" else "Goods"


def _detail(unit: str, hsn: str, invoice_date: date) -> str:
    """Second description line printed under an item on dense invoices.

    Deliberately contains extra numbers (codes, references) next to the real amounts,
    the way real invoices do, so extractors must tell them apart.

    Args:
        unit: The item's unit of measure.
        hsn: The item's HSN/SAC code.
        invoice_date: The invoice date (services show a billing period).

    Returns:
        e.g. 'Item code 9405390 - delivery note DN-05390' or
        'Service period Apr 2026 - ref SR-940539'.
    """
    if unit in _SERVICE_UNITS:
        return f"Service period {invoice_date:%b %Y} - ref SR-{hsn[:6]}"
    return f"Item code {hsn} - delivery note DN-{hsn[-5:]}"


@dataclass
class InvoiceSpec:
    """One generated invoice.

    Attributes:
        truth: The normalized Invoice a perfect extractor should return.
        view: Pre-formatted strings for the HTML template.
        edge_cases: Names of the edge cases this invoice exercises.
    """

    truth: Invoice
    view: dict[str, Any]
    edge_cases: list[str] = field(default_factory=list)


def _money(value: Decimal) -> Decimal:
    """Round to cents the way invoices do (half up).

    Args:
        value: Any Decimal.

    Returns:
        The value rounded to 2 decimal places.
    """
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _gstin(rng: random.Random, state: str) -> str:
    """Generate a structurally valid GSTIN with a correct check character.

    Args:
        rng: Seeded random generator.
        state: Two-digit GST state code, e.g. '29'.

    Returns:
        A 15-character GSTIN such as '29ABCDE1234F1Z5'.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pan = (
        "".join(rng.choice(letters) for _ in range(5))
        + "".join(rng.choice("0123456789") for _ in range(4))
        + rng.choice(letters)
    )
    first14 = f"{state}{pan}{rng.choice('123')}Z"
    return first14 + gstin_check_char(first14)


def _tax_id(rng: random.Random, country: Country, state: str) -> tuple[str, str]:
    """Generate a tax ID and its printed label for a party in a region.

    Args:
        rng: Seeded random generator.
        country: Region code.
        state: GST state code (only used for India).

    Returns:
        (label, tax_id), e.g. ('GSTIN', '29ABCDE1234F1Z5') or ('VAT No.', 'DE123456789').
    """
    digits = "".join(rng.choice("0123456789") for _ in range(9))
    if country == "IN":
        return "GSTIN", _gstin(rng, state)
    if country == "US":
        return "EIN", f"{digits[:2]}-{digits[2:]}"
    if country == "EU":
        return "VAT No.", rng.choice(list(_EU_VAT)) + digits
    return "VAT Reg. No.", "GB" + digits


def _invoice_number(rng: random.Random, country: Country, credit: bool) -> str:
    """Generate an invoice number in a region-typical format.

    Args:
        rng: Seeded random generator.
        country: Region code.
        credit: True for a credit note (uses a CN prefix).

    Returns:
        e.g. 'INV-2026-0042', '2026/IN/00123', 'RE-2026-0815' or 'CN-2026-0007'.
    """
    n = rng.randint(1, 9999)
    if credit:
        return f"CN-2026-{n:04d}"
    return rng.choice(
        {
            "IN": [f"INV-2026-{n:04d}", f"2026-27/{n:05d}", f"GST/{n:04d}/26-27"],
            "US": [f"INV-{n:05d}", f"{rng.randint(100000, 999999)}", f"A-{n:04d}"],
            "EU": [f"RE-2026-{n:04d}", f"R{rng.randint(100000, 999999)}"],
            "UK": [f"INV{n:06d}", f"SI-{n:05d}"],
        }[country]
    )


def _party(fake: Faker, rng: random.Random, country: Country, state: str) -> dict[str, Any]:
    """Generate one company with address lines, tax ID and contacts.

    Args:
        fake: Faker seeded for this invoice, in the region's locale.
        rng: Seeded random generator.
        country: Region code.
        state: GST state code (India only).

    Returns:
        A dict with name, address_lines, tax_label, tax_id, email and phone.
    """
    label, tax_id = _tax_id(rng, country, state)
    lines = [ln.strip() for ln in fake.address().replace(", ", "\n").split("\n") if ln.strip()]
    return {
        "name": fake.company(),
        "address_lines": lines[:3],
        "tax_label": label,
        "tax_id": tax_id,
        "email": fake.company_email(),
        "phone": fake.phone_number(),
    }


def build_invoice(
    rng: random.Random, fake: Faker, *, country: Country, multi_page: bool = False
) -> InvoiceSpec:
    """Generate one invoice with exact truth and a matching printable view.

    Edge cases are drawn at random: missing PO (20%), discount (25%), freight line
    (20%), credit note (8%), due on receipt (10%), buyer without tax ID (30%).

    Args:
        rng: Seeded random generator (controls every random choice).
        fake: Faker instance seeded for this invoice, in the region's locale.
        country: Region whose conventions to follow ('IN', 'US', 'EU', 'UK').
        multi_page: True to generate 28-40 line items so the table spans pages.

    Returns:
        An InvoiceSpec with the truth Invoice, the view dict for templates, and the
        list of edge cases it contains.
    """
    p = PROFILES[country]
    edge: list[str] = []
    credit = rng.random() < 0.08
    if credit:
        edge.append("credit_note")

    # Parties
    vendor_state = rng.choice(_IN_STATES)
    buyer_state = vendor_state if rng.random() < 0.5 else rng.choice(_IN_STATES)
    vendor = _party(fake, rng, country, vendor_state)
    buyer = _party(fake, rng, country, buyer_state)
    if rng.random() < 0.3:
        buyer["tax_id"] = None
        edge.append("buyer_without_tax_id")

    # Line items (prices scaled to the currency: INR catalog / ~80 for USD/EUR/GBP)
    scale = Decimal(1) if country == "IN" else Decimal("0.0125")
    n_items = rng.randint(28, 40) if multi_page else rng.randint(1, 8)
    if multi_page:
        edge.append("multi_page")
    # rows: (description, quantity, unit_price, amount, unit, hsn)
    rows: list[tuple[str, Decimal, Decimal, Decimal, str, str]] = []
    for desc, unit, (lo, hi), fractional in rng.choices(_CATALOG, k=n_items):
        price = _money(Decimal(rng.randint(lo * 100, hi * 100)) / 100 * scale)
        qty = Decimal(rng.randint(2, 40)) / 2 if fractional else Decimal(rng.choice([1, 2, 5, 10]))
        hsn = str(rng.randint(1000, 99999999))
        rows.append((desc, qty, price, _money(qty * price), unit, hsn))
    if rng.random() < 0.2:
        freight = _money(Decimal(rng.randint(150, 2500)) * scale)
        rows.append(("Freight & handling", Decimal(1), freight, freight, "lot", "996511"))
        edge.append("freight_line")
    items = [
        LineItem(description=d, quantity=q, unit_price=u, amount=a) for d, q, u, a, _, _ in rows
    ]

    subtotal = sum((r[3] for r in rows), Decimal(0))
    discount: Decimal | None = None
    if rng.random() < 0.25:
        discount = _money(subtotal * Decimal(rng.choice([5, 10])) / 100)
        edge.append("discount")
    taxable = subtotal - (discount or Decimal(0))

    # Taxes are country-specific. India splits GST into CGST+SGST (same state) or IGST
    # (inter-state); EU VAT depends on the vendor's member state; UK VAT is 20%.
    rate = _EU_VAT[vendor["tax_id"][:2]] if country == "EU" else rng.choice(p.tax_rates)
    tax_lines: list[tuple[str, Decimal]] = []
    if country == "IN" and vendor_state == buyer_state:
        half = rate / 2
        cgst = sgst = _money(taxable * half / 100)
        tax_lines = [(f"CGST @ {half:g}%", cgst), (f"SGST @ {half:g}%", sgst)]
    elif country == "IN":
        tax_lines = [(f"IGST @ {rate:g}%", _money(taxable * rate / 100))]
    elif country == "US" and rate > 0:
        tax_lines = [(f"Sales Tax ({rate:g}%)", _money(taxable * rate / 100))]
    elif country in ("EU", "UK"):
        tax_lines = [(f"VAT {rate:g}%", _money(taxable * rate / 100))]
    tax = sum((amt for _, amt in tax_lines), Decimal(0)) if tax_lines else None
    total = taxable + (tax or Decimal(0))

    # Dates and terms
    invoice_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 270))
    if rng.random() < 0.1:
        terms, due_date = "Due on receipt", None
        edge.append("due_on_receipt")
    else:
        days = rng.choice([7, 14, 15, 30, 45, 60])
        terms = rng.choice([f"Net {days}", f"Payable within {days} days"])
        due_date = invoice_date + timedelta(days=days)
    po_number: str | None = rng.choice(
        [f"PO-{rng.randint(1000, 9999)}", f"45000{rng.randint(10000, 99999)}",
         f"PO/2026/{rng.randint(100, 999)}"]
    )  # fmt: skip
    if rng.random() < 0.2:
        po_number = None
        edge.append("missing_po")

    invoice_number = _invoice_number(rng, country, credit)
    truth = Invoice(
        document_type="credit_note" if credit else "invoice",
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        due_date=due_date,
        po_number=po_number,
        currency=p.currency,
        vendor=Party(
            name=vendor["name"], address=", ".join(vendor["address_lines"]), tax_id=vendor["tax_id"]
        ),
        buyer=Party(
            name=buyer["name"], address=", ".join(buyer["address_lines"]), tax_id=buyer["tax_id"]
        ),
        line_items=items,
        subtotal=subtotal,
        discount=discount,
        tax=tax,
        total=total,
        payment_terms=terms,
    )

    # Printable view: every value as a string, in this invoice's regional style
    style, symbol, date_fmt = p.number_style, rng.choice(p.symbols), rng.choice(p.date_formats)

    view: dict[str, Any] = {
        "title": "CREDIT NOTE" if credit else ("TAX INVOICE" if country == "IN" else "INVOICE"),
        "invoice_number": invoice_number,
        "invoice_date": format_date(invoice_date, date_fmt),
        "due_date": format_date(due_date, date_fmt) if due_date else None,
        "po_number": po_number,
        "payment_terms": terms,
        "currency": p.currency,
        "vendor": vendor,
        "buyer": buyer,
        "items": [
            {
                "no": i + 1,
                "description": desc,
                "hsn": hsn,
                "unit": unit,
                "qty": format_quantity(qty, style),
                "unit_price": format_number(price, style),
                "amount": format_number(amount, style),
                "section": _section(unit),
                "detail": _detail(unit, hsn, invoice_date),
            }
            for i, (desc, qty, price, amount, unit, hsn) in enumerate(rows)
        ],
        "subtotal": format_money(subtotal, style, symbol, suffix=p.money_suffix),
        "discount": (
            format_money(discount, style, symbol, suffix=p.money_suffix)
            if discount is not None
            else None
        ),
        "taxes": [
            {"label": label, "amount": format_money(amt, style, symbol, suffix=p.money_suffix)}
            for label, amt in tax_lines
        ],
        "total": format_money(total, style, symbol, suffix=p.money_suffix),
        "bank": {
            "name": fake.company() + " Bank",
            "account": "".join(rng.choice("0123456789") for _ in range(12)),
        },
    }
    return InvoiceSpec(truth=truth, view=view, edge_cases=edge)
