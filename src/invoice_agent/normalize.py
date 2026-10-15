"""Deterministic normalization of values the LLM transcribed as printed.

The LLM never converts numbers or dates; this module does, so the conversion is testable
and identical on every run. Anything that cannot be parsed confidently becomes a
ValidationIssue instead of a guess.
"""

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from invoice_agent.schemas import (
    Invoice,
    LineItem,
    Party,
    RawInvoice,
    RawParty,
    ValidationIssue,
)

# ---------- amounts ----------

_CURRENCY_NOISE = re.compile(r"[^\d,.\-()]")


def parse_amount(text: str) -> Decimal:
    """'₹1,23,456.00' -> 123456.00, '1.234,56 €' -> 1234.56, '(500.00)' -> -500.00."""
    s = text.strip()
    s = s.removesuffix("/-").strip()  # Indian 'Rs. 4,500/-' means "4,500 only", not negative
    negative = (
        (s.startswith("(") and s.endswith(")"))
        or s.startswith("-")
        or s.endswith("-")
        or s.upper().endswith("CR")
    )
    s = _CURRENCY_NOISE.sub("", s).replace("(", "").replace(")", "").replace("-", "")
    s = s.strip(".,")  # stray separators left over from 'Rs.' or a trailing period
    if not s or not any(c.isdigit() for c in s):
        raise ValueError(f"no digits in amount {text!r}")

    if "," in s and "." in s:
        # Whichever separator comes last is the decimal point.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        # A single comma followed by 1-2 digits is a decimal comma ('12,5' / '1234,56').
        s = f"{head}.{tail}" if s.count(",") == 1 and len(tail) in (1, 2) else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")  # '1.234.567' thousands dots

    try:
        value = Decimal(s)
    except InvalidOperation as e:
        raise ValueError(f"unparseable amount {text!r}") from e
    return -value if negative else value


# ---------- dates ----------

_NUMERIC_DATE = re.compile(r"^\s*(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$")
_TEXT_FORMATS = (
    "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y", "%b %d, %Y", "%B %d, %Y",
    "%b %d %Y", "%B %d %Y", "%d %b, %Y", "%d %B, %Y", "%d-%b-%y", "%d %b %y",
)  # fmt: skip
_ORDINAL = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)

DAYFIRST_CURRENCIES = {"INR", "EUR", "GBP", "AUD", "NZD", "SGD", "AED", "ZAR"}


class AmbiguousDateError(ValueError):
    pass


def parse_date(text: str, *, dayfirst: bool | None = None) -> date:
    """Parse a printed date. Numeric dates where both parts are <= 12 need `dayfirst`."""
    s = _ORDINAL.sub(r"\1", text.strip()).replace("Sept", "Sep")
    m = _NUMERIC_DATE.match(s)
    if m:
        a, b, c = m.groups()
        if len(a) == 4:  # ISO-ish: 2026-04-03
            return date(int(a), int(b), int(c))
        year = int(c) + (2000 if len(c) == 2 else 0)
        first, second = int(a), int(b)
        if first > 12:
            day, month = first, second
        elif second > 12:
            month, day = first, second
        elif first == second:
            day = month = first
        elif dayfirst is None:
            raise AmbiguousDateError(f"{text!r} could be DD/MM or MM/DD")
        else:
            day, month = (first, second) if dayfirst else (second, first)
        return date(year, month, day)

    for fmt in _TEXT_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date {text!r}")


# ---------- currency ----------

_SYMBOLS = {"₹": "INR", "RS": "INR", "RS.": "INR", "€": "EUR", "£": "GBP", "¥": "JPY", "$": "USD"}
_ISO = re.compile(r"\b([A-Z]{3})\b")


def parse_currency(text: str) -> str:
    s = text.strip().upper()
    if s in _SYMBOLS:
        return _SYMBOLS[s]
    m = _ISO.search(s)
    if m:
        return m.group(1)
    for sym, code in _SYMBOLS.items():
        if sym in s:
            return code
    raise ValueError(f"unknown currency {text!r}")


# ---------- whole invoice ----------


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    return text or None


def _party(raw: RawParty) -> Party:
    return Party(name=_clean(raw.name), address=_clean(raw.address), tax_id=_clean(raw.tax_id))


def to_invoice(raw: RawInvoice) -> tuple[Invoice, list[ValidationIssue]]:
    """Normalize a transcription. Returns the invoice plus any normalization issues."""
    issues: list[ValidationIssue] = []

    def amount(field: str, text: str | None) -> Decimal | None:
        if not _clean(text):
            return None
        try:
            return parse_amount(text)  # type: ignore[arg-type]
        except ValueError as e:
            issues.append(ValidationIssue(field=field, code="normalize.amount", message=str(e)))
            return None

    currency: str | None = None
    if _clean(raw.currency):
        try:
            currency = parse_currency(raw.currency)  # type: ignore[arg-type]
        except ValueError as e:
            issues.append(
                ValidationIssue(
                    field="currency", code="normalize.currency", message=str(e), severity="warning"
                )
            )

    dayfirst = None if currency is None else currency in DAYFIRST_CURRENCIES

    def when(field: str, text: str | None) -> date | None:
        if not _clean(text):
            return None
        try:
            return parse_date(text, dayfirst=dayfirst)  # type: ignore[arg-type]
        except AmbiguousDateError as e:
            issues.append(
                ValidationIssue(field=field, code="normalize.date_ambiguous", message=str(e))
            )
        except ValueError as e:
            issues.append(ValidationIssue(field=field, code="normalize.date", message=str(e)))
        return None

    items = [
        LineItem(
            description=_clean(li.description),
            quantity=amount(f"line_items[{i}].quantity", li.quantity),
            unit_price=amount(f"line_items[{i}].unit_price", li.unit_price),
            amount=amount(f"line_items[{i}].amount", li.amount),
        )
        for i, li in enumerate(raw.line_items)
    ]

    invoice = Invoice(
        document_type=raw.document_type,
        invoice_number=_clean(raw.invoice_number),
        invoice_date=when("invoice_date", raw.invoice_date),
        due_date=when("due_date", raw.due_date),
        po_number=_clean(raw.po_number),
        currency=currency,
        vendor=_party(raw.vendor),
        buyer=_party(raw.buyer),
        line_items=items,
        subtotal=amount("subtotal", raw.subtotal),
        discount=amount("discount", raw.discount),
        tax=amount("tax", raw.tax),
        total=amount("total", raw.total),
        payment_terms=_clean(raw.payment_terms),
    )
    return invoice, issues
