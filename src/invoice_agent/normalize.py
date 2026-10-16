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
    """Convert a printed money or quantity string into an exact Decimal.

    Handles the formats seen on real invoices: currency symbols and codes, Indian lakh
    grouping, European decimal commas, thousands separators, and the accounting
    conventions for negatives (parentheses, leading/trailing minus, 'CR'). The Indian
    suffix '/-' ("only") is treated as decoration, not a minus sign.

    Args:
        text: The value exactly as printed on the invoice, e.g. '₹1,23,456.00',
            '1.234,56 €', '(500.00)', 'Rs. 4,500/-'.

    Returns:
        The numeric value as a Decimal (never a float, so money stays exact).
        Negative for credit/accounting-negative formats.

    Raises:
        ValueError: If the text contains no digits or cannot be read as a number.

    Example:
        >>> parse_amount("₹1,23,456.00")
        Decimal('123456.00')
    """
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
_SEPT = re.compile(r"\bSept\b\.?", re.IGNORECASE)  # 'Sept' -> 'Sep', but not inside 'September'

DAYFIRST_CURRENCIES = {"INR", "EUR", "GBP", "AUD", "NZD", "SGD", "AED", "ZAR"}


class AmbiguousDateError(ValueError):
    """Raised when a numeric date could be DD/MM or MM/DD and no hint says which."""


def parse_date(text: str, *, dayfirst: bool | None = None) -> date:
    """Convert a printed date string into a `datetime.date`.

    Supports ISO dates, numeric dates with '/', '-' or '.' separators, and text dates
    such as '3 Apr 2026', 'April 3rd, 2026' or '03-Apr-2026'. A numeric date is only
    resolved without a hint when it is unambiguous (one part > 12, or day == month).

    Args:
        text: The date exactly as printed on the invoice.
        dayfirst: How to read an ambiguous numeric date such as '03/04/2026'.
            True = DD/MM (India, Europe), False = MM/DD (US), None = no hint, in which
            case an ambiguous date raises instead of being guessed.

    Returns:
        The parsed calendar date.

    Raises:
        AmbiguousDateError: If the date is ambiguous and `dayfirst` is None.
        ValueError: If the text matches no supported date format.

    Example:
        >>> parse_date("03/04/2026", dayfirst=True)
        datetime.date(2026, 4, 3)
    """
    s = _SEPT.sub("Sep", _ORDINAL.sub(r"\1", text.strip()))
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
    """Convert a printed currency symbol or code into an ISO 4217 code.

    An explicit 3-letter code wins over a symbol, so 'CAD $' becomes CAD rather than USD.
    A bare '$' defaults to USD.

    Args:
        text: The currency as printed, e.g. '₹', 'Rs.', 'INR', '€', 'usd', 'CAD $'.

    Returns:
        The uppercase ISO 4217 code, e.g. 'INR'.

    Raises:
        ValueError: If no known symbol or 3-letter code is found.

    Example:
        >>> parse_currency("Rs.")
        'INR'
    """
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
    """Collapse whitespace in a printed string and turn empty strings into None.

    Args:
        text: Any transcribed text value, or None.

    Returns:
        The text with runs of whitespace/newlines collapsed to single spaces, or None
        if the input was None or contained only whitespace.
    """
    if text is None:
        return None
    text = " ".join(text.split())
    return text or None


def _abs(value: Decimal | None) -> Decimal | None:
    """Make an amount positive, keeping None as None.

    Used for discounts, which invoices print either as '500.00' or '-500.00' but which
    always mean a reduction.

    Args:
        value: A parsed amount or None.

    Returns:
        The absolute value, or None.
    """
    return None if value is None else abs(value)


def _sum_taxes(amounts: list[Decimal | None]) -> Decimal | None:
    """Add up the parsed tax lines (e.g. CGST + SGST) into the invoice's total tax.

    The LLM transcribes each tax line separately; adding them up is done here, in code,
    so the LLM never does arithmetic.

    Args:
        amounts: Parsed tax-line amounts; None for lines that failed to parse.

    Returns:
        The sum of all parsed amounts, or None if there were no tax lines or none parsed
        (a failed line is already reported as an issue by the caller).
    """
    parsed = [a for a in amounts if a is not None]
    return sum(parsed, Decimal(0)) if parsed else None


def _party(raw: RawParty) -> Party:
    """Convert a transcribed party (vendor or buyer) into the normalized Party model.

    Args:
        raw: The party as the LLM transcribed it.

    Returns:
        A Party with whitespace-cleaned name, address and tax ID (None where blank).
    """
    return Party(name=_clean(raw.name), address=_clean(raw.address), tax_id=_clean(raw.tax_id))


def to_invoice(raw: RawInvoice) -> tuple[Invoice, list[ValidationIssue]]:
    """Normalize a full LLM transcription into a typed Invoice.

    Converts every amount to Decimal, every date to `date` and the currency to ISO 4217.
    The currency also decides how ambiguous dates are read (₹/€/£ -> day first,
    $ -> month first). A value that fails to parse is set to None and reported as a
    ValidationIssue; it is never guessed.

    Args:
        raw: The invoice transcribed exactly as printed (output of the LLM).

    Returns:
        A tuple of:
        - the normalized Invoice (unparseable fields are None), and
        - the list of normalization issues found (empty when everything parsed).
    """
    issues: list[ValidationIssue] = []

    def amount(field: str, text: str | None) -> Decimal | None:
        """Parse one amount, recording an issue instead of raising.

        Args:
            field: Dotted field path used in the issue, e.g. 'line_items[0].amount'.
            text: The printed value, or None if the field was absent.

        Returns:
            The Decimal value, or None if the field was absent or unparseable.
        """
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
        """Parse one date using the invoice's currency as the day/month hint.

        Args:
            field: Field name used in the issue, e.g. 'invoice_date'.
            text: The printed date, or None if the field was absent.

        Returns:
            The parsed date, or None if absent, ambiguous or unrecognized (the last two
            are recorded as issues).
        """
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
        discount=_abs(amount("discount", raw.discount)),  # printed as '-500' or '500'
        tax=_sum_taxes([amount(f"tax_lines[{i}]", t.amount) for i, t in enumerate(raw.tax_lines)]),
        total=amount("total", raw.total),
        payment_terms=_clean(raw.payment_terms),
    )
    return invoice, issues
