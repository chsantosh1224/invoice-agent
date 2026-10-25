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
    RawLineItem,
    RawParty,
    ValidationIssue,
)
from invoice_agent.summary import SummaryRow, resolve_summary

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
        head, _, tail = s.rpartition(".")
        # '1.234.567' = thousands dots. But '10.476.38' (OCR read a comma as a dot) ends in
        # 2 digits, which only a decimal part does, so the last dot is the decimal point.
        s = f"{head.replace('.', '')}.{tail}" if len(tail) == 2 else s.replace(".", "")

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


# A time after the date: ', 10:56:14 AM', ' at 9:30pm', 'T10:56:14Z'.
_TIME = re.compile(
    r"(?:[,\s]+(?:at\s+)?|T)\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*(?:[AaPp]\.?\s*[Mm]\.?)?"
    r"\s*(?:Z|[+-]\d{2}:?\d{2}|[A-Z]{2,4})?\s*$"
)


def strip_time(text: str) -> str:
    """Remove a time of day printed after a date.

    Args:
        text: A printed date, possibly with a time, e.g. 'Dec 11, 2020, 10:56:14 AM'.

    Returns:
        The date part only, e.g. 'Dec 11, 2020'; the text unchanged if there is no time.
    """
    return _TIME.sub("", text.strip())


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
    s = _SEPT.sub("Sep", _ORDINAL.sub(r"\1", strip_time(text)))
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
# Dollar signs with a country prefix are unambiguous. Longest first: 'CA$' contains 'A$'.
_PREFIXED_DOLLARS = {
    "US$": "USD", "AU$": "AUD", "CA$": "CAD", "NZ$": "NZD", "HK$": "HKD",
    "A$": "AUD", "C$": "CAD", "S$": "SGD",
}  # fmt: skip
# Symbols used by several currencies; the first code is what we assume.
AMBIGUOUS_SYMBOLS = {
    "$": ("USD", "AUD", "CAD", "NZD", "SGD", "HKD", "MXN"),
    "¥": ("JPY", "CNY"),
}
_NUMERIC_CHARS = set("0123456789.,-()/ ")


def parse_currency(text: str) -> str:
    """Convert a printed currency symbol or code into an ISO 4217 code.

    Checked in order: prefixed dollars ('A$' -> AUD), an explicit 3-letter code ('CAD $'
    -> CAD), then plain symbols. A bare '$' returns USD; use `ambiguous_symbol` to find out
    whether that was an assumption.

    Args:
        text: The currency as printed, e.g. '₹', 'Rs.', 'INR', '€', 'usd', 'A$', '$93.50'.

    Returns:
        The uppercase ISO 4217 code, e.g. 'INR'.

    Raises:
        ValueError: If no known symbol or 3-letter code is found.

    Example:
        >>> parse_currency("A$ 93.50")
        'AUD'
    """
    s = text.strip().upper()
    if s in _SYMBOLS:
        return _SYMBOLS[s]
    for prefix, code in _PREFIXED_DOLLARS.items():
        if prefix in s:
            return code
    m = _ISO.search(s)
    if m:
        return m.group(1)
    for sym, code in _SYMBOLS.items():
        if sym in s:
            return code
    raise ValueError(f"unknown currency {text!r}")


def ambiguous_symbol(text: str) -> str | None:
    """Tell whether a printed currency relies on a symbol shared by several currencies.

    Args:
        text: The currency or amount as printed, e.g. '$', '$93.50', 'US$ 5', 'USD 5'.

    Returns:
        The ambiguous symbol ('$' or '¥') if that is all the text says, or None when the
        currency is unambiguous (an ISO code, a prefixed dollar, ₹/€/£, or no symbol).

    Example:
        >>> ambiguous_symbol("$93.50"), ambiguous_symbol("AUD 93.50")
        ('$', None)
    """
    s = text.strip().upper()
    if _ISO.search(s) or any(prefix in s for prefix in _PREFIXED_DOLLARS):
        return None
    return next((sym for sym in AMBIGUOUS_SYMBOLS if sym in s), None)


def _has_currency_marker(text: str) -> bool:
    """Check whether a printed amount carries a currency symbol or code.

    Args:
        text: An amount as printed, e.g. '$93.50' or '93.50'.

    Returns:
        True if it contains anything besides digits, separators, signs and spaces.
    """
    return any(c not in _NUMERIC_CHARS for c in text)


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


def _discount(value: Decimal | None) -> Decimal | None:
    """Normalize a discount: always positive, and a zero discount means no discount.

    Invoices print discounts as '500.00', '-500.00' or '(500.00)'; all mean a reduction.
    Models also sometimes invent 'discount: 0.00' when none is printed - equivalent to
    no discount, so it becomes None instead of a (hallucinated) value.

    Args:
        value: A parsed amount or None.

    Returns:
        The absolute value, or None if absent or zero.
    """
    return None if value is None or value == 0 else abs(value)


def _tax_id(text: str | None) -> str | None:
    """Clean a tax ID; text without any digit is not a tax ID.

    GST invoices print 'GSTIN: Unregistered' for unregistered buyers, and models copy the
    word into the tax-ID field (the only error left in the Day 3 bake-off). Real tax IDs
    (GSTIN, VAT, EIN) always contain digits.

    Args:
        text: The tax ID as transcribed, or None.

    Returns:
        The whitespace-cleaned ID, or None if absent or digit-free ('Unregistered', 'N/A').
    """
    cleaned = _clean(text)
    return cleaned if cleaned and any(c.isdigit() for c in cleaned) else None


def _party(raw: RawParty) -> Party:
    """Convert a transcribed party (vendor or buyer) into the normalized Party model.

    Args:
        raw: The party as the LLM transcribed it.

    Returns:
        A Party with whitespace-cleaned name and address, and a validated tax ID
        (None where blank or not a real ID).
    """
    return Party(name=_clean(raw.name), address=_clean(raw.address), tax_id=_tax_id(raw.tax_id))


def _amount_field(field: str, text: str | None, issues: list[ValidationIssue]) -> Decimal | None:
    """Parse one amount field, recording an issue instead of raising.

    Args:
        field: Dotted field path used in the issue, e.g. 'line_items[0].amount'.
        text: The printed value, or None if the field was absent.
        issues: List the issue is appended to when the value cannot be parsed.

    Returns:
        The Decimal value, or None if the field was absent or unparseable.
    """
    cleaned = _clean(text)
    if cleaned is None:
        return None
    try:
        return parse_amount(cleaned)
    except ValueError as e:
        issues.append(ValidationIssue(field=field, code="normalize.amount", message=str(e)))
        return None


def _date_field(
    field: str, text: str | None, dayfirst: bool | None, issues: list[ValidationIssue]
) -> date | None:
    """Parse one date field, recording an issue instead of raising.

    Args:
        field: Field name used in the issue, e.g. 'invoice_date'.
        text: The printed date, or None if the field was absent.
        dayfirst: How to read ambiguous numeric dates (from the invoice's currency).
        issues: List the issue is appended to when the date is ambiguous or unrecognized.

    Returns:
        The parsed date, or None if absent, ambiguous or unrecognized.
    """
    cleaned = _clean(text)
    if cleaned is None:
        return None
    try:
        return parse_date(cleaned, dayfirst=dayfirst)
    except AmbiguousDateError as e:
        issues.append(ValidationIssue(field=field, code="normalize.date_ambiguous", message=str(e)))
    except ValueError as e:
        issues.append(ValidationIssue(field=field, code="normalize.date", message=str(e)))
    return None


def _printed_amounts(raw: RawInvoice) -> list[str]:
    """Collect every amount string the model transcribed (totals, taxes, line items).

    Args:
        raw: The transcription.

    Returns:
        The non-empty amount strings, total rows first.
    """
    totals = [r.amount for r in raw.summary_lines if r.kind == "total"]
    texts = totals + [r.amount for r in raw.summary_lines if r.kind != "total"]
    for li in raw.line_items:
        texts += [li.amount, li.unit_price]
    return [t for t in texts if t and t.strip()]


def _currency_from_amounts(
    raw: RawInvoice, issues: list[ValidationIssue]
) -> tuple[str | None, str | None]:
    """Work out the currency from the symbols printed on the amounts.

    Used when the model left `currency` empty because the document has no separate
    currency line - the symbol on '$93.50' is still printed evidence, not a guess.

    Args:
        raw: The transcription.
        issues: Receives a warning saying the currency was taken from the amounts, or
            that the amounts carry conflicting currencies.

    Returns:
        (ISO code, an example amount it came from), or (None, None) if no amount carries
        a currency or the amounts disagree.
    """
    found: dict[str, str] = {}
    for text in _printed_amounts(raw):
        if not _has_currency_marker(text):
            continue
        try:
            found.setdefault(parse_currency(text), text)
        except ValueError:
            continue
    if len(found) > 1:
        issues.append(
            ValidationIssue(
                field="currency",
                code="currency.conflicting_symbols",
                message=f"amounts are printed in different currencies: {sorted(found)}",
            )
        )
        return None, None
    if not found:
        return None, None
    code, example = next(iter(found.items()))
    issues.append(
        ValidationIssue(
            field="currency",
            code="currency.inferred",
            message=f"no currency stated; taken from amounts printed like {example!r}",
            severity="warning",
        )
    )
    return code, example


def _currency_field(raw: RawInvoice, issues: list[ValidationIssue]) -> str | None:
    """Resolve the invoice currency: the currency field, else the symbols on the amounts.

    A bare '$' or '¥' is resolved to its most common currency (USD, JPY) but reported as
    `currency.ambiguous_symbol`, so the validation gate sends the invoice to review instead
    of silently assuming. That assumption also decides how DD/MM vs MM/DD dates are read.

    Args:
        raw: The transcription.
        issues: Receives warnings (unknown, inferred, ambiguous) and the conflict error.

    Returns:
        The ISO 4217 code, or None if no currency can be determined.
    """
    printed = _clean(raw.currency)
    code: str | None = None
    if printed is not None:
        try:
            code = parse_currency(printed)
        except ValueError as e:
            issues.append(
                ValidationIssue(
                    field="currency", code="normalize.currency", message=str(e), severity="warning"
                )
            )
    if code is None:
        code, printed = _currency_from_amounts(raw, issues)
    if code is not None and printed is not None:
        symbol = ambiguous_symbol(printed)
        if symbol is not None:
            options = ", ".join(AMBIGUOUS_SYMBOLS[symbol])
            issues.append(
                ValidationIssue(
                    field="currency",
                    code="currency.ambiguous_symbol",
                    message=f"'{symbol}' alone could be {options}; assumed {code}. "
                    "Confirm against the vendor or PO.",
                    severity="warning",
                )
            )
    return code


def _line_item(index: int, raw: RawLineItem, issues: list[ValidationIssue]) -> LineItem:
    """Normalize one line-item row, including its own discount and tax columns.

    Args:
        index: Row position, used in issue field paths.
        raw: The row as transcribed.
        issues: Receives an issue for every value that cannot be parsed.

    Returns:
        The LineItem. A discount printed with '%' becomes `discount_pct`, otherwise an
        amount in `discount`; the tax rate is a percentage number (18 for '18%').
    """
    path = f"line_items[{index}]"
    discount_text = _clean(raw.discount)
    is_pct = discount_text is not None and "%" in discount_text
    discount = _amount_field(f"{path}.discount", discount_text, issues)
    return LineItem(
        description=_clean(raw.description),
        quantity=_amount_field(f"{path}.quantity", raw.quantity, issues),
        unit_price=_amount_field(f"{path}.unit_price", raw.unit_price, issues),
        amount=_amount_field(f"{path}.amount", raw.amount, issues),
        discount=None if is_pct or discount is None or discount == 0 else abs(discount),
        discount_pct=abs(discount) if is_pct and discount else None,
        tax_rate=_amount_field(f"{path}.tax_rate", raw.tax_rate, issues),
        tax_amount=_amount_field(f"{path}.tax_amount", raw.tax_amount, issues),
    )


def to_invoice(raw: RawInvoice) -> tuple[Invoice, list[ValidationIssue]]:
    """Normalize a full LLM transcription into a typed Invoice.

    Converts every amount to Decimal, every date to `date` and the currency to ISO 4217.
    The currency also decides how ambiguous dates are read (₹/€/£ -> day first,
    $ -> month first). The totals block is resolved by `summary.resolve_summary` (kinds
    checked by arithmetic). A value that fails to parse is set to None and reported as a
    ValidationIssue; it is never guessed.

    Args:
        raw: The invoice transcribed exactly as printed (output of the LLM).

    Returns:
        A tuple of:
        - the normalized Invoice (unparseable fields are None), and
        - the list of normalization issues found (empty when everything parsed).
    """
    issues: list[ValidationIssue] = []
    currency = _currency_field(raw, issues)
    dayfirst = None if currency is None else currency in DAYFIRST_CURRENCIES

    items = [_line_item(i, li, issues) for i, li in enumerate(raw.line_items)]
    amounts = [li.amount for li in items if li.amount is not None]
    lines_sum = sum(amounts, Decimal(0)) if items and len(amounts) == len(items) else None
    rows = [
        SummaryRow(index=i, label=_clean(r.label), kind=r.kind, amount=amount)
        for i, r in enumerate(raw.summary_lines)
        if (amount := _amount_field(f"summary_lines[{i}]", r.amount, issues)) is not None
    ]
    totals = resolve_summary(rows, lines_sum)
    issues += totals.issues

    invoice = Invoice(
        document_type=raw.document_type,
        invoice_number=_clean(raw.invoice_number),
        invoice_date=_date_field("invoice_date", raw.invoice_date, dayfirst, issues),
        due_date=_date_field("due_date", raw.due_date, dayfirst, issues),
        po_number=_clean(raw.po_number),
        currency=currency,
        vendor=_party(raw.vendor),
        buyer=_party(raw.buyer),
        line_items=items,
        subtotal=totals.subtotal,
        discount=_discount(totals.discount),
        tax=totals.tax,
        tax_included=totals.tax_included,
        other_charges=totals.charges,
        total=totals.total,
        payment_terms=_clean(raw.payment_terms),
    )
    return invoice, issues
