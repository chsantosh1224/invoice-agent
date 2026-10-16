"""Print amounts and dates the way different regions do on real invoices.

Every style here must be readable by `normalize.py` back to the exact original value;
`tests/unit/test_synth.py` checks that round trip for every style.
"""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Literal

NumberStyle = Literal["us", "indian", "eu", "space"]

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTHS_LONG = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip

DATE_FORMATS: dict[str, Callable[[date], str]] = {
    "dd/mm/yyyy": lambda d: f"{d.day:02d}/{d.month:02d}/{d.year}",
    "mm/dd/yyyy": lambda d: f"{d.month:02d}/{d.day:02d}/{d.year}",
    "dd.mm.yyyy": lambda d: f"{d.day:02d}.{d.month:02d}.{d.year}",
    "dd-mm-yyyy": lambda d: f"{d.day:02d}-{d.month:02d}-{d.year}",
    "dd Mon yyyy": lambda d: f"{d.day} {_MONTHS[d.month - 1]} {d.year}",
    "dd-Mon-yyyy": lambda d: f"{d.day:02d}-{_MONTHS[d.month - 1]}-{d.year}",
    "Month dd, yyyy": lambda d: f"{_MONTHS_LONG[d.month - 1]} {d.day}, {d.year}",
    "yyyy-mm-dd": lambda d: d.isoformat(),
}


def _group_indian(digits: str) -> str:
    """Insert Indian lakh/crore separators into a string of digits.

    Args:
        digits: The integer part only, e.g. '12345678'.

    Returns:
        The digits grouped as 3 then 2s from the right, e.g. '1,23,45,678'.
    """
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    pairs: list[str] = []
    while len(head) > 2:
        pairs.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head, *pairs, tail]) if head else ",".join([*pairs, tail])


def format_number(value: Decimal, style: NumberStyle, *, places: int = 2) -> str:
    """Print a number with a region's thousands and decimal separators.

    Args:
        value: The exact value. Negative values get a leading '-'.
        style: 'us' (1,234,567.89), 'indian' (12,34,567.89), 'eu' (1.234.567,89) or
            'space' (1 234 567,89).
        places: Decimal places to print; 0 prints an integer.

    Returns:
        The formatted number without any currency symbol.
    """
    sign = "-" if value < 0 else ""
    text = f"{abs(value):.{places}f}"
    whole, _, frac = text.partition(".")
    if style == "indian":
        grouped = _group_indian(whole)
    else:
        grouped = f"{int(whole):,}"
        if style == "eu":
            grouped = grouped.replace(",", ".")
        elif style == "space":
            grouped = grouped.replace(",", " ")
    decimal_mark = "," if style in ("eu", "space") else "."
    return f"{sign}{grouped}{decimal_mark}{frac}" if frac else f"{sign}{grouped}"


def format_quantity(value: Decimal, style: NumberStyle) -> str:
    """Print a quantity: whole numbers without decimals, fractional ones minimally.

    Args:
        value: The quantity, e.g. Decimal('10') or Decimal('7.5').
        style: Number style; 'eu'/'space' use a decimal comma.

    Returns:
        '10', '7.5' or '7,5' (never '10.00').
    """
    if value == value.to_integral_value():
        return format_number(value, style, places=0)
    exponent = value.normalize().as_tuple().exponent
    places = max(0, -exponent) if isinstance(exponent, int) else 2
    return format_number(value, style, places=places)


def format_money(value: Decimal, style: NumberStyle, symbol: str, *, suffix: bool) -> str:
    """Print an amount with its currency symbol, before or after the number.

    Args:
        value: The exact amount.
        style: Number style (see `format_number`).
        symbol: Currency symbol or code, e.g. '₹', 'Rs.', '$', '€', 'EUR'.
        suffix: True for '1.234,56 €' (EU), False for '€1,234.56' / 'Rs. 4,500.00'.

    Returns:
        The amount with symbol.
    """
    number = format_number(value, style)
    if suffix:
        return f"{number} {symbol}"
    spacer = " " if symbol[-1].isalpha() or symbol.endswith(".") else ""
    return f"{symbol}{spacer}{number}"


def format_date(value: date, fmt: str) -> str:
    """Print a date in one of the supported invoice formats.

    Args:
        value: The date.
        fmt: A key of DATE_FORMATS, e.g. 'dd/mm/yyyy' or 'Month dd, yyyy'.

    Returns:
        The printed date, e.g. '03/04/2026' or 'April 3, 2026'.

    Raises:
        KeyError: If `fmt` is not a known format.
    """
    return DATE_FORMATS[fmt](value)
