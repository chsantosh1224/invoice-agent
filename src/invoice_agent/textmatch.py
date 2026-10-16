"""Matching normalized values against document text (used by validation and evals).

Comparisons are on meaning, not formatting: amounts are parsed to Decimal, dates are tried
in every known print format, IDs are compared without whitespace and case.
"""

import re
from datetime import date
from decimal import Decimal

from invoice_agent.normalize import parse_amount
from invoice_agent.synth.formatting import DATE_FORMATS

_NUMBER = re.compile(r"\d[\d.,]*\d|\d")
_SPACES = re.compile(r"\s+")


def numbers_in(text: str) -> set[Decimal]:
    """Find every number-like token in a text and parse it as an amount.

    Args:
        text: Any text (a whole document or a single line).

    Returns:
        The absolute values of every token that parses (unparseable tokens are skipped).
    """
    found = set()
    for token in _NUMBER.findall(text):
        try:
            found.add(abs(parse_amount(token)))
        except ValueError:
            continue
    return found


def compact(text: str) -> str:
    """Lowercase and remove all whitespace, for spacing-insensitive containment checks.

    Args:
        text: Any text.

    Returns:
        The text lowercased with all whitespace removed.
    """
    return _SPACES.sub("", text).lower()


def date_present(value: date, compact_text: str) -> bool:
    """Check whether a date appears in a text in any supported print format.

    Args:
        value: The date to look for.
        compact_text: Text already passed through `compact`.

    Returns:
        True if any format of the date (e.g. '03/04/2026', '3 Apr 2026') is present.
    """
    return any(compact(fmt(value)) in compact_text for fmt in DATE_FORMATS.values())
