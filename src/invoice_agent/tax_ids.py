"""Tax ID helpers: Indian GSTIN checksum (used by the synthetic generator and validation)."""

import re

_GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GSTIN_SHAPE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


def gstin_check_char(first14: str) -> str:
    """Compute the 15th (check) character of an Indian GSTIN.

    GSTIN uses a Luhn-style mod-36 checksum: walking from the right, every second
    character's value is doubled, each product is folded (quotient + remainder by 36),
    and the check character brings the total to a multiple of 36.

    Args:
        first14: The first 14 characters (state code, PAN, entity number, 'Z').

    Returns:
        The single check character, '0'-'9' or 'A'-'Z'.

    Raises:
        ValueError: If `first14` is not 14 characters from 0-9/A-Z.

    Example:
        >>> gstin_check_char("27AAPFU0939F1Z")
        'V'
    """
    s = first14.upper()
    if len(s) != 14 or any(c not in _GSTIN_CHARS for c in s):
        raise ValueError(f"expected 14 GSTIN characters, got {first14!r}")
    total = 0
    for i, ch in enumerate(reversed(s)):
        product = _GSTIN_CHARS.index(ch) * (2 if i % 2 == 0 else 1)
        total += product // 36 + product % 36
    return _GSTIN_CHARS[(36 - total % 36) % 36]


def is_valid_gstin(gstin: str) -> bool:
    """Check that a GSTIN has the right shape and a correct check character.

    Args:
        gstin: The tax ID as printed (spaces are ignored, case-insensitive).

    Returns:
        True if it is a structurally valid GSTIN, otherwise False. A misread OCR
        character almost always makes this False, which is why validation uses it.
    """
    s = gstin.replace(" ", "").upper()
    return bool(_GSTIN_SHAPE.match(s)) and gstin_check_char(s[:14]) == s[14]
