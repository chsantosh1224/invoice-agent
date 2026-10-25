"""Turn the totals block (summary lines with a `kind`) into the invoice's totals.

The LLM copies every row of the totals block and proposes a `kind` for each one
(subtotal, discount, tax, charge, ...). This module applies the kinds and checks them
with arithmetic, which the model cannot influence:

    subtotal - discounts + taxes + charges - deductions +/- adjustments = total

- Rows the model marked "other" are classified by the arithmetic when exactly one reading
  (+ or -) makes the block add up; otherwise they stay unclassified and are reported.
- If the model's kinds do not add up but flipping the sign of exactly one row does, that
  reading is used and reported. Every such inference is a warning that sends the invoice
  to review; nothing is silently guessed.
- Rows of kind "tax_included" are informational: that tax is already inside the prices.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from itertools import product

from invoice_agent.schemas import Charge, SummaryKind, ValidationIssue

TOLERANCE = Decimal("0.02")
MAX_UNKNOWN_ROWS = 4  # 2**4 sign combinations at most


@dataclass
class SummaryRow:
    """One parsed row of the totals block.

    Attributes:
        index: Position in the printed block (0 = top).
        label: The label as printed.
        kind: The kind the model proposed.
        amount: The amount as printed, parsed (negative if printed with a minus/brackets).
    """

    index: int
    label: str | None
    kind: SummaryKind
    amount: Decimal


@dataclass
class Totals:
    """The invoice totals derived from the summary rows.

    Attributes:
        subtotal: The first 'subtotal' row (None if there is none).
        discount: Sum of all discounts, as a positive number (None if none).
        tax: Sum of all added taxes (None if none).
        tax_included: Sum of informational 'tax included' rows (None if none).
        charges: Charges, deductions and adjustments with signed amounts.
        total: The last 'total' row: the amount payable (None if there is none).
        issues: Warnings/errors found while resolving the block.
    """

    subtotal: Decimal | None = None
    discount: Decimal | None = None
    tax: Decimal | None = None
    tax_included: Decimal | None = None
    charges: list[Charge] = field(default_factory=list)
    total: Decimal | None = None
    issues: list[ValidationIssue] = field(default_factory=list)


def signed_effect(row: SummaryRow) -> Decimal:
    """How a row changes the amount payable, according to its kind.

    Args:
        row: A parsed summary row.

    Returns:
        A positive number for rows that add (tax, charge), a negative number for rows that
        reduce (discount, deduction), the printed sign for adjustments, and 0 for rows
        that do not enter the sum (subtotal, total, tax_included, other).
    """
    if row.kind in ("discount", "deduction"):
        return -abs(row.amount)
    if row.kind in ("tax", "adjustment"):
        return row.amount
    if row.kind == "charge":
        return row.amount  # a charge printed negative is reported, but taken as printed
    return Decimal(0)


def _reconciles(base: Decimal, effects: list[Decimal], total: Decimal) -> bool:
    """Check whether base plus all row effects equals the total (within 2 cents).

    Args:
        base: Subtotal (or the sum of the line items when no subtotal row is printed).
        effects: Signed effect of every non-subtotal, non-total row.
        total: The printed total.

    Returns:
        True if the block adds up.
    """
    return abs(base + sum(effects, Decimal(0)) - total) <= TOLERANCE


def _issue(row: SummaryRow, code: str, message: str, *, error: bool = False) -> ValidationIssue:
    """Build an issue about one summary row.

    Args:
        row: The row concerned.
        code: Machine-readable code, e.g. 'summary.kind_inferred'.
        message: Human-readable explanation.
        error: True for an error (fails the gate), False for a warning.

    Returns:
        The ValidationIssue, pointing at summary_lines[index].
    """
    return ValidationIssue(
        field=f"summary_lines[{row.index}]",
        code=code,
        message=f"{row.label or '(no label)'} {row.amount}: {message}",
        severity="error" if error else "warning",
    )


def _sign_assignments(
    unknown: list[SummaryRow], base: Decimal, known_effect: Decimal, total: Decimal
) -> list[tuple[int, ...]]:
    """Find every +/- reading of the unclassified rows that makes the block add up.

    Args:
        unknown: Rows whose kind is 'other'.
        base: Subtotal or line-item sum.
        known_effect: Combined effect of all classified rows.
        total: The printed total.

    Returns:
        Each sign tuple (+1 adds, -1 subtracts, one per unknown row) that reconciles;
        empty if none does or there are too many unknown rows to try.
    """
    if len(unknown) > MAX_UNKNOWN_ROWS:
        return []
    fits = []
    for signs in product((1, -1), repeat=len(unknown)):
        effects = [s * abs(r.amount) for s, r in zip(signs, unknown, strict=True)]
        if _reconciles(base, [known_effect, *effects], total):
            fits.append(signs)
    return fits


def _resolve_unknown(
    rows: list[SummaryRow], base: Decimal, total: Decimal
) -> list[ValidationIssue]:
    """Classify 'other' rows (and fix one wrong sign) using the arithmetic.

    Changes `kind` (and for a flipped row, `amount`) on the rows in place.

    Args:
        rows: All rows of the block except subtotal/total/tax_included.
        base: Subtotal or line-item sum.
        total: The printed total.

    Returns:
        Warnings for every inferred kind or corrected sign, and errors for rows that could
        not be classified.
    """
    issues: list[ValidationIssue] = []
    unknown = [r for r in rows if r.kind == "other"]
    known_effect = sum((signed_effect(r) for r in rows if r.kind != "other"), Decimal(0))

    if unknown:
        fits = _sign_assignments(unknown, base, known_effect, total)
        if len(fits) == 1:
            for sign, row in zip(fits[0], unknown, strict=True):
                row.kind = "charge" if sign > 0 else "deduction"
                reason = f"read as a {row.kind} because that makes the totals add up"
                issues.append(_issue(row, "summary.kind_inferred", reason))
        else:
            reason = "cannot tell whether this row adds or subtracts"
            issues += [_issue(r, "summary.unclassified", reason, error=True) for r in unknown]
        return issues

    if _reconciles(base, [known_effect], total):
        return issues
    flips = [
        r
        for r in rows
        if r.kind in ("discount", "tax", "charge", "deduction", "adjustment")
        and _reconciles(base, [known_effect - 2 * signed_effect(r)], total)
    ]
    if len(flips) == 1:
        row = flips[0]
        row.amount = -signed_effect(row)
        row.kind = "adjustment"
        reason = "the totals only add up with the opposite sign"
        issues.append(_issue(row, "summary.sign_corrected", reason))
    return issues  # otherwise the gate reports math.total


def resolve_summary(rows: list[SummaryRow], lines_sum: Decimal | None) -> Totals:
    """Derive the invoice totals from the summary rows, checking the kinds with arithmetic.

    Args:
        rows: The parsed rows of the totals block, top to bottom.
        lines_sum: Sum of the line-item amounts, used as the base when no subtotal row is
            printed (None if line amounts are missing).

    Returns:
        A Totals with subtotal, discount, tax, tax_included, charges, total and issues.
    """
    totals = Totals()
    subtotals = [r for r in rows if r.kind == "subtotal"]
    finals = [r for r in rows if r.kind == "total"]
    totals.subtotal = abs(subtotals[0].amount) if subtotals else None
    totals.total = abs(finals[-1].amount) if finals else None

    body = [r for r in rows if r.kind not in ("subtotal", "total", "tax_included")]
    for r in body:
        if r.kind == "charge" and r.amount < 0:
            reason = "marked as a charge but printed negative"
            totals.issues.append(_issue(r, "summary.sign_mismatch", reason))
    base = totals.subtotal if totals.subtotal is not None else lines_sum
    if base is not None and totals.total is not None:
        totals.issues += _resolve_unknown(body, base, totals.total)
    else:
        totals.issues += [
            _issue(r, "summary.unclassified", "no subtotal/total to check it against", error=True)
            for r in body
            if r.kind == "other"
        ]

    discounts = [abs(r.amount) for r in body if r.kind == "discount"]
    taxes = [r.amount for r in body if r.kind == "tax"]
    included = [abs(r.amount) for r in rows if r.kind == "tax_included"]
    totals.discount = sum(discounts, Decimal(0)) if discounts and any(discounts) else None
    totals.tax = sum(taxes, Decimal(0)) if taxes else None
    totals.tax_included = sum(included, Decimal(0)) if included else None
    totals.charges = [
        Charge(label=r.label, kind=r.kind, amount=signed_effect(r))
        for r in body
        if r.kind in ("charge", "deduction", "adjustment")
    ]
    return totals
