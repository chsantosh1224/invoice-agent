"""The validation gate: deterministic checks that decide whether to trust an extraction.

The LLM is the only probabilistic step in the pipeline, so its output is verified by code
against independent evidence - the document's own text and word positions:

- required:  the fields AP cannot work without are present
- math:      qty x price = amount, sum of lines = subtotal, subtotal - discount + tax = total
- grounding: every extracted value actually appears on the page
- labels:    a value sits next to the right label (catches swapped dates / totals)
- format:    GSTIN checksum, due date not before invoice date, sane years, distinct parties

Errors fail the gate. Each field also gets a confidence; the invoice's score is the lowest
confidence among its critical fields. The model's self-reported confidence is never used.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal

from invoice_agent.ingest.document import Document, Span
from invoice_agent.ingest.layout import group_rows
from invoice_agent.schemas import Invoice, Severity, ValidationIssue
from invoice_agent.synth.formatting import DATE_FORMATS, format_date
from invoice_agent.tax_ids import is_valid_gstin
from invoice_agent.textmatch import compact, date_present, numbers_in

DEFAULT_THRESHOLD = 0.8
LINE_TOLERANCE = Decimal("0.011")  # one cent of rounding per line
TOTAL_TOLERANCE = Decimal("0.02")
REQUIRED = ("invoice_number", "invoice_date", "vendor.name", "currency", "total")
CRITICAL = (*REQUIRED, "due_date", "po_number", "subtotal", "tax", "discount")
_GSTIN_LIKE = re.compile(r"^\d{2}[0-9A-Z]{13}$")

# Label phrases per field. Order matters: earlier fields win when a phrase matches several
# ('Total Taxable Value' is a subtotal, 'Amount due' is a total not a due date, and
# 'Due Date' must not count as an invoice-date label).
LABELS: dict[str, re.Pattern[str]] = {
    "subtotal": re.compile(r"\b(sub\s?-?total|net amount|taxable value|goods & services)\b"),
    "total": re.compile(r"\b(total|amount due|balance due)\b"),
    "due_date": re.compile(r"\b(due date|payment due|due)\b"),
    "invoice_date": re.compile(r"\b(invoice date|document date|issued|date)\b"),
    "po_number": re.compile(r"\b(p\.?\s?o\.?|purchase order|order)\b"),
    "invoice_number": re.compile(r"\b(invoice (no|number|#)|credit note no|number|no\.?)\b"),
}
COMPETING = ({"invoice_date", "due_date"}, {"subtotal", "total"})
BUYER_LABEL = re.compile(
    r"\b(bill(ed)? to|invoice to|customer|receiver|sold to|buyer)\b", re.IGNORECASE
)


@dataclass
class GateReport:
    """The gate's verdict on one extraction.

    Attributes:
        issues: Problems found; any with severity 'error' fails the gate.
        field_confidence: Field path -> confidence in 0..1.
        score: Lowest confidence among the critical fields that are present.
        passed: True if there are no errors and score >= the threshold.
    """

    issues: list[ValidationIssue] = field(default_factory=list)
    field_confidence: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    passed: bool = False

    @property
    def errors(self) -> list[ValidationIssue]:
        """Issues with severity 'error'.

        Returns:
            The subset of issues that fail the gate.
        """
        return [i for i in self.issues if i.severity == "error"]


def _flat_fields(inv: Invoice) -> dict[str, object]:
    """Flatten the header fields of an invoice into path -> value.

    Args:
        inv: The normalized invoice.

    Returns:
        e.g. {'invoice_number': 'INV-1', 'vendor.name': 'Acme', 'total': Decimal(...)}.
    """
    flat: dict[str, object] = {
        k: getattr(inv, k)
        for k in (
            "invoice_number",
            "invoice_date",
            "due_date",
            "po_number",
            "currency",
            "subtotal",
            "discount",
            "tax",
            "total",
        )
    }
    for party in ("vendor", "buyer"):
        for k in ("name", "tax_id"):
            flat[f"{party}.{k}"] = getattr(getattr(inv, party), k)
    return flat


# ---------- phrase spans and label proximity ----------


def phrases(spans: list[Span]) -> list[Span]:
    """Merge word spans into phrases: neighbours on the same row with a small gap.

    PDF text layers give one span per word ('Invoice', 'No:', 'INV-1'); OCR already gives
    phrases. Labels like 'Due Date' need to be one span to be recognized.

    Args:
        spans: Spans of one page.

    Returns:
        Phrase spans with merged text and bounding boxes.
    """
    merged: list[Span] = []
    for row in group_rows(spans):
        current = row[0]
        for s in row[1:]:
            if s.x0 - current.x1 < 1.2 * max(current.height, s.height):
                current = Span(
                    f"{current.text} {s.text}",
                    current.x0,
                    min(current.y0, s.y0),
                    s.x1,
                    max(current.y1, s.y1),
                    min(current.conf, s.conf),
                )
            else:
                merged.append(current)
                current = s
        merged.append(current)
    return merged


def _label_of(text: str) -> str | None:
    """Classify a phrase as a field label.

    Args:
        text: Phrase text.

    Returns:
        The field whose label pattern matches first (see LABELS order), or None.
    """
    lowered = text.lower()
    return next((f for f, pattern in LABELS.items() if pattern.search(lowered)), None)


def _contains_value(name: str, value: object, text: str) -> bool:
    """Check whether a phrase contains a field's value.

    Args:
        name: Field path.
        value: Normalized value (date, Decimal or str).
        text: Phrase text.

    Returns:
        True if the value is printed in the phrase (any date format / exact amount / ID).
    """
    if name.endswith("date"):
        return date_present(value, compact(text))  # type: ignore[arg-type]
    if isinstance(value, Decimal):
        return abs(value) in numbers_in(text)
    return compact(str(value)) in compact(text)


def _nearest_label(value_span: Span, labels: list[tuple[Span, str]]) -> str | None:
    """Find the label a value belongs to: on its left in the same row, else just above it.

    A label on the same row always wins over one above (in a two-column meta table the
    'Date' label of the previous row is physically close to the 'Due Date' value).

    Args:
        value_span: The phrase holding the value.
        labels: (label phrase, field) pairs on the same page.

    Returns:
        The field of the nearest label, or None if no label is close enough.
    """
    h = max(value_span.height, 1e-6)
    left = [
        (value_span.x0 - span.x1, fld)
        for span, fld in labels
        if abs(span.cy - value_span.cy) < 0.6 * h and span.x1 <= value_span.x0 + h
    ]
    if left:
        return min(left)[1]
    above = [
        (value_span.y0 - span.y1, fld)
        for span, fld in labels
        if 0 < value_span.y0 - span.y1 < 3 * h and abs(span.x0 - value_span.x0) < 0.08
    ]
    return min(above)[1] if above else None


def label_field_for(name: str, value: object, doc: Document) -> str | bool | None:
    """Work out which label the printed value of a field sits next to.

    Args:
        name: Field path (only fields in LABELS are checked).
        value: Its normalized value.
        doc: The ingested document.

    Returns:
        True if at least one occurrence sits next to its own label; otherwise the field
        of the nearest foreign label (e.g. 'invoice_date' for a misassigned due date),
        or None if no occurrence has a label nearby.
    """
    foreign: str | None = None
    for page in doc.pages:
        page_phrases = phrases(page.spans)
        labels = [(p, f) for p in page_phrases if (f := _label_of(p.text))]
        for p in page_phrases:
            if not _contains_value(name, value, p.text):
                continue
            own = _label_of(_strip_value(name, value, p.text))
            found = own or _nearest_label(p, labels)
            if found == name:
                return True
            foreign = foreign or found
    return foreign


def _strip_value(name: str, value: object, text: str) -> str:
    """Remove the printed value from a phrase, leaving only a possible label.

    Args:
        name: Field path.
        value: Normalized value.
        text: Phrase text, e.g. 'Due Date: 2026-01-20'.

    Returns:
        The phrase without the value, e.g. 'Due Date:'.
    """
    if name.endswith("date"):
        for fmt in DATE_FORMATS:
            text = text.replace(format_date(value, fmt), "")  # type: ignore[arg-type]
        return text
    return re.sub(r"[\d.,]+", "", text) if isinstance(value, Decimal) else text


def _only_under_buyer_label(name: str, doc: Document) -> bool:
    """Check whether a company name appears only in the buyer block of the page.

    The real vendor is in the letterhead / 'From' / 'Supplier' area; a 'vendor' found only
    directly under 'Bill To' / 'Invoice to' / 'Customer' was read from the wrong block.

    Args:
        name: The extracted vendor name.
        doc: The ingested document.

    Returns:
        True if the name occurs, and every occurrence sits right below or beside a
        buyer-side label.
    """
    seen = under_buyer = 0
    target = compact(name)
    for page in doc.pages:
        page_phrases = phrases(page.spans)
        buyer_labels = [p for p in page_phrases if BUYER_LABEL.search(p.text)]
        for p in page_phrases:
            if target not in compact(p.text):
                continue
            seen += 1
            h = max(p.height, 1e-6)
            if BUYER_LABEL.search(p.text) or any(
                (0 < p.y0 - b.y1 < 3 * h and abs(b.x0 - p.x0) < 0.08)
                or (abs(b.cy - p.cy) < 0.6 * h and b.x1 <= p.x0 + h)
                for b in buyer_labels
            ):
                under_buyer += 1
    return seen > 0 and seen == under_buyer


# ---------- the gate ----------


class _Findings:
    """Collects the gate's issues and per-field confidence while the checks run."""

    def __init__(self, issues: list[ValidationIssue], conf: dict[str, float]) -> None:
        """Start from issues already known (from normalization) and initial confidences.

        Args:
            issues: Issues found so far.
            conf: Field path -> confidence (1.0 for every field that has a value).
        """
        self.issues = issues
        self.conf = conf

    def flag(
        self, fld: str, code: str, message: str, *, error: bool = True, cap: float = 0.2
    ) -> None:
        """Record an issue and cap the field's confidence.

        Args:
            fld: Field path the issue is about.
            code: Machine-readable issue code, e.g. 'math.total'.
            message: Human-readable explanation.
            error: True for an error (fails the gate), False for a warning.
            cap: Maximum confidence the field may keep after this issue.

        Returns:
            None.
        """
        severity: Severity = "error" if error else "warning"
        self.issues.append(
            ValidationIssue(field=fld, code=code, message=message, severity=severity)
        )
        if fld in self.conf:
            self.conf[fld] = min(self.conf[fld], cap)


def validate(
    inv: Invoice,
    doc: Document,
    *,
    normalize_issues: list[ValidationIssue] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> GateReport:
    """Run every gate check on an extraction and compute field confidences.

    Args:
        inv: The normalized invoice from the LLM.
        doc: The same document as ingested (text + word positions), the evidence.
        normalize_issues: Issues already found while normalizing (they count as errors).
        threshold: Minimum score to pass (calibrated on the dev set).

    Returns:
        A GateReport with issues, per-field confidence, score and the pass/fail verdict.
    """
    flat = _flat_fields(inv)
    scanned = doc.kind == "scanned"
    findings = _Findings(
        issues=list(normalize_issues or []),
        conf={k: 1.0 for k, v in flat.items() if v is not None},
    )
    text = doc.text
    flat_text, nums = compact(text), numbers_in(text)

    # required
    for fld in REQUIRED:
        if flat.get(fld) is None:
            findings.flag(fld, "required.missing", f"{fld} is missing")

    # math
    reconciled: set[str] = set()
    for i, li in enumerate(inv.line_items):
        q, p, a = li.quantity, li.unit_price, li.amount
        if q is not None and p is not None and a is not None and abs(q * p - a) > LINE_TOLERANCE:
            findings.flag(f"line_items[{i}].amount", "math.line", f"{q} x {p} != {a}")
    amounts = [li.amount for li in inv.line_items if li.amount is not None]
    if inv.line_items and len(amounts) == len(inv.line_items):
        lines_sum = sum(amounts, Decimal(0))
        if inv.subtotal is not None:
            if abs(lines_sum - inv.subtotal) > TOTAL_TOLERANCE:
                findings.flag(
                    "subtotal",
                    "math.subtotal",
                    f"line items sum to {lines_sum}, subtotal is {inv.subtotal}",
                )
            else:
                reconciled.add("subtotal")
        base = inv.subtotal if inv.subtotal is not None else lines_sum
        if inv.total is not None:
            expected = base - (inv.discount or Decimal(0)) + (inv.tax or Decimal(0))
            if abs(expected - inv.total) > TOTAL_TOLERANCE:
                findings.flag(
                    "total",
                    "math.total",
                    f"subtotal - discount + tax = {expected}, total is {inv.total}",
                )
            else:
                reconciled |= {"total", "tax", "discount"} | ({"subtotal"} & reconciled)

    # grounding: every value must be printed somewhere in the document
    for fld, value in flat.items():
        if value is None:
            continue
        if fld.endswith("date"):
            grounded = date_present(value, flat_text)  # type: ignore[arg-type]
        elif isinstance(value, Decimal):
            grounded = abs(value) in nums
        elif fld.endswith(".name"):
            tokens = [t for t in re.split(r"\W+", str(value).lower()) if len(t) > 1]
            grounded = sum(t in flat_text for t in tokens) >= 0.8 * max(len(tokens), 1)
        elif fld == "currency":
            continue  # usually printed as a symbol; validated by the total's currency
        else:
            grounded = compact(str(value)) in flat_text
        if grounded:
            continue
        if fld in ("tax",) and fld in reconciled:
            continue  # tax is the sum of printed CGST+SGST rows; math already confirms it
        if scanned:
            # OCR may have misread what the vision model read correctly from the image.
            findings.flag(
                fld,
                "grounding.ocr_mismatch",
                f"{fld}={value!r} not found in OCR text",
                error=False,
                cap=0.5 if fld not in reconciled else 0.95,
            )
        else:
            findings.flag(
                fld,
                "grounding.missing",
                f"{fld}={value!r} is not printed on the document",
                error=fld not in reconciled and not fld.endswith(".name"),
                cap=0.3,
            )

    # labels: values must sit next to their own label, not a competing one
    for fld in ("invoice_date", "due_date", "total"):
        value = flat.get(fld)
        if value is None:
            continue
        group = next(g for g in COMPETING if fld in g)
        other = next(iter(group - {fld}))
        if flat.get(other) == value:
            continue  # identical values (e.g. total == subtotal) cannot be told apart
        found = label_field_for(fld, value, doc)
        if isinstance(found, str) and found in group and found != fld:
            findings.flag(
                fld, "label.mismatch", f"{fld} value is printed next to the '{found}' label"
            )

    # format
    for party in ("vendor", "buyer"):
        tax_id = flat.get(f"{party}.tax_id")
        if isinstance(tax_id, str):
            compact_id = re.sub(r"\s", "", tax_id.upper())
            if _GSTIN_LIKE.match(compact_id) and not is_valid_gstin(compact_id):
                findings.flag(
                    f"{party}.tax_id", "format.gstin_checksum", f"{tax_id} fails the checksum"
                )
    if flat.get("buyer.tax_id") and flat.get("buyer.tax_id") == flat.get("vendor.tax_id"):
        findings.flag("buyer.tax_id", "format.same_tax_id", "buyer has the vendor's tax ID")
    vendor, buyer = inv.vendor.name, inv.buyer.name
    if vendor and buyer and compact(vendor) == compact(buyer):
        findings.flag("vendor.name", "party.same_name", "vendor and buyer have the same name")
    elif vendor and _only_under_buyer_label(vendor, doc):
        findings.flag(
            "vendor.name",
            "party.vendor_under_bill_to",
            f"vendor {vendor!r} only appears under a Bill-To/Customer label",
        )
    if inv.invoice_date and inv.due_date and inv.due_date < inv.invoice_date:
        findings.flag(
            "due_date",
            "format.due_before_invoice",
            f"due date {inv.due_date} is before invoice date {inv.invoice_date}",
        )
    for fld in ("invoice_date", "due_date"):
        d = getattr(inv, fld)
        if d is not None and not 2000 <= d.year <= 2100:
            findings.flag(fld, "format.implausible_year", f"{fld} year {d.year} is implausible")

    conf = findings.conf
    for fld in reconciled & conf.keys():
        conf[fld] = max(conf[fld], 0.95)  # arithmetic corroborates these amounts
    if scanned:
        conf = {k: min(v, doc.ocr_confidence) if v == 1.0 else v for k, v in conf.items()}
    critical = [conf[k] for k in CRITICAL if k in conf]
    score = min(critical) if critical else 0.0
    report = GateReport(issues=findings.issues, field_confidence=conf, score=score)
    report.passed = not report.errors and score >= threshold
    return report
