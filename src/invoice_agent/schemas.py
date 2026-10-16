"""Data contracts.

Two layers, on purpose:
- `Raw*` models are what the LLM fills in: every value transcribed *exactly as printed*
  (strings, nullable, no extra keys) so the JSON schema stays strict-mode friendly and the
  grounding check can look each value up in the OCR text.
- `Invoice` is the typed, normalized result produced by deterministic code (normalize.py).
"""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Base for LLM-facing models: rejects unknown keys (additionalProperties: false)."""

    model_config = ConfigDict(extra="forbid")


# ---------- LLM-facing (transcription) ----------


class RawParty(_Strict):
    """A company on the invoice. Copy each value exactly as printed; use null if absent."""

    name: str | None = Field(description="Legal/trading name as printed")
    address: str | None = Field(description="Full address as printed, lines joined with ', '")
    tax_id: str | None = Field(description="GSTIN / VAT / EIN / TIN exactly as printed")


class RawLineItem(_Strict):
    """One row of the line-item table. Copy values exactly as printed; null if a cell is empty."""

    description: str | None
    quantity: str | None = Field(description="As printed, e.g. '2' or '1.5'")
    unit_price: str | None = Field(description="As printed, including separators")
    amount: str | None = Field(description="Line total as printed")


class RawTaxLine(_Strict):
    """One tax line from the totals block, e.g. 'CGST @ 9%' or 'VAT 19%'. Copy as printed."""

    label: str | None = Field(description="Tax name and rate as printed, e.g. 'IGST @ 18%'")
    amount: str | None = Field(description="Tax amount as printed")


class RawInvoice(_Strict):
    """The full invoice transcribed exactly as printed.

    Do not calculate, convert or reformat anything: keep separators, symbols and date
    formats as they appear. Use null for any field that is not on the document.
    """

    document_type: Literal["invoice", "credit_note"]
    invoice_number: str | None
    invoice_date: str | None = Field(description="Exactly as printed, e.g. '03/04/2026'")
    due_date: str | None
    po_number: str | None = Field(description="Purchase order number; null if absent")
    currency: str | None = Field(description="ISO code or symbol as printed, e.g. 'INR' or '₹'")
    vendor: RawParty = Field(description="The party issuing the invoice (Bill From / Seller)")
    buyer: RawParty = Field(description="The party being billed (Bill To / Buyer)")
    line_items: list[RawLineItem]
    subtotal: str | None
    discount: str | None
    tax_lines: list[RawTaxLine] = Field(
        description="Every tax line in the totals block (CGST and SGST are separate lines); "
        "empty list if no tax is shown. Do not add them up."
    )
    total: str | None = Field(description="Grand total / amount due")
    payment_terms: str | None


# ---------- Normalized (domain) ----------


class Party(BaseModel):
    """A normalized vendor or buyer (whitespace cleaned, None where blank)."""

    name: str | None = None
    address: str | None = None
    tax_id: str | None = None


class LineItem(BaseModel):
    """A normalized line item; numbers are exact Decimals."""

    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None


class Invoice(BaseModel):
    """The normalized invoice: Decimal money, real dates, ISO 4217 currency.

    Produced from a RawInvoice by `normalize.to_invoice`; this is what validation,
    matching and storage work with.
    """

    document_type: Literal["invoice", "credit_note"] = "invoice"
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    po_number: str | None = None
    currency: str | None = Field(default=None, description="ISO 4217")
    vendor: Party = Field(default_factory=Party)
    buyer: Party = Field(default_factory=Party)
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = None
    discount: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None
    payment_terms: str | None = None


# ---------- Validation and results ----------

Severity = Literal["error", "warning"]


class ValidationIssue(BaseModel):
    """One problem found by normalization or the validation gate."""

    field: str = Field(description="Dotted path, e.g. 'total' or 'line_items[2].amount'")
    code: str = Field(description="Machine-readable, e.g. 'math.total_mismatch'")
    message: str
    severity: Severity = "error"


Route = Literal["fast_path", "agent"]
Status = Literal["accepted", "needs_human_review"]


class ExtractionResult(BaseModel):
    """The final output of extraction for one document, including its audit trail."""

    status: Status
    route: Route
    invoice: Invoice
    raw: RawInvoice | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    score: float = 0.0
    notes: list[str] = Field(default_factory=list, description="Agent audit trail")
    model: str
    prompt_version: str
    run_id: str
