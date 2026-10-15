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
    model_config = ConfigDict(extra="forbid")


# ---------- LLM-facing (transcription) ----------


class RawParty(_Strict):
    name: str | None = Field(description="Legal/trading name as printed")
    address: str | None = Field(description="Full address as printed, lines joined with ', '")
    tax_id: str | None = Field(description="GSTIN / VAT / EIN / TIN exactly as printed")


class RawLineItem(_Strict):
    description: str | None
    quantity: str | None = Field(description="As printed, e.g. '2' or '1.5'")
    unit_price: str | None = Field(description="As printed, including separators")
    amount: str | None = Field(description="Line total as printed")


class RawInvoice(_Strict):
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
    tax: str | None = Field(description="Total tax amount (sum of GST/VAT lines)")
    total: str | None = Field(description="Grand total / amount due")
    payment_terms: str | None


# ---------- Normalized (domain) ----------


class Party(BaseModel):
    name: str | None = None
    address: str | None = None
    tax_id: str | None = None


class LineItem(BaseModel):
    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None


class Invoice(BaseModel):
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
    field: str = Field(description="Dotted path, e.g. 'total' or 'line_items[2].amount'")
    code: str = Field(description="Machine-readable, e.g. 'math.total_mismatch'")
    message: str
    severity: Severity = "error"


Route = Literal["fast_path", "agent"]
Status = Literal["accepted", "needs_human_review"]


class ExtractionResult(BaseModel):
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
