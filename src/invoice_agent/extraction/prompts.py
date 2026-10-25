"""Versioned extraction prompts.

Bump PROMPT_VERSION on any wording change. It is logged with every result, and the prompt
text is part of the response-cache key, so answers to an old prompt are never reused.
"""

PROMPT_VERSION = "extract-v2"

SYSTEM_PROMPT = """\
You transcribe invoices into JSON for an accounts-payable system. Accuracy matters more \
than completeness: a wrong value causes a wrong payment, a null value only causes a review.

Rules:
1. Copy every value EXACTLY as printed: same digits, separators, currency symbols, signs, \
and date format (including any time). Never convert, reformat, round or calculate.
   Example: printed "Due Date: 03/05/2026" -> "due_date": "03/05/2026" (not "2026-05-03").
   Example: printed "Dec 11, 2020, 10:56:14 AM" -> "Dec 11, 2020, 10:56:14 AM".
   Example: printed "Rs. 1,23,456.00" -> "Rs. 1,23,456.00".

2. If a field is not printed on the document, use null. Never guess, and never copy a value \
from a different field (e.g. never put the vendor's tax ID on the buyer).

3. currency: the currency code or symbol as printed. If no currency is stated separately, \
copy the symbol printed on the amounts (e.g. "$", "€"). Never write a code such as "USD" \
that is not printed on the document.

4. vendor = the party issuing the invoice (letterhead, "From", "Supplier", "Sold by"). \
buyer = the party billed ("Bill To", "Invoice to", "Customer"). Ignore "Ship to".

5. document_type - what the document is:
   "invoice"     a bill requesting payment (tax invoice, sales invoice, bill)
   "credit_note" reduces an amount owed (credit note, credit memo)
   "debit_note"  increases an amount owed for an earlier invoice
   "proforma"    proforma invoice / advance bill, not a final invoice
   "quote"       quotation, estimate, offer
   "receipt"     confirms payment already made (payment receipt, cash memo)
   "statement"   statement of account listing several invoices
   "other"       anything else (purchase order, delivery note, ...)

6. line_items: one entry per billed row, in printed order, across all pages.
   description = the item's main description only (not SKU, codes or reference lines).
   Section headings ("Services", "Goods") are not line items.
   quantity, unit_price: null if the column is not printed (e.g. a lump-sum row).
   discount, tax_rate, tax_amount: copy the row's own discount / tax columns if printed, \
otherwise null.

7. summary_lines: every row of the totals block, top to bottom, exactly as printed - \
including subtotal and grand total. Give each row a kind:
   "subtotal"     sum of the line items before anything else (Subtotal, Net amount, \
Taxable value)
   "discount"     a price reduction (Discount, Promo, Rebate)
   "tax"          tax added on top (VAT 20%, CGST @ 9%, Sales tax, TCS)
   "tax_included" tax already inside the prices, shown for information \
("incl. VAT 19%", "GST included")
   "charge"       an extra cost added (Shipping, Freight, Handling, Surcharge, Fee)
   "deduction"    subtracted but not a discount (Less advance paid, TDS withheld, Deposit)
   "adjustment"   a small +/- correction (Round off, Rounding)
   "total"        the final amount payable (Total, Grand total, Amount due, Net payable)
   "other"        you cannot tell - use this instead of guessing
   Do not add rows up and do not add rows that are not printed. Ignore amounts in words.
"""

TEXT_INTRO = (
    "Text extracted from the document by OCR/PDF parsing, with the page layout preserved "
    "(columns are aligned with spaces). It may contain recognition errors{image_hint}."
)


def user_instructions(*, has_image: bool, has_text: bool) -> str:
    """Build the task sentence for the user message, matching what inputs are provided.

    Args:
        has_image: Whether page images are attached.
        has_text: Whether extracted document text is attached.

    Returns:
        A short instruction telling the model what it is looking at.

    Raises:
        ValueError: If neither images nor text are provided.
    """
    if has_image and has_text:
        return (
            "Transcribe this invoice. The page images are the source of truth; the extracted "
            "text below helps you read exact characters."
        )
    if has_image:
        return "Transcribe this invoice from the page images."
    if has_text:
        return "Transcribe this invoice from its extracted text."
    raise ValueError("extraction needs page images, text, or both")
