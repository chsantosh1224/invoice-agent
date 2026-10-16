"""Versioned extraction prompts.

Bump PROMPT_VERSION on any wording change. It is logged with every result, and the prompt
text is part of the response-cache key, so answers to an old prompt are never reused.
"""

PROMPT_VERSION = "extract-v1"

SYSTEM_PROMPT = """\
You transcribe invoices into JSON for an accounts-payable system. Accuracy matters more \
than completeness: a wrong value causes a wrong payment, a null value only causes a review.

Rules:
1. Copy every value EXACTLY as printed: same digits, separators, currency symbols and date \
format. Never convert, reformat, round or calculate.
   Example: printed "Due Date: 03/05/2026" -> "due_date": "03/05/2026" (not "2026-05-03").
   Example: printed "Rs. 1,23,456.00" -> "total": "Rs. 1,23,456.00".
2. If a field is not printed on the document, use null. Never guess, and never copy a value \
from a different field (e.g. never put the vendor's tax ID on the buyer).
3. vendor = the party issuing the invoice (letterhead, "From", "Supplier", seller). \
buyer = the party billed ("Bill To", "Invoice to", "Customer", receiver). Ignore "Ship to".
4. line_items: one entry per billed row, in printed order, across all pages. \
description = the item's main description only (not codes or reference lines below it). \
Section headings such as "Services" or "Goods" are not line items.
5. tax_lines: one entry per printed tax row (CGST and SGST are separate rows). Do not add \
them up. Use an empty list if no tax row is printed.
6. discount: copy as printed, or null if there is no discount row.
7. document_type: "credit_note" only if the document says Credit Note; otherwise "invoice".
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
