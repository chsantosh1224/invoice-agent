"""The agent's tools. All read-only: they inspect the document, none of them change it.

The same functions are reused by the MCP server (Day 6), so any MCP client can inspect
invoices the way the agent does.
"""

from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from invoice_agent.extraction.agent import workspace
from invoice_agent.extraction.validate import phrases, validate
from invoice_agent.ingest.pages import page_images
from invoice_agent.normalize import to_invoice
from invoice_agent.schemas import RawInvoice
from invoice_agent.textmatch import compact

ZOOM_DPI = 300
_ocr: Any = None


def _ocr_engine() -> Any:
    """Load RapidOCR once, on first use.

    Returns:
        A RapidOCR instance.
    """
    global _ocr
    if _ocr is None:
        from rapidocr import RapidOCR

        _ocr = RapidOCR()
    return _ocr


@tool
def get_page_text(page: int, doc_id: Annotated[str, InjectedState("doc_id")]) -> str:
    """Return the layout-preserving text of one page (columns aligned with spaces).

    Use it to read a page you have not seen yet, e.g. page 2 of a multi-page invoice.

    Args:
        page: 1-based page number.
        doc_id: Injected automatically; do not provide.

    Returns:
        The page text, or an error message if the page does not exist.
    """
    pages = workspace.get(doc_id).doc.pages
    if not 1 <= page <= len(pages):
        return f"No page {page}; the document has {len(pages)} page(s)."
    return pages[page - 1].text


@tool
def search_text(query: str, doc_id: Annotated[str, InjectedState("doc_id")]) -> str:
    """Find every line of the document containing a word, symbol or number.

    Case- and spacing-insensitive. Useful for locating a value or label anywhere in the
    document, e.g. 'GSTIN', 'Total', '$', '€', 'PO', or part of an invoice number.

    Args:
        query: Text to look for.
        doc_id: Injected automatically; do not provide.

    Returns:
        Up to 15 matching lines as 'p<page>: <line>', or a note that nothing matched.
    """
    needle = compact(query)
    hits = [
        f"p{page.number}: {line.strip()}"
        for page in workspace.get(doc_id).doc.pages
        for line in page.text.splitlines()
        if needle and needle in compact(line)
    ]
    return "\n".join(hits[:15]) if hits else f"No line contains {query!r}."


@tool
def reread_region(
    near_text: str, doc_id: Annotated[str, InjectedState("doc_id")], page: int | None = None
) -> str:
    """Re-read the row around a piece of text at 300 DPI with fresh OCR.

    Use it when a value may be misread (blurry scan, a checksum failure, a digit that
    looks wrong): the higher resolution often resolves 0/O, 1/7, 5/S, 8/B confusions.

    Args:
        near_text: Text on the row to re-read, e.g. 'GSTIN' or 'Invoice No'.
        doc_id: Injected automatically; do not provide.
        page: 1-based page to search; all pages if omitted.

    Returns:
        The current text of that row and the high-resolution re-read, for comparison.
    """
    ws = workspace.get(doc_id)
    target = compact(near_text)
    for p in ws.doc.pages:
        if page is not None and p.number != page:
            continue
        match = next((ph for ph in phrases(p.spans) if target in compact(ph.text)), None)
        if match is None:
            continue
        band = (max(0.0, match.y0 - 1.5 * match.height), min(1.0, match.y1 + 1.5 * match.height))
        key = (p.number, round(band[0] * 1000))
        if key not in ws.zoom_cache:
            img = page_images(ws.path, dpi=ZOOM_DPI)[p.number - 1]
            crop = img.crop((0, int(band[0] * img.height), img.width, int(band[1] * img.height)))
            result = _ocr_engine()(crop)
            texts = [] if result.txts is None else list(result.txts)
            ws.zoom_cache[key] = " | ".join(texts) or "(nothing recognized)"
        return f"Current text: {match.text}\nRe-read at {ZOOM_DPI} DPI: {ws.zoom_cache[key]}"
    return f"Could not find {near_text!r} on the page(s)."


class CheckArgs(BaseModel):
    """Arguments for check_draft (doc_id is injected by the graph, never by the model)."""

    draft: RawInvoice = Field(description="The full invoice, every value copied as printed")
    doc_id: Annotated[str, InjectedState("doc_id")]


@tool(args_schema=CheckArgs)
def check_draft(draft: RawInvoice, doc_id: Annotated[str, InjectedState("doc_id")]) -> str:
    """Run the validation gate on a draft before submitting it.

    Checks arithmetic, that every value is printed on the document, labels, GSTIN
    checksums and dates - the same checks your submission will face.

    Args:
        draft: The full invoice as you would submit it.
        doc_id: Injected automatically; do not provide.

    Returns:
        'PASSED' with the score, or the list of remaining problems to fix.
    """
    invoice, issues = to_invoice(draft)
    report = validate(invoice, workspace.get(doc_id).doc, normalize_issues=issues)
    if report.passed:
        return f"PASSED (score {report.score:.2f}). You can submit this draft."
    lines = [f"- [{i.severity}] {i.field}: {i.message}" for i in report.issues]
    return f"NOT PASSED (score {report.score:.2f}):\n" + "\n".join(lines)


class SubmitArgs(BaseModel):
    """Arguments for submit_invoice."""

    draft: RawInvoice = Field(description="The final invoice, every value copied as printed")
    notes: list[str] = Field(description="What you checked or corrected, and how")


@tool(args_schema=SubmitArgs)
def submit_invoice(draft: RawInvoice, notes: list[str]) -> str:
    """Submit the final invoice and finish.

    Call it once check_draft passes, or when you cannot improve the draft further (say so
    in the notes).

    Args:
        draft: The final invoice, every value copied as printed.
        notes: Short audit notes: what was wrong and how you verified the fix.

    Returns:
        A confirmation (the graph intercepts this call; it is never executed).
    """
    return "submitted"


READ_ONLY_TOOLS = [get_page_text, search_text, reread_region, check_draft]
ALL_TOOLS = [*READ_ONLY_TOOLS, submit_invoice]
