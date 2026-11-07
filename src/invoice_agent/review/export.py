"""Turn approved reviews into eval cases: the feedback loop from human review to evals.

Each approved review becomes `review-0007.truth.json` in the same format as the
synthetic answer keys in data/samples, with the reviewer's corrected invoice as the
truth. If its document was kept (REVIEW_KEEP_FILES=true), it is copied beside it, so
`invoice-agent eval` can re-run real mistakes as regression tests. The summary of which
fields reviewers corrected most shows where the prompts or the gate need work.
"""

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from invoice_agent.ingest.pages import detect_kind
from invoice_agent.review.store import Review, ReviewStore
from invoice_agent.schemas import Invoice

# The answer-key fields the extraction eval scores (see data/samples/*.truth.json).
TRUTH_FIELDS = ["document_type", "invoice_number", "invoice_date", "due_date", "po_number",
                "currency", "vendor", "buyer", "line_items", "subtotal", "discount", "tax",
                "total", "payment_terms"]  # fmt: skip
ITEM_FIELDS = ["description", "quantity", "unit_price", "amount"]


def truth_of(invoice: Invoice) -> dict[str, Any]:
    """An invoice in the answer-key format.

    Args:
        invoice: The reviewer's corrected invoice.

    Returns:
        The 'invoice' object of a .truth.json file (amounts and dates as text).
    """
    data = invoice.model_dump(mode="json")
    truth = {k: data[k] for k in TRUTH_FIELDS}
    truth["line_items"] = [{k: item[k] for k in ITEM_FIELDS} for item in data["line_items"]]
    return truth


def export_case(review: Review, document: Path | None, out: Path) -> bool:
    """Write one approved review as an eval case.

    Args:
        review: An approved review (with its corrected invoice).
        document: Its stored file, if kept; copied as the case's document.
        out: The output folder.

    Returns:
        True if the document was included (the case can be re-run), False if only the
        answer key was written.
    """
    assert review.corrected is not None, "only approved reviews are exported"
    case = f"review-{review.id:04d}"
    key = {
        "id": case,
        "file": f"{case}{document.suffix}" if document else None,
        "kind": detect_kind(document) if document else "unknown",
        "split": "review",
        "source": "human_review",
        "changed_fields": review.changed_fields,
        "override": bool(review.final_issues),
        "model": review.result.model,
        "prompt_version": review.result.prompt_version,
        "invoice": truth_of(review.corrected),
    }
    (out / f"{case}.truth.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    if document is not None:
        shutil.copyfile(document, out / f"{case}{document.suffix}")
    return document is not None


def export_reviews(store: ReviewStore, out: Path) -> tuple[int, int, Counter[str]]:
    """Export every approved review.

    Args:
        store: The review queue.
        out: The output folder (created if missing).

    Returns:
        (cases written, cases with their document, how often each field was corrected).
        Line items are counted per column: 'line_items[3].amount' -> 'line_items[].amount'.
    """
    out.mkdir(parents=True, exist_ok=True)
    written = with_docs = 0
    corrected: Counter[str] = Counter()
    for review in store.list_reviews("approved"):
        with_docs += export_case(review, store.file_of(review.id), out)
        written += 1
        corrected.update(re.sub(r"\[\d+\]", "[]", f) for f in review.changed_fields)
    return written, with_docs, corrected
