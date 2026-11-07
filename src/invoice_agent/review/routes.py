"""The review page and its admin API: list the queue, open one, approve or reject.

    GET  /admin/reviews                       the review page (admins only, see SessionGate)
    GET  /api/admin/reviews?status=pending    the queue, plus the pending count
    GET  /api/admin/reviews/{id}              one review: model output, gate issues, uploader
    GET  /api/admin/reviews/{id}/file         the uploaded invoice, shown beside the form
    POST /api/admin/reviews/{id}/approve      approve the (corrected) invoice
    POST /api/admin/reviews/{id}/reject       reject it, with a reason

Approving re-runs the validation gate on the corrected invoice: a reviewer can't approve
totals that don't add up without writing why (`note`); that is recorded as an override.
Every decision goes to Langfuse as a score on the run's trace (the feedback loop).
"""

import threading
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from invoice_agent import tracing
from invoice_agent.auth.routes import CurrentUser, Store
from invoice_agent.auth.store import AuthStore
from invoice_agent.config import get_settings
from invoice_agent.extraction.validate import GateReport, validate
from invoice_agent.ingest.light import LightIngestor
from invoice_agent.review.store import Review, ReviewStore, Status
from invoice_agent.schemas import Invoice

log = structlog.get_logger()

WEB = Path(__file__).parent.parent / "web"
MEDIA_TYPES = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
               ".jpeg": "image/jpeg"}  # fmt: skip

router = APIRouter()
_store_lock = threading.Lock()


def get_reviews(request: Request) -> ReviewStore:
    """The app's review queue, opened on first use.

    Tests replace it by setting `app.state.reviews` to their own ReviewStore.

    Args:
        request: Any request (gives access to the app).

    Returns:
        The shared ReviewStore.
    """
    with _store_lock:
        store: ReviewStore | None = getattr(request.app.state, "reviews", None)
        if store is None:
            settings = get_settings()
            store = ReviewStore(settings.review_db, keep_approved_files=settings.review_keep_files)
            request.app.state.reviews = store
        return store


Reviews = Annotated[ReviewStore, Depends(get_reviews)]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Turn nested dicts and lists into {'vendor.name': ..., 'line_items[0].amount': ...}.

    Args:
        value: A model_dump() of an Invoice, or a part of one.
        prefix: The path so far.

    Returns:
        Leaf path -> value (an empty list counts as one leaf).
    """
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat |= _flatten(item, f"{prefix}.{key}" if prefix else key)
        return flat
    if isinstance(value, list) and value:
        flat = {}
        for i, item in enumerate(value):
            flat |= _flatten(item, f"{prefix}[{i}]")
        return flat
    return {prefix: value}


def changed_fields(before: Invoice, after: Invoice) -> list[str]:
    """Which fields a reviewer changed.

    Values are compared as numbers and dates, so '75' and '75.00' count as the same.

    Args:
        before: The invoice the pipeline produced.
        after: The invoice the reviewer approved.

    Returns:
        The changed field paths, sorted, e.g. ['currency', 'line_items[1].amount'].
    """
    old, new = _flatten(before.model_dump()), _flatten(after.model_dump())
    return sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))


def recheck(invoice: Invoice, path: Path | None) -> GateReport | None:
    """Run the validation gate on a reviewer's invoice, against the uploaded document.

    Args:
        invoice: The corrected invoice.
        path: The stored upload (None if it is gone).

    Returns:
        The gate's report, or None if the document isn't available to check against.
    """
    if path is None:
        return None
    return validate(invoice, LightIngestor().ingest(path), threshold=get_settings().gate_threshold)


def _summary(review: Review, users: AuthStore) -> dict[str, Any]:
    """A review as the queue lists it.

    Args:
        review: The review.
        users: The user database (for the uploader's email).

    Returns:
        JSON-ready fields for the review page's list.
    """
    uploader = users.get(review.uploaded_by)
    return {
        "id": review.id,
        "status": review.status,
        "filename": review.filename,
        "uploaded_by": uploader.email if uploader else "(deleted user)",
        "created_at": review.created_at,
        "decided_at": review.decided_at,
        "route": review.result.route,
        "score": review.result.score,
        "issues": [i.code for i in review.result.issues if i.severity == "error"],
        "changed_fields": review.changed_fields,
        "override": bool(review.final_issues),
        "note": review.note,
        "has_file": review.has_file,
    }


def _pending(reviews: ReviewStore, review_id: int) -> Review:
    """Fetch a review that still waits for a decision.

    Args:
        reviews: The review queue.
        review_id: The review's id.

    Returns:
        The pending Review.

    Raises:
        HTTPException: 404 if there is no such review; 409 if it was already decided.
    """
    review = reviews.get(review_id)
    if review is None:
        raise HTTPException(404, "No such review")
    if review.status != "pending":
        raise HTTPException(409, f"This invoice was already {review.status}")
    return review


# ----- page -----


@router.get("/admin/reviews", include_in_schema=False)
def review_page() -> FileResponse:
    """The review page: the queue, and one invoice beside its editable fields.

    Returns:
        review.html, not cacheable.
    """
    return FileResponse(
        WEB / "review.html", media_type="text/html", headers={"Cache-Control": "no-store"}
    )


# ----- API -----


@router.get("/api/admin/reviews")
def list_reviews(reviews: Reviews, users: Store, status: Status | None = None) -> dict[str, Any]:
    """The review queue.

    Args:
        reviews: The review queue.
        users: The user database.
        status: Only 'pending', 'approved' or 'rejected' reviews; all if omitted.

    Returns:
        {'reviews': [...], 'pending': number waiting}.
    """
    items = [_summary(r, users) for r in reviews.list_reviews(status)]
    return {"reviews": items, "pending": reviews.count_pending()}


@router.get("/api/admin/reviews/{review_id}")
def get_review(review_id: int, reviews: Reviews, users: Store) -> dict[str, Any]:
    """One review with everything the reviewer needs.

    Args:
        review_id: The review's id.
        reviews: The review queue.
        users: The user database.

    Returns:
        The summary plus 'result' (model output and gate issues), 'corrected',
        'final_issues' and 'file_type' (e.g. '.pdf', or None if the file is gone).

    Raises:
        HTTPException: 404 if there is no such review.
    """
    review = reviews.get(review_id)
    if review is None:
        raise HTTPException(404, "No such review")
    path = reviews.file_of(review_id)
    return _summary(review, users) | {
        "result": review.result.model_dump(mode="json"),
        "corrected": review.corrected.model_dump(mode="json") if review.corrected else None,
        "final_issues": [i.model_dump() for i in review.final_issues],
        "file_type": path.suffix if path else None,
    }


@router.get("/api/admin/reviews/{review_id}/file")
def review_file(review_id: int, reviews: Reviews) -> FileResponse:
    """The uploaded invoice, for viewing in the browser next to the fields.

    Args:
        review_id: The review's id.
        reviews: The review queue.

    Returns:
        The file, shown inline, with its type fixed by its (checked) extension.

    Raises:
        HTTPException: 404 if there is no such review or its file was deleted.
    """
    path = reviews.file_of(review_id)
    if path is None:
        raise HTTPException(404, "The file is no longer stored")
    return FileResponse(
        path, media_type=MEDIA_TYPES[path.suffix], content_disposition_type="inline",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )  # fmt: skip


class ApproveBody(BaseModel):
    """The reviewer's corrected invoice, and why, if they override the gate."""

    invoice: Invoice
    note: str = Field(default="", max_length=1000)


@router.post("/api/admin/reviews/{review_id}/approve")
def approve(
    review_id: int, body: ApproveBody, user: CurrentUser, reviews: Reviews
) -> dict[str, Any]:
    """Approve a flagged invoice, with the reviewer's corrections.

    The gate re-checks the corrected invoice against the document. If it still fails,
    approval needs a note explaining why; it is then recorded as an override.

    Args:
        review_id: The review's id.
        body: The corrected invoice and an optional note.
        user: The deciding admin.
        reviews: The review queue.

    Returns:
        {'ok': True, 'changed_fields': [...], 'gate_passed': bool, 'override': bool}.

    Raises:
        HTTPException: 404/409 (see `_pending`); 422 with the gate's issues if the
            corrected invoice fails the gate and there is no note.
    """
    review = _pending(reviews, review_id)
    report = recheck(body.invoice, reviews.file_of(review_id))
    passed = report is not None and report.passed
    note = body.note.strip()
    if not passed and not note:
        issues = [i.model_dump() for i in report.issues] if report else []
        message = (
            "The corrected invoice still fails validation."
            if report
            else "The document is gone, so the corrections can't be checked."
        )
        raise HTTPException(
            422, {"message": f"{message} Fix it, or add a note to approve anyway.",
                  "issues": issues},
        )  # fmt: skip
    changed = changed_fields(review.result.invoice, body.invoice)
    final_issues = [] if passed or report is None else report.issues
    if not reviews.decide(
        review_id, "approved", user.id, corrected=body.invoice, changed_fields=changed,
        final_issues=final_issues, note=note,
    ):  # fmt: skip
        raise HTTPException(409, "This invoice was already decided")
    outcome = "corrected" if changed else "approved"
    tracing.score_review(
        get_settings(), review.trace_id, outcome, len(changed), override=not passed, note=note
    )
    log.info("review.approved", review_id=review_id, reviewer=user.id, changed=changed,
             override=not passed)  # fmt: skip
    return {"ok": True, "changed_fields": changed, "gate_passed": passed, "override": not passed}


class RejectBody(BaseModel):
    """Why the invoice is rejected (not an invoice, duplicate, unreadable...)."""

    note: str = Field(min_length=3, max_length=1000)


@router.post("/api/admin/reviews/{review_id}/reject")
def reject(review_id: int, body: RejectBody, user: CurrentUser, reviews: Reviews) -> dict[str, Any]:
    """Reject a flagged invoice.

    Args:
        review_id: The review's id.
        body: The reason.
        user: The deciding admin.
        reviews: The review queue.

    Returns:
        {'ok': True}.

    Raises:
        HTTPException: 404/409 (see `_pending`).
    """
    review = _pending(reviews, review_id)
    if not reviews.decide(review_id, "rejected", user.id, note=body.note):
        raise HTTPException(409, "This invoice was already decided")
    tracing.score_review(get_settings(), review.trace_id, "rejected", 0, note=body.note.strip())
    log.info("review.rejected", review_id=review_id, reviewer=user.id)
    return {"ok": True}
