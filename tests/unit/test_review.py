import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invoice_agent import api, tracing
from invoice_agent.auth.routes import COOKIE
from invoice_agent.auth.store import AuthStore
from invoice_agent.config import Settings
from invoice_agent.review import routes
from invoice_agent.review.store import ReviewStore
from invoice_agent.schemas import ExtractionResult, Invoice, ValidationIssue

DOC = Path("data/samples/inv-034-minimal-US.pdf")
TRUTH = json.loads(DOC.with_suffix(".truth.json").read_text(encoding="utf-8"))["invoice"]


def truth_invoice(**changes: Any) -> Invoice:
    """The sample's correct invoice, with some fields changed."""
    return Invoice.model_validate(TRUTH | changes)


def flagged(path: Path, settings: Any, **kwargs: Any) -> tuple[ExtractionResult, float]:
    """Stand-in pipeline: the model dropped the currency, so the gate flagged it."""
    issue = ValidationIssue(field="currency", code="required.missing", message="missing")
    result = ExtractionResult(
        status="needs_human_review", route="agent", invoice=truth_invoice(currency=None),
        issues=[issue], score=0.4, model="fake/m", prompt_version="v", run_id="r1",
        trace_id="t" * 32,
    )  # fmt: skip
    return result, 2.0


class ScoreRecorder:
    """Records tracing.score_review calls instead of sending them to Langfuse."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, settings: Settings, trace_id: str | None, outcome: str,
                 fields: int, **extra: Any) -> None:  # fmt: skip
        self.calls.append((trace_id, outcome, fields, extra.get("override")))


@pytest.fixture
def users(tmp_path: Path) -> Iterator[AuthStore]:
    """A throwaway user database with an admin."""
    store = AuthStore(tmp_path / "auth.sqlite")
    store.create_admin("admin@example.com", "Admin", "admin-password-1")
    api.app.state.auth = store
    yield store
    api.app.state.auth = None
    store.close()


@pytest.fixture
def reviews(tmp_path: Path) -> Iterator[ReviewStore]:
    """A throwaway review queue."""
    store = ReviewStore(tmp_path / "data" / "reviews.sqlite")
    api.app.state.reviews = store
    yield store
    api.app.state.reviews = None
    store.close()


@pytest.fixture
def scores(monkeypatch: pytest.MonkeyPatch) -> ScoreRecorder:
    """Capture the scores sent for review decisions."""
    recorder = ScoreRecorder()
    monkeypatch.setattr(tracing, "score_review", recorder)
    return recorder


@pytest.fixture
def admin(users: AuthStore, reviews: ReviewStore, scores: ScoreRecorder,
          monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:  # fmt: skip
    """An admin's browser, with a pipeline that always flags the invoice."""
    monkeypatch.setattr(api, "process_document", flagged)
    with TestClient(api.app) as c:
        c.cookies.set(COOKIE, users.create_session(users.list_users()[0].id))
        yield c


def upload(client: TestClient) -> int:
    """Upload the sample through /extract and return the review id it was queued as."""
    with DOC.open("rb") as fh:
        resp = client.post("/extract", files={"file": ("acme.pdf", fh, "application/pdf")})
    assert resp.status_code == 200 and resp.json()["result"]["status"] == "needs_human_review"
    review_id: int = resp.json()["review_id"]
    return review_id


def approve(client: TestClient, review_id: int, invoice: Invoice, note: str = "") -> Any:
    """Post an approval with the given invoice."""
    body = {"invoice": invoice.model_dump(mode="json"), "note": note}
    return client.post(f"/api/admin/reviews/{review_id}/approve", json=body)


def test_flagged_invoice_is_queued_with_its_file(admin: TestClient, reviews: ReviewStore) -> None:
    """A flagged upload lands in the queue; the reviewer can open it and its document."""
    review_id = upload(admin)
    queue = admin.get("/api/admin/reviews?status=pending").json()
    assert queue["pending"] == 1 and queue["reviews"][0]["uploaded_by"] == "admin@example.com"
    detail = admin.get(f"/api/admin/reviews/{review_id}").json()
    assert detail["issues"] == ["required.missing"] and detail["file_type"] == ".pdf"
    file = admin.get(f"/api/admin/reviews/{review_id}/file")
    assert file.headers["content-type"] == "application/pdf" and file.content == DOC.read_bytes()
    assert admin.get("/admin/reviews").status_code == 200


def test_correction_passes_the_gate_and_feeds_back(
    admin: TestClient, reviews: ReviewStore, scores: ScoreRecorder
) -> None:
    """Approve with the currency fixed: re-checked, recorded, file deleted, scored."""
    review_id = upload(admin)
    resp = approve(admin, review_id, truth_invoice())
    assert resp.status_code == 200, resp.json()
    assert resp.json() == {"ok": True, "changed_fields": ["currency"], "gate_passed": True,
                           "override": False}  # fmt: skip
    review = reviews.get(review_id)
    assert review is not None and review.status == "approved" and not review.has_file
    assert review.corrected is not None and review.corrected.currency == "USD"
    assert scores.calls == [("t" * 32, "corrected", 1, False)]
    assert approve(admin, review_id, truth_invoice()).status_code == 409  # decided once


def test_wrong_correction_needs_a_note_and_is_an_override(
    admin: TestClient, reviews: ReviewStore, scores: ScoreRecorder
) -> None:
    """A total that doesn't match the document is refused, unless the reviewer explains."""
    review_id = upload(admin)
    wrong = truth_invoice(total="1.00")
    refused = approve(admin, review_id, wrong)
    assert refused.status_code == 422
    assert any(i["field"] == "total" for i in refused.json()["detail"]["issues"])
    assert reviews.get(review_id).status == "pending"  # type: ignore[union-attr]

    ok = approve(admin, review_id, wrong, note="Vendor confirmed a credit by phone")
    assert ok.status_code == 200 and ok.json()["override"] is True
    review = reviews.get(review_id)
    assert review is not None and review.final_issues and review.note.startswith("Vendor")
    assert scores.calls[-1][1:] == ("corrected", 2, True)  # currency and total changed


def test_reject_needs_a_reason(
    admin: TestClient, reviews: ReviewStore, scores: ScoreRecorder
) -> None:
    """Rejecting without a reason is refused; with one, the review is closed and scored."""
    review_id = upload(admin)
    assert (
        admin.post(f"/api/admin/reviews/{review_id}/reject", json={"note": ""}).status_code == 422
    )
    resp = admin.post(f"/api/admin/reviews/{review_id}/reject", json={"note": "Duplicate"})
    assert resp.status_code == 200 and reviews.get(review_id).status == "rejected"  # type: ignore[union-attr]
    assert scores.calls == [("t" * 32, "rejected", 0, None)]


def test_non_admins_cannot_review(admin: TestClient, users: AuthStore) -> None:
    """Uploaders don't see the queue: 403 for the API, a redirect for the page."""
    review_id = upload(admin)
    users.request_access("Ann", "ann@example.com", "", "")
    ann = next(u for u in users.list_users() if u.email == "ann@example.com")
    token = users.approve(ann.id)
    assert token is not None and users.accept_invite(token, "ann-password-1") is not None
    admin.cookies.set(COOKIE, users.create_session(ann.id))
    assert admin.get("/api/admin/reviews").status_code == 403
    assert admin.get(f"/api/admin/reviews/{review_id}/file").status_code == 403
    assert admin.get("/admin/reviews", follow_redirects=False).headers["location"] == "/app"


def test_approved_files_can_be_kept_for_evals(tmp_path: Path) -> None:
    """With keep_approved_files, an approved review keeps its document; rejected never."""
    store = ReviewStore(tmp_path / "reviews.sqlite", keep_approved_files=True)
    result, _ = flagged(DOC, None)
    kept = store.add(result, "a.pdf", b"%PDF-1", ".pdf", 1)
    dropped = store.add(result, "b.pdf", b"%PDF-1", ".pdf", 1)
    assert store.decide(kept, "approved", 1, corrected=truth_invoice())
    assert store.decide(dropped, "rejected", 1, note="duplicate")
    assert store.file_of(kept) is not None and store.file_of(dropped) is None
    store.close()


def test_changed_fields_compares_values_not_spelling() -> None:
    """'75' and '75.00' are the same amount; a removed line item counts as a change."""
    before = truth_invoice()
    same = truth_invoice(total=str(before.total) + "0")
    assert routes.changed_fields(before, same) == []
    fewer = truth_invoice(line_items=TRUTH["line_items"][:1])
    assert routes.changed_fields(before, fewer)[0].startswith("line_items[1].")


class FakeLangfuse:
    """Records create_score calls."""

    scores: list[dict[str, Any]] = []  # noqa: RUF012

    def __init__(self, *keys: str) -> None:
        FakeLangfuse.scores = []

    def create_score(self, **score: Any) -> None:
        FakeLangfuse.scores.append(score)


def test_score_review_sends_two_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decision becomes 'human_review' and 'fields_corrected' scores on the trace."""
    from pydantic import SecretStr

    monkeypatch.setattr(tracing, "_client", FakeLangfuse)
    on = Settings(_env_file=None, langfuse_public_key="pk-lf-x",
                  langfuse_secret_key=SecretStr("sk-lf-x"))  # fmt: skip
    tracing.score_review(on, "abc", "corrected", 2, override=True, note="checked")
    assert [(s["name"], s["value"]) for s in FakeLangfuse.scores] == [
        ("human_review", "corrected"), ("fields_corrected", 2.0)]  # fmt: skip
    assert FakeLangfuse.scores[0]["metadata"] == {"override": True}
    tracing.score_review(on, None, "approved", 0)  # tracing was off for that run
    assert len(FakeLangfuse.scores) == 2


def test_export_turns_approvals_into_eval_cases(tmp_path: Path) -> None:
    """An approved review becomes an answer key the extraction eval can score against."""
    from invoice_agent.evals.extraction import score_extraction
    from invoice_agent.review.export import export_reviews

    store = ReviewStore(tmp_path / "reviews.sqlite", keep_approved_files=True)
    result, _ = flagged(DOC, None)
    review_id = store.add(result, "acme.pdf", DOC.read_bytes(), ".pdf", 1)
    store.decide(review_id, "approved", 1, corrected=truth_invoice(), changed_fields=["currency"])
    store.add(result, "waiting.pdf", b"%PDF-1", ".pdf", 1)  # pending: not exported

    written, with_docs, corrected = export_reviews(store, tmp_path / "out")
    assert (written, with_docs, corrected) == (1, 1, {"currency": 1})
    key = json.loads((tmp_path / "out" / "review-0001.truth.json").read_text(encoding="utf-8"))
    assert key["kind"] == "digital" and (tmp_path / "out" / key["file"]).is_file()
    score = score_extraction(truth_invoice(), key["invoice"])
    assert all(score.fields.values()) and score.items_matched == score.items_truth == 3
    store.close()
