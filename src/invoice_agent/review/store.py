"""The human review queue in a small SQLite database, with the flagged files beside it.

A review's life: the gate flags an extraction ('pending', the uploaded file is saved);
an admin corrects and approves it ('approved') or rejects it ('rejected'). Deciding
deletes the file, unless the settings keep approved files for building eval data.
"""

import json
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from invoice_agent.db import SQLiteStore, iso, now
from invoice_agent.schemas import ExtractionResult, Invoice, ValidationIssue

Status = Literal["pending", "approved", "rejected"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    filename TEXT NOT NULL,
    stored_file TEXT,
    uploaded_by INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    trace_id TEXT,
    result_json TEXT NOT NULL,
    corrected_json TEXT,
    changed_fields TEXT NOT NULL DEFAULT '[]',
    final_issues TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    reviewer_id INTEGER,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS reviews_status ON reviews (status, created_at);
"""


@dataclass(frozen=True)
class Review:
    """One flagged extraction and, once decided, what the reviewer did with it.

    Attributes:
        result: The extraction as the pipeline returned it (model output + gate issues).
        corrected: The invoice the reviewer approved (None while pending or if rejected).
        changed_fields: Field paths the reviewer changed, e.g. ['total', 'currency'].
        final_issues: Gate issues still present on the approved invoice (an override).
        note: The reviewer's note (required to override the gate or to reject).
        has_file: Whether the uploaded file is still stored.
    """

    id: int
    status: Status
    filename: str
    uploaded_by: int
    run_id: str
    trace_id: str | None
    result: ExtractionResult
    corrected: Invoice | None
    changed_fields: list[str]
    final_issues: list[ValidationIssue]
    note: str
    reviewer_id: int | None
    created_at: str
    decided_at: str | None
    has_file: bool


class ReviewStore(SQLiteStore):
    """All reads and writes of the review queue (see `SQLiteStore` for threading)."""

    def __init__(self, path: Path, *, keep_approved_files: bool = False) -> None:
        """Open (and if needed create) the database and the folder for uploaded files.

        Args:
            path: The SQLite file; uploaded files go in 'review-files' next to it.
            keep_approved_files: Keep an approved review's file instead of deleting it.
        """
        super().__init__(path, _SCHEMA)
        self.files = path.parent / "review-files"
        self.files.mkdir(parents=True, exist_ok=True)
        self.keep_approved_files = keep_approved_files

    def _review(self, row: sqlite3.Row) -> Review:
        """Convert a database row into a Review.

        Args:
            row: A row from the reviews table.

        Returns:
            The Review, with its JSON columns parsed.
        """
        corrected = row["corrected_json"]
        return Review(
            id=row["id"],
            status=row["status"],
            filename=row["filename"],
            uploaded_by=row["uploaded_by"],
            run_id=row["run_id"],
            trace_id=row["trace_id"],
            result=ExtractionResult.model_validate_json(row["result_json"]),
            corrected=Invoice.model_validate_json(corrected) if corrected else None,
            changed_fields=json.loads(row["changed_fields"]),
            final_issues=[ValidationIssue(**i) for i in json.loads(row["final_issues"])],
            note=row["note"],
            reviewer_id=row["reviewer_id"],
            created_at=row["created_at"],
            decided_at=row["decided_at"],
            has_file=self._stored(row) is not None,
        )

    def _stored(self, row: sqlite3.Row) -> Path | None:
        """Where a review's uploaded file is stored, if it still is.

        Args:
            row: A row from the reviews table.

        Returns:
            The file's path, or None if it was deleted (or never saved).
        """
        name = row["stored_file"]
        path = self.files / name if name else None
        return path if path is not None and path.is_file() else None

    def add(
        self, result: ExtractionResult, filename: str, data: bytes, suffix: str, uploaded_by: int
    ) -> int:
        """Put a flagged extraction in the queue, saving its file for the reviewer.

        Args:
            result: The pipeline's result (status 'needs_human_review').
            filename: The name the uploader gave the file (shown to the reviewer only).
            data: The uploaded file's bytes.
            suffix: Its checked extension, e.g. '.pdf' (the stored name is random).
            uploaded_by: The uploader's user id.

        Returns:
            The new review's id.
        """
        stored = f"{secrets.token_hex(16)}{suffix}"
        (self.files / stored).write_bytes(data)
        cur = self._write(
            "INSERT INTO reviews (status, filename, stored_file, uploaded_by, run_id, trace_id,"
            " result_json, created_at) VALUES ('pending', ?, ?, ?, ?, ?, ?, ?)",
            (filename[:200], stored, uploaded_by, result.run_id, result.trace_id,
             result.model_dump_json(), iso(now())),
        )  # fmt: skip
        return int(cur.lastrowid or 0)

    def count_pending(self) -> int:
        """How many reviews wait for a decision.

        Returns:
            The number of pending reviews.
        """
        row = self._one("SELECT COUNT(*) AS n FROM reviews WHERE status = 'pending'")
        return int(row["n"]) if row else 0

    def list_reviews(self, status: Status | None = None) -> list[Review]:
        """Reviews for the review page: oldest pending first, decided ones newest first.

        Args:
            status: Only reviews with this status; None for all.

        Returns:
            The reviews.
        """
        order = "ORDER BY status != 'pending', CASE WHEN status = 'pending' THEN id END, id DESC"
        if status is None:
            rows = self._all(f"SELECT * FROM reviews {order}")
        else:
            rows = self._all(f"SELECT * FROM reviews WHERE status = ? {order}", (status,))
        return [self._review(r) for r in rows]

    def get(self, review_id: int) -> Review | None:
        """One review by id.

        Args:
            review_id: The review's id.

        Returns:
            The Review, or None if there is none with that id.
        """
        row = self._one("SELECT * FROM reviews WHERE id = ?", (review_id,))
        return self._review(row) if row else None

    def file_of(self, review_id: int) -> Path | None:
        """Where a review's uploaded file is stored, if it still is.

        Args:
            review_id: The review's id.

        Returns:
            The file's path, or None.
        """
        row = self._one("SELECT stored_file FROM reviews WHERE id = ?", (review_id,))
        return self._stored(row) if row else None

    def decide(
        self,
        review_id: int,
        status: Literal["approved", "rejected"],
        reviewer_id: int,
        *,
        corrected: Invoice | None = None,
        changed_fields: list[str] | None = None,
        final_issues: list[ValidationIssue] | None = None,
        note: str = "",
    ) -> bool:
        """Record the reviewer's decision on a pending review and delete its file.

        Args:
            review_id: The review's id.
            status: 'approved' (with the corrected invoice) or 'rejected'.
            reviewer_id: The deciding admin's user id.
            corrected: The approved invoice.
            changed_fields: Field paths the reviewer changed.
            final_issues: Gate issues the reviewer overrode.
            note: The reviewer's note.

        Returns:
            True if the review was pending and is now decided; False if it doesn't
            exist or was already decided (so two admins can't both decide it).
        """
        path = self.file_of(review_id)
        cur = self._write(
            "UPDATE reviews SET status = ?, corrected_json = ?, changed_fields = ?,"
            " final_issues = ?, note = ?, reviewer_id = ?, decided_at = ?"
            " WHERE id = ? AND status = 'pending'",
            (status, corrected.model_dump_json() if corrected else None,
             json.dumps(changed_fields or []),
             json.dumps([i.model_dump() for i in final_issues or []]),
             note.strip(), reviewer_id, iso(now()), review_id),
        )  # fmt: skip
        if cur.rowcount == 0:
            return False
        if path is not None and not (status == "approved" and self.keep_approved_files):
            path.unlink(missing_ok=True)
            self._write("UPDATE reviews SET stored_file = NULL WHERE id = ?", (review_id,))
        return True
