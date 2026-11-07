"""A small SQLite helper shared by the app's stores (accounts, reviews)."""

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path


def now() -> datetime:
    """The current time in UTC.

    Returns:
        A timezone-aware datetime.
    """
    return datetime.now(UTC)


def iso(when: datetime) -> str:
    """Format a time for the database.

    Args:
        when: A timezone-aware datetime.

    Returns:
        ISO 8601 text to the second, e.g. '2026-10-25T09:30:00+00:00'.
    """
    return when.isoformat(timespec="seconds")


class SQLiteStore:
    """One SQLite connection shared by FastAPI's worker threads, guarded by a lock.

    The demo has a handful of users, so serializing database calls costs nothing.
    Subclasses pass their CREATE TABLE statements as `schema`.
    """

    def __init__(self, path: Path, schema: str) -> None:
        """Open (and if needed create) the database.

        Args:
            path: The SQLite file; its folder is created if missing.
            schema: SQL run at startup (CREATE TABLE IF NOT EXISTS ...).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(schema)
        self._lock = threading.Lock()

    def close(self) -> None:
        """Close the database connection.

        Returns:
            None.
        """
        self._db.close()

    def _write(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        """Run one statement that changes data, and commit it.

        Args:
            sql: The SQL statement.
            params: Values for its '?' placeholders.

        Returns:
            The cursor (for rowcount / lastrowid).
        """
        with self._lock, self._db:
            return self._db.execute(sql, params)

    def _one(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
        """Run a query and return its first row.

        Args:
            sql: The SQL query.
            params: Values for its '?' placeholders.

        Returns:
            The first row, or None if there is none.
        """
        with self._lock:
            row: sqlite3.Row | None = self._db.execute(sql, params).fetchone()
            return row

    def _all(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        """Run a query and return all its rows.

        Args:
            sql: The SQL query.
            params: Values for its '?' placeholders.

        Returns:
            The rows (possibly empty).
        """
        with self._lock:
            return self._db.execute(sql, params).fetchall()
