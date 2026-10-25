"""Users and sessions in a small SQLite database.

A user's life: someone submits the request-access form ('pending'); the admin approves
('approved', with a one-time invite link) or rejects ('rejected'); the user opens the
invite link and sets a password; later the admin may disable them ('disabled'). Only
approved users with a password can log in.
"""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from invoice_agent.auth.passwords import (
    DUMMY_HASH,
    hash_password,
    new_token,
    token_hash,
    verify_password,
)

Status = Literal["pending", "approved", "rejected", "disabled"]

SESSION_DAYS = 14  # how long a login lasts
INVITE_DAYS = 7  # how long an invite link stays valid

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    company TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    password_hash TEXT,
    invite_hash TEXT UNIQUE,
    invite_expires TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    last_login TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class User:
    """One account as the app and the admin page see it (never includes secrets)."""

    id: int
    email: str
    name: str
    company: str
    reason: str
    status: Status
    is_admin: bool
    has_password: bool
    invite_pending: bool
    created_at: str
    decided_at: str | None
    last_login: str | None


def _now() -> datetime:
    """The current time in UTC.

    Returns:
        A timezone-aware datetime.
    """
    return datetime.now(UTC)


def _iso(when: datetime) -> str:
    """Format a time for the database.

    Args:
        when: A timezone-aware datetime.

    Returns:
        ISO 8601 text to the second, e.g. '2026-10-25T09:30:00+00:00'.
    """
    return when.isoformat(timespec="seconds")


def normalize_email(email: str) -> str:
    """Make emails comparable: 'Ann@X.com ' and 'ann@x.com' are the same account.

    Args:
        email: An email address as typed.

    Returns:
        The address trimmed and lower-cased.
    """
    return email.strip().lower()


def _user(row: sqlite3.Row) -> User:
    """Convert a database row into a User.

    Args:
        row: A row from the users table.

    Returns:
        The User, with password and invite reduced to yes/no flags.
    """
    return User(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        company=row["company"],
        reason=row["reason"],
        status=row["status"],
        is_admin=bool(row["is_admin"]),
        has_password=row["password_hash"] is not None,
        invite_pending=row["invite_hash"] is not None,
        created_at=row["created_at"],
        decided_at=row["decided_at"],
        last_login=row["last_login"],
    )


class AuthStore:
    """All reads and writes of users and sessions.

    One SQLite connection shared by FastAPI's worker threads, guarded by a lock: the
    demo has a handful of users, so serializing database calls costs nothing.
    """

    def __init__(self, path: Path) -> None:
        """Open (and if needed create) the database.

        Args:
            path: The SQLite file; its folder is created if missing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
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

    # ----- access requests and admin decisions -----

    def request_access(self, name: str, email: str, company: str, reason: str) -> bool:
        """Record a request from the public form.

        Args:
            name: The requester's name.
            email: Their email (normalized here).
            company: Their company, or ''.
            reason: Why they want access, or ''.

        Returns:
            True if this is a new request; False if the email is already known (the
            existing account is left untouched).
        """
        cur = self._write(
            "INSERT OR IGNORE INTO users (email, name, company, reason, status, created_at)"
            " VALUES (?, ?, ?, ?, 'pending', ?)",
            (normalize_email(email), name.strip(), company.strip(), reason.strip(), _iso(_now())),
        )
        return cur.rowcount == 1

    def count_pending(self) -> int:
        """Count requests waiting for a decision.

        Returns:
            The number of pending users.
        """
        row = self._one("SELECT COUNT(*) AS n FROM users WHERE status = 'pending'")
        return int(row["n"]) if row else 0

    def list_users(self) -> list[User]:
        """All users for the admin page: pending first, then newest first.

        Returns:
            The users.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM users ORDER BY status != 'pending', created_at DESC, id DESC"
            ).fetchall()
        return [_user(r) for r in rows]

    def get(self, user_id: int) -> User | None:
        """Look up a user by id.

        Args:
            user_id: The user's id.

        Returns:
            The User, or None if there is no such id.
        """
        row = self._one("SELECT * FROM users WHERE id = ?", (user_id,))
        return _user(row) if row else None

    def approve(self, user_id: int) -> str | None:
        """Approve a user (or re-enable one) and issue a fresh one-time invite.

        Also used to resend an invite or to let a user reset a forgotten password: their
        old password keeps working until the new invite is used.

        Args:
            user_id: The user's id.

        Returns:
            The invite token (put it in the link sent to the user), or None if there is
            no such user. Only the token's hash is stored.
        """
        token = new_token()
        cur = self._write(
            "UPDATE users SET status = 'approved', decided_at = ?, invite_hash = ?,"
            " invite_expires = ? WHERE id = ?",
            (
                _iso(_now()),
                token_hash(token),
                _iso(_now() + timedelta(days=INVITE_DAYS)),
                user_id,
            ),
        )
        return token if cur.rowcount == 1 else None

    def set_status(self, user_id: int, status: Literal["rejected", "disabled"]) -> bool:
        """Reject or disable a user: cancels any invite and logs them out everywhere.

        Args:
            user_id: The user's id.
            status: 'rejected' (a request turned down) or 'disabled' (access removed).

        Returns:
            True if the user existed.
        """
        cur = self._write(
            "UPDATE users SET status = ?, decided_at = ?, invite_hash = NULL,"
            " invite_expires = NULL WHERE id = ?",
            (status, _iso(_now()), user_id),
        )
        self._write("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return cur.rowcount == 1

    def create_admin(self, email: str, name: str, password: str) -> User:
        """Create the admin account, or promote and reset an existing one.

        Args:
            email: The admin's email.
            name: The admin's name.
            password: The admin's password (hashed here).

        Returns:
            The admin User.
        """
        email = normalize_email(email)
        now = _iso(_now())
        self._write(
            "INSERT INTO users (email, name, status, is_admin, password_hash, created_at,"
            " decided_at) VALUES (?, ?, 'approved', 1, ?, ?, ?)"
            " ON CONFLICT(email) DO UPDATE SET name = excluded.name, status = 'approved',"
            " is_admin = 1, password_hash = excluded.password_hash",
            (email, name.strip(), hash_password(password), now, now),
        )
        row = self._one("SELECT * FROM users WHERE email = ?", (email,))
        assert row is not None
        return _user(row)

    # ----- invites, logins and sessions -----

    def accept_invite(self, token: str, password: str) -> User | None:
        """Set a password from an invite link; the link then stops working.

        Args:
            token: The token from the invite link.
            password: The new password (hashed here).

        Returns:
            The User, or None if the link is unknown, used, expired or the user is no
            longer approved.
        """
        row = self._one(
            "SELECT * FROM users WHERE invite_hash = ? AND status = 'approved'",
            (token_hash(token),),
        )
        if row is None or row["invite_expires"] < _iso(_now()):
            return None
        self._write(
            "UPDATE users SET password_hash = ?, invite_hash = NULL, invite_expires = NULL"
            " WHERE id = ?",
            (hash_password(password), row["id"]),
        )
        return self.get(row["id"])

    def authenticate(self, email: str, password: str) -> User | None:
        """Check a login.

        Args:
            email: The email typed on the login page.
            password: The password typed.

        Returns:
            The User if the email belongs to an approved user and the password matches;
            otherwise None. Takes the same time whether or not the email exists.
        """
        row = self._one("SELECT * FROM users WHERE email = ?", (normalize_email(email),))
        stored = row["password_hash"] if row is not None and row["password_hash"] else DUMMY_HASH
        if not verify_password(password, stored) or row is None or row["status"] != "approved":
            return None
        self._write("UPDATE users SET last_login = ? WHERE id = ?", (_iso(_now()), row["id"]))
        return _user(row)

    def create_session(self, user_id: int) -> str:
        """Start a login session.

        Args:
            user_id: The logged-in user's id.

        Returns:
            The session token for the cookie. Only its hash is stored.
        """
        token = new_token()
        self._write("DELETE FROM sessions WHERE expires < ?", (_iso(_now()),))
        self._write(
            "INSERT INTO sessions (token_hash, user_id, expires) VALUES (?, ?, ?)",
            (token_hash(token), user_id, _iso(_now() + timedelta(days=SESSION_DAYS))),
        )
        return token

    def session_user(self, token: str) -> User | None:
        """Find who a session cookie belongs to.

        Args:
            token: The cookie's value.

        Returns:
            The User if the session exists, hasn't expired and the user is still
            approved; otherwise None.
        """
        row = self._one(
            "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id"
            " WHERE sessions.token_hash = ? AND sessions.expires > ?"
            " AND users.status = 'approved'",
            (token_hash(token), _iso(_now())),
        )
        return _user(row) if row else None

    def end_session(self, token: str) -> None:
        """Log out one session.

        Args:
            token: The cookie's value.

        Returns:
            None.
        """
        self._write("DELETE FROM sessions WHERE token_hash = ?", (token_hash(token),))
