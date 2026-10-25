from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invoice_agent import api
from invoice_agent.auth import routes
from invoice_agent.auth.passwords import hash_password, verify_password
from invoice_agent.auth.routes import COOKIE
from invoice_agent.auth.store import AuthStore, _iso, _now
from invoice_agent.auth.throttle import LoginThrottle

ADMIN_PASSWORD = "admin-password-1"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[AuthStore]:
    """A throwaway user database with one admin, used by the app for this test."""
    s = AuthStore(tmp_path / "auth.sqlite")
    s.create_admin("Admin@Example.com", "Admin", ADMIN_PASSWORD)
    api.app.state.auth = s
    yield s
    api.app.state.auth = None
    s.close()


@pytest.fixture
def client(store: AuthStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A logged-out browser, with a fresh login throttle."""
    monkeypatch.setattr(routes, "throttle", LoginThrottle())
    with TestClient(api.app) as c:
        yield c


def login(client: TestClient, email: str, password: str) -> int:
    """Log in through the API and return the status code."""
    return client.post("/api/login", json={"email": email, "password": password}).status_code


def request_access(client: TestClient, email: str, **extra: str) -> int:
    """Submit the public request-access form and return the status code."""
    body = {"name": "Ann Client", "email": email, "company": "Acme", "reason": "Try it"} | extra
    return client.post("/api/access-requests", json=body).status_code


# ---------- passwords and store ----------


def test_password_hash_round_trip_and_salting() -> None:
    """The right password verifies, a wrong one doesn't, and equal passwords hash differently."""
    stored = hash_password("correct horse")
    assert verify_password("correct horse", stored)
    assert not verify_password("correct horsE", stored)
    assert stored != hash_password("correct horse")
    assert not verify_password("x", "not-a-hash")


def test_expired_invite_is_refused(store: AuthStore) -> None:
    """An invite link older than its expiry can't be used."""
    store.request_access("Ann", "ann@example.com", "", "")
    user = next(u for u in store.list_users() if u.email == "ann@example.com")
    token = store.approve(user.id)
    assert token is not None
    store._write(
        "UPDATE users SET invite_expires = ? WHERE id = ?",
        (_iso(_now() - timedelta(minutes=1)), user.id),
    )
    assert store.accept_invite(token, "a-long-password") is None


# ---------- the gate ----------


def test_logged_out_visitors_are_sent_to_login(client: TestClient) -> None:
    """Pages redirect to /login, API calls get 401, public pages stay open."""
    page = client.get("/", follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/login"
    assert client.post("/extract").status_code == 401
    assert client.get("/api/me").status_code == 401
    for path in ("/login", "/request-access", "/invite", "/health", "/static/auth.css"):
        assert client.get(path).status_code == 200, path


# ---------- the full journey ----------


def test_request_approve_invite_login_journey(client: TestClient, store: AuthStore) -> None:
    """Request -> admin approves -> invite sets password -> user logs in -> admin disables."""
    assert request_access(client, "Ann@Example.com") == 202
    assert login(client, "ann@example.com", "anything-at-all") == 401  # not approved yet

    assert login(client, "admin@example.com", ADMIN_PASSWORD) == 200  # email case ignored
    users = client.get("/api/admin/users").json()["users"]
    ann = next(u for u in users if u["email"] == "ann@example.com")
    assert ann["status"] == "pending" and ann["company"] == "Acme"
    assert "password_hash" not in ann and "invite_hash" not in ann  # secrets never leave

    invite_url = client.post(f"/api/admin/users/{ann['id']}/approve").json()["invite_url"]
    assert "/invite#" in invite_url
    token = invite_url.split("#", 1)[1]
    client.post("/api/logout")
    client.cookies.clear()

    accepted = client.post(
        "/api/invite/accept", json={"token": token, "password": "ann-password-1"}
    )
    assert accepted.status_code == 200 and COOKIE in accepted.cookies
    assert client.get("/api/me").json() == {
        "name": "Ann Client",
        "email": "ann@example.com",
        "is_admin": False,
    }
    reuse = client.post("/api/invite/accept", json={"token": token, "password": "other-password"})
    assert reuse.status_code == 400  # one-time link

    # A normal user can't use the admin API.
    assert client.get("/api/admin/users").status_code == 403
    assert client.get("/admin", follow_redirects=False).headers["location"] == "/"

    # Disabling logs Ann out everywhere and blocks new logins.
    store.set_status(ann["id"], "disabled")
    assert client.get("/api/me").status_code == 401
    assert login(client, "ann@example.com", "ann-password-1") == 401


def test_logout_ends_the_session(client: TestClient) -> None:
    """After logout the old cookie no longer works."""
    assert login(client, "admin@example.com", ADMIN_PASSWORD) == 200
    old_cookie = client.cookies[COOKIE]
    client.post("/api/logout")
    client.cookies.set(COOKIE, old_cookie)
    assert client.get("/api/me").status_code == 401


# ---------- abuse cases ----------


def test_repeat_and_bot_requests_get_the_same_answer(client: TestClient, store: AuthStore) -> None:
    """A known email and a filled honeypot both get 202 but create nothing new."""
    assert request_access(client, "admin@example.com") == 202
    assert request_access(client, "bot@example.com", website="http://spam") == 202
    assert [u.email for u in store.list_users()] == ["admin@example.com"]
    assert request_access(client, "not-an-email") == 422


def test_too_many_failed_logins_are_throttled(client: TestClient) -> None:
    """Five wrong passwords lock that email for a while, even for the right password."""
    for _ in range(5):
        assert login(client, "admin@example.com", "wrong-password") == 401
    assert login(client, "admin@example.com", ADMIN_PASSWORD) == 429


def test_admin_cannot_lock_themselves_out(client: TestClient, store: AuthStore) -> None:
    """Disabling your own admin account is refused."""
    assert login(client, "admin@example.com", ADMIN_PASSWORD) == 200
    admin_id = store.list_users()[0].id
    assert client.post(f"/api/admin/users/{admin_id}/disable").status_code == 400
