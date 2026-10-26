from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from invoice_agent import api
from invoice_agent.auth import routes
from invoice_agent.auth.mailer import Email, Mailer, MailError
from invoice_agent.auth.passwords import hash_password, verify_password
from invoice_agent.auth.routes import COOKIE
from invoice_agent.auth.store import AuthStore, _iso, _now
from invoice_agent.auth.throttle import LoginThrottle
from invoice_agent.config import Settings

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


class FakeMailer(Mailer):
    """Records emails instead of sending them; can be switched off or made to fail."""

    def __init__(self, enabled: bool = False, fail: bool = False) -> None:
        super().__init__(Settings())
        self.on, self.fail, self.sent = enabled, fail, []  # type: ignore[var-annotated]

    @property
    def enabled(self) -> bool:
        return self.on

    def send(self, email: Email) -> None:
        if self.fail:
            raise MailError("SMTPAuthenticationError: bad key")
        self.sent.append(email)


@pytest.fixture
def mailer() -> Iterator[FakeMailer]:
    """The app's mailer for this test: off unless a test turns it on."""
    fake = FakeMailer()
    api.app.state.mailer = fake
    yield fake
    api.app.state.mailer = None


@pytest.fixture
def client(
    store: AuthStore, mailer: FakeMailer, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
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


def test_logged_out_visitors_see_home_and_are_sent_to_login(client: TestClient) -> None:
    """Home is public; the app redirects to the login pop-up; API calls get 401."""
    for path in ("/", "/request-access", "/invite", "/health", "/static/auth.css"):
        assert client.get(path).status_code == 200, path
    for path in ("/app", "/login", "/admin"):
        page = client.get(path, follow_redirects=False)
        assert page.status_code == 303 and page.headers["location"] == "/?login", path
    assert client.post("/extract").status_code == 401
    assert client.get("/api/me").status_code == 401


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
    assert client.get("/admin", follow_redirects=False).headers["location"] == "/app"
    assert client.get("/login", follow_redirects=False).headers["location"] == "/app"

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


# ---------- email ----------


def pending_id(store: AuthStore, email: str) -> int:
    """The id of the user with this email."""
    return next(u.id for u in store.list_users() if u.email == email)


def test_request_emails_the_admins(client: TestClient, mailer: FakeMailer) -> None:
    """A new request sends each admin a notice with the details and the admin-page link."""
    mailer.on = True
    assert request_access(client, "ann@example.com", name="Ann\nClient") == 202
    [notice] = mailer.sent
    assert notice.to == "admin@example.com"
    assert notice.subject == "Access request: Ann Client"  # no line break in the header
    assert "ann@example.com" in notice.text and "http://testserver/admin" in notice.text
    assert request_access(client, "ann@example.com") == 202  # repeat: no second email
    assert len(mailer.sent) == 1


def test_approval_emails_the_invite_but_new_links_do_not(
    client: TestClient, store: AuthStore, mailer: FakeMailer
) -> None:
    """Approve -> invite emailed; 'new invite link' later -> shown to the admin only."""
    request_access(client, "ann@example.com")
    mailer.on = True
    login(client, "admin@example.com", ADMIN_PASSWORD)
    ann = pending_id(store, "ann@example.com")
    first = client.post(f"/api/admin/users/{ann}/approve").json()
    assert first["emailed"] is True and first["email_error"] is None
    [invite] = mailer.sent
    assert invite.to == "ann@example.com" and first["invite_url"] in invite.text
    assert first["invite_url"] in invite.html
    again = client.post(f"/api/admin/users/{ann}/approve").json()
    assert again["emailed"] is False and len(mailer.sent) == 1


def test_failed_email_still_gives_the_admin_the_link(
    client: TestClient, store: AuthStore, mailer: FakeMailer
) -> None:
    """If the provider refuses, approval still succeeds and the admin sees why."""
    request_access(client, "ann@example.com")
    mailer.on, mailer.fail = True, True
    login(client, "admin@example.com", ADMIN_PASSWORD)
    resp = client.post(f"/api/admin/users/{pending_id(store, 'ann@example.com')}/approve")
    body = resp.json()
    assert resp.status_code == 200 and "/invite#" in body["invite_url"]
    assert body["emailed"] is False and "bad key" in body["email_error"]


def test_invite_email_escapes_names() -> None:
    """A name with HTML in it is shown as text in the email, not as markup."""
    from invoice_agent.auth.mailer import invite_email
    from invoice_agent.auth.store import User

    user = User(1, "x@example.com", "<b>Eve</b>", "", "", "approved", False, False, True,
                "2026-01-01", None, None)  # fmt: skip
    email = invite_email(user, "https://site/invite#tok")
    assert "<b>Eve</b>" not in email.html and "&lt;b&gt;Eve&lt;/b&gt;" in email.html


class FakeSMTP:
    """Stands in for smtplib.SMTP and records what the mailer does."""

    calls: ClassVar[list[str]] = []

    def __init__(self, host: str, port: int, timeout: float) -> None:
        FakeSMTP.calls = [f"connect {host}:{port}"]

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *exc: object) -> None:
        FakeSMTP.calls.append("quit")

    def starttls(self, context: object) -> None:
        FakeSMTP.calls.append("starttls")

    def login(self, user: str, password: str) -> None:
        FakeSMTP.calls.append(f"login {user}")

    def send_message(self, msg: object) -> None:
        FakeSMTP.calls.append(f"send to {msg['To']} from {msg['From']}")  # type: ignore[index]


def test_mailer_uses_starttls_and_logs_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Port 587: connect, upgrade to TLS, log in, send, quit - with the right From."""
    from pydantic import SecretStr

    monkeypatch.setattr("invoice_agent.auth.mailer.smtplib.SMTP", FakeSMTP)
    settings = Settings(
        smtp_host="smtp.resend.com",
        smtp_user="resend",
        smtp_password=SecretStr("re_x"),
        mail_from="invites@aicruit.in",
    )
    Mailer(settings).send(Email("ann@example.com", "Hi", "text", "<p>html</p>"))
    assert FakeSMTP.calls == [
        "connect smtp.resend.com:587", "starttls", "login resend",
        "send to ann@example.com from Invoice Agent <invites@aicruit.in>", "quit",
    ]  # fmt: skip
    assert not Mailer(Settings(smtp_host="")).enabled
