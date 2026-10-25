"""Login, access requests, invites and the admin page, plus the gate that guards the app.

    Pages (HTML)                 API (JSON)
    GET / (home, login pop-up)   POST /api/login, POST /api/logout, GET /api/me
    GET /request-access          POST /api/access-requests
    GET /invite#<token>          POST /api/invite/accept
    GET /admin (admin only)      GET /api/admin/users, POST /api/admin/users/{id}/<action>

Everything else (the app at /app, /extract) requires a logged-in user; see `SessionGate`.
/login is kept for old links and redirects to the home page's login pop-up.
"""

import re
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from invoice_agent.auth.passwords import MIN_PASSWORD_LENGTH
from invoice_agent.auth.store import SESSION_DAYS, AuthStore, User, normalize_email
from invoice_agent.auth.throttle import LoginThrottle
from invoice_agent.config import get_settings

log = structlog.get_logger()

WEB = Path(__file__).parent.parent / "web"
COOKIE = "ia_session"
LOGIN_URL = "/?login"  # the home page, with its login pop-up open
MAX_PENDING = 200  # stop accepting requests if this many are waiting (spam guard)
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Reachable without logging in. Everything else needs a session.
PUBLIC_PATHS = {
    "/", "/health", "/login", "/request-access", "/invite", "/static/auth.css",
    "/api/login", "/api/logout", "/api/access-requests", "/api/invite/accept",
}  # fmt: skip

router = APIRouter()
throttle = LoginThrottle()
_store_lock = threading.Lock()


def get_store(request: Request) -> AuthStore:
    """The app's user database, opened on first use.

    Tests replace it by setting `app.state.auth` to their own AuthStore.

    Args:
        request: Any request (gives access to the app).

    Returns:
        The shared AuthStore.
    """
    with _store_lock:
        store: AuthStore | None = getattr(request.app.state, "auth", None)
        if store is None:
            store = AuthStore(get_settings().auth_db)
            request.app.state.auth = store
        return store


def current_user(request: Request) -> User:
    """The logged-in user, as found by `SessionGate`.

    Args:
        request: The current request.

    Returns:
        The User.

    Raises:
        HTTPException: 401 if nobody is logged in.
    """
    user: User | None = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(401, "Please log in")
    return user


Store = Annotated[AuthStore, Depends(get_store)]
CurrentUser = Annotated[User, Depends(current_user)]


class SessionGate(BaseHTTPMiddleware):
    """Find the user from the session cookie and keep everyone else out.

    Logged-out visitors are sent to the home page's login pop-up (pages) or get 401 (API
    calls); non-admins can't reach /admin or /api/admin/*.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Attach `request.state.user`, then allow, redirect or refuse the request.

        Args:
            request: The incoming request.
            call_next: Starlette's handle to the rest of the app.

        Returns:
            The app's response, a redirect to the login pop-up (or /app), or a 401/403
            JSON error.
        """
        token = request.cookies.get(COOKIE)
        user = get_store(request).session_user(token) if token else None
        request.state.user = user
        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)
        is_page = request.method == "GET" and not path.startswith("/api/")
        if user is None:
            if is_page:
                return RedirectResponse(LOGIN_URL, status_code=303)
            return JSONResponse({"detail": "Please log in"}, status_code=401)
        if (path == "/admin" or path.startswith("/api/admin/")) and not user.is_admin:
            if is_page:
                return RedirectResponse("/app", status_code=303)
            return JSONResponse({"detail": "Admins only"}, status_code=403)
        return await call_next(request)


def _page(name: str) -> FileResponse:
    """Serve one of the HTML pages in web/.

    Args:
        name: The file name, e.g. 'login.html'.

    Returns:
        The page, marked not cacheable so updates show at once.
    """
    return FileResponse(WEB / name, media_type="text/html", headers={"Cache-Control": "no-store"})


def _start_session(request: Request, response: Response, user: User, store: AuthStore) -> None:
    """Log a user in: create a session and set its cookie on the response.

    Args:
        request: The current request (tells whether the site is on HTTPS).
        response: The response that will carry the cookie.
        user: The user to log in.
        store: The user database.

    Returns:
        None.
    """
    response.set_cookie(
        COOKIE, store.create_session(user.id), max_age=SESSION_DAYS * 86400, path="/",
        httponly=True,  # page scripts can't read it
        samesite="lax",  # other sites can't make the browser send it with a POST
        secure=request.url.scheme == "https",  # never sent unencrypted on the live site
    )  # fmt: skip


# ----- pages -----


@router.get("/login", include_in_schema=False)
def login_page(request: Request) -> RedirectResponse:
    """Old login address: send people to the login pop-up (or the app, if logged in).

    Args:
        request: The current request.

    Returns:
        A redirect to /app if a session is active, otherwise to the home page's pop-up.
    """
    if getattr(request.state, "user", None) is not None:
        return RedirectResponse("/app", status_code=303)
    return RedirectResponse(LOGIN_URL, status_code=303)


@router.get("/request-access", include_in_schema=False)
def request_page() -> FileResponse:
    """The public form for asking for access.

    Returns:
        request.html.
    """
    return _page("request.html")


@router.get("/invite", include_in_schema=False)
def invite_page() -> FileResponse:
    """The page where an invited user sets a password (the token is in the URL #fragment).

    Returns:
        invite.html.
    """
    return _page("invite.html")


@router.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    """The admin page: requests and users (admins only, enforced by SessionGate).

    Returns:
        admin.html.
    """
    return _page("admin.html")


@router.get("/static/auth.css", include_in_schema=False)
def auth_css() -> FileResponse:
    """The stylesheet shared by the login, request, invite and admin pages.

    Returns:
        auth.css.
    """
    return FileResponse(WEB / "auth.css", media_type="text/css")


# ----- login -----


class LoginBody(BaseModel):
    """Login form."""

    email: str = Field(max_length=254)
    password: str = Field(max_length=200)


@router.post("/api/login")
def login(body: LoginBody, request: Request, response: Response, store: Store) -> dict[str, Any]:
    """Log in with email and password.

    Args:
        body: The email and password.
        request: The current request.
        response: The response (receives the session cookie).
        store: The user database.

    Returns:
        {'ok': True} with the session cookie set.

    Raises:
        HTTPException: 429 after too many failed attempts for this email; 401 for a
            wrong email/password or an account that isn't approved (one message for
            all, so it doesn't reveal which emails have accounts).
    """
    key = normalize_email(body.email)
    if throttle.blocked(key):
        raise HTTPException(429, "Too many attempts. Please wait 15 minutes and try again.")
    user = store.authenticate(body.email, body.password)
    if user is None:
        throttle.failed(key)
        log.info("auth.login_failed")
        raise HTTPException(401, "Wrong email or password")
    throttle.succeeded(key)
    _start_session(request, response, user, store)
    log.info("auth.login", user_id=user.id)
    return {"ok": True}


@router.post("/api/logout")
def logout(request: Request, response: Response, store: Store) -> dict[str, Any]:
    """Log out: end the session and clear the cookie.

    Args:
        request: The current request (carries the cookie).
        response: The response (clears the cookie).
        store: The user database.

    Returns:
        {'ok': True}, whether or not a session existed.
    """
    token = request.cookies.get(COOKIE)
    if token:
        store.end_session(token)
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.get("/api/me")
def me(user: CurrentUser) -> dict[str, Any]:
    """Who is logged in, for the page header.

    Args:
        user: The logged-in user.

    Returns:
        {'name': ..., 'email': ..., 'is_admin': ...}.
    """
    return {"name": user.name, "email": user.email, "is_admin": user.is_admin}


# ----- access requests and invites -----


class AccessRequestBody(BaseModel):
    """The public request-access form."""

    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=254)
    company: str = Field(default="", max_length=100)
    reason: str = Field(default="", max_length=1000)
    website: str = Field(default="", max_length=200)  # hidden field: only bots fill it in


@router.post("/api/access-requests", status_code=202)
def request_access(body: AccessRequestBody, store: Store) -> dict[str, Any]:
    """Record a request for access, for the admin to approve.

    Args:
        body: The form fields.
        store: The user database.

    Returns:
        {'ok': True}: the same answer for new, repeated and bot requests, so the form
        doesn't reveal who already has an account.

    Raises:
        HTTPException: 422 for an invalid email; 503 if too many requests are waiting.
    """
    if not EMAIL.match(body.email.strip()):
        raise HTTPException(422, "Please enter a valid email address")
    if body.website:
        log.info("auth.request_honeypot")
        return {"ok": True}
    if store.count_pending() >= MAX_PENDING:
        raise HTTPException(503, "We aren't taking new requests right now. Please try later.")
    if store.request_access(body.name, body.email, body.company, body.reason):
        log.info("auth.access_requested")
    return {"ok": True}


class InviteBody(BaseModel):
    """The set-password form on the invite page."""

    token: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=200)


@router.post("/api/invite/accept")
def accept_invite(
    body: InviteBody, request: Request, response: Response, store: Store
) -> dict[str, Any]:
    """Set a password from an invite link and log in.

    Args:
        body: The invite token and the chosen password.
        request: The current request.
        response: The response (receives the session cookie).
        store: The user database.

    Returns:
        {'ok': True} with the session cookie set.

    Raises:
        HTTPException: 400 if the link is invalid, already used or expired.
    """
    user = store.accept_invite(body.token, body.password)
    if user is None:
        raise HTTPException(400, "This invite link is invalid, used or expired. Ask for a new one.")
    _start_session(request, response, user, store)
    log.info("auth.invite_accepted", user_id=user.id)
    return {"ok": True}


# ----- admin -----


@router.get("/api/admin/users")
def admin_users(store: Store) -> dict[str, Any]:
    """All users and requests, for the admin page (SessionGate allows admins only).

    Args:
        store: The user database.

    Returns:
        {'users': [...]} with each user's details (never passwords or tokens).
    """
    return {"users": [asdict(u) for u in store.list_users()]}


@router.post("/api/admin/users/{user_id}/approve")
def admin_approve(user_id: int, request: Request, store: Store) -> dict[str, Any]:
    """Approve a request (or re-enable a user, or resend an invite) and get the invite link.

    Args:
        user_id: The user to approve.
        request: The current request (its address is used to build the link).
        store: The user database.

    Returns:
        {'invite_url': 'https://<this site>/invite#<token>'}: send it to the user. The
        token sits after '#', so browsers never send it to the server in logs or
        Referer headers.

    Raises:
        HTTPException: 404 if there is no such user.
    """
    token = store.approve(user_id)
    if token is None:
        raise HTTPException(404, "No such user")
    log.info("auth.approved", user_id=user_id)
    return {"invite_url": f"{str(request.base_url).rstrip('/')}/invite#{token}"}


@router.post("/api/admin/users/{user_id}/reject")
def admin_reject(user_id: int, admin: CurrentUser, store: Store) -> dict[str, Any]:
    """Turn down a request.

    Args:
        user_id: The user to reject.
        admin: The logged-in admin.
        store: The user database.

    Returns:
        {'ok': True}.

    Raises:
        HTTPException: 400 for the admin's own account; 404 if there is no such user.
    """
    return _set_status(user_id, "rejected", admin, store)


@router.post("/api/admin/users/{user_id}/disable")
def admin_disable(user_id: int, admin: CurrentUser, store: Store) -> dict[str, Any]:
    """Remove a user's access and log them out everywhere.

    Args:
        user_id: The user to disable.
        admin: The logged-in admin.
        store: The user database.

    Returns:
        {'ok': True}.

    Raises:
        HTTPException: 400 for the admin's own account; 404 if there is no such user.
    """
    return _set_status(user_id, "disabled", admin, store)


def _set_status(
    user_id: int, status: Literal["rejected", "disabled"], admin: User, store: AuthStore
) -> dict[str, Any]:
    """Shared body of reject and disable.

    Args:
        user_id: The user to change.
        status: 'rejected' or 'disabled'.
        admin: The logged-in admin (who may not lock themselves out).
        store: The user database.

    Returns:
        {'ok': True}.

    Raises:
        HTTPException: 400 for the admin's own account; 404 if there is no such user.
    """
    if user_id == admin.id:
        raise HTTPException(400, "You can't remove your own access")
    if not store.set_status(user_id, status):
        raise HTTPException(404, "No such user")
    log.info("auth.status_changed", user_id=user_id, status=status)
    return {"ok": True}
