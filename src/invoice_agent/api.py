"""HTTP API and demo page.

    GET  /         the demo page (upload an invoice, see the result)
    GET  /health   liveness check
    POST /extract  upload one invoice (PDF/PNG/JPG), get the ExtractionResult as JSON

Run with `invoice-agent serve` (or `uvicorn invoice_agent.api:app`). When DEMO_PASSWORD is
set (the hosted demo), every route except /health asks for a user name and password.
"""

import base64
import binascii
import secrets
import tempfile
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from invoice_agent import __version__
from invoice_agent.config import get_settings
from invoice_agent.extraction.router import process_document
from invoice_agent.llm_cache import ResponseCache

log = structlog.get_logger()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB: plenty for invoices, blocks abuse
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
PAGE = Path(__file__).parent / "web" / "index.html"

OPEN_PATHS = {"/health"}  # the host's uptime checks must reach it without a password


class DemoLogin(BaseHTTPMiddleware):
    """Password-protect the hosted demo with HTTP Basic auth.

    The browser shows its own login box and then resends the credentials with every
    request, so the page and the API need no login code of their own. Disabled when
    DEMO_PASSWORD is empty, so local development stays login-free.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Let the request through if no password is set, the path is open, or it logs in.

        Args:
            request: The incoming request.
            call_next: Starlette's handle to the rest of the app.

        Returns:
            The app's response, or 401 with a login prompt when the credentials are
            missing or wrong.
        """
        settings = get_settings()
        password = settings.demo_password.get_secret_value()
        if not password or request.url.path in OPEN_PATHS:
            return await call_next(request)
        if _credentials_match(
            request.headers.get("authorization", ""), settings.demo_user, password
        ):
            return await call_next(request)
        return PlainTextResponse(
            "Login required", status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Invoice Agent demo", charset="UTF-8"'},
        )  # fmt: skip


def _credentials_match(header: str, user: str, password: str) -> bool:
    """Check an 'Authorization: Basic ...' header against the demo user and password.

    Args:
        header: The raw Authorization header value ('' when absent).
        user: The expected user name.
        password: The expected password.

    Returns:
        True only for a well-formed Basic header with both values correct. Compared in
        constant time, so response timing doesn't reveal how much of a guess was right.
    """
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        given_user, _, given_password = base64.b64decode(encoded).decode().partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    user_ok = secrets.compare_digest(given_user.encode(), user.encode())
    password_ok = secrets.compare_digest(given_password.encode(), password.encode())
    return user_ok and password_ok


app = FastAPI(title="Invoice Agent", version=__version__)
app.add_middleware(DemoLogin)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the demo page.

    Returns:
        The single-page HTML app (upload form + result view), marked not cacheable so a
        browser always shows the latest version of the page.
    """
    return FileResponse(PAGE, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is up and which models it uses.

    Returns:
        {'status': 'ok', 'version': ..., 'extractor_model': ..., 'agent_model': ...}.
    """
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "extractor_model": settings.extractor_model,
        "agent_model": settings.agent_model,
    }


def _check_upload(filename: str, data: bytes) -> str:
    """Validate an uploaded file's type and size.

    Args:
        filename: The name the client sent (only its extension is used).
        data: The file contents.

    Returns:
        The lowercase file extension, e.g. '.pdf'.

    Raises:
        HTTPException: 415 for an unsupported type, 413 if too large, 400 if empty.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Unsupported file type {suffix or '(none)'}; use PDF, PNG or JPG")
    if not data:
        raise HTTPException(400, "The uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large; the limit is {MAX_UPLOAD_BYTES // 2**20} MB")
    return suffix


@app.post("/extract")
def extract(
    file: Annotated[UploadFile, File(description="Invoice as PDF, PNG or JPG")],
    use_agent: Annotated[bool, Form(description="Let the agent fix gate failures")] = True,
) -> dict[str, Any]:
    """Extract one uploaded invoice: fast path, then the agent if the gate fails.

    This is a plain (sync) endpoint: FastAPI runs it in a worker thread, so a slow LLM
    call doesn't block other requests.

    Args:
        file: The uploaded invoice.
        use_agent: True to escalate gate failures to the agent ('router' mode); False to
            stop after the fast path.

    Returns:
        {'result': ExtractionResult as JSON, 'seconds': wall-clock time, 'filename': ...}.

    Raises:
        HTTPException: For an invalid upload (see `_check_upload`).
    """
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    suffix = _check_upload(file.filename or "", data)
    settings = get_settings()
    cache = ResponseCache(settings.cache_dir / "llm.sqlite") if settings.llm_cache_enabled else None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"  # never trust the client's file name
        path.write_bytes(data)
        try:
            result, seconds = process_document(
                path, settings, mode="router" if use_agent else "fast", cache=cache
            )
        finally:
            if cache is not None:
                cache.close()
    log.info("api.extract", filename=file.filename, status=result.status, seconds=round(seconds, 1))
    return {
        "filename": file.filename,
        "seconds": round(seconds, 1),
        "result": result.model_dump(mode="json"),
    }
