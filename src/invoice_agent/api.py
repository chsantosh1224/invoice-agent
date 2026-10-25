"""HTTP API and demo page.

    GET  /         home page: what the product does, with a login pop-up (public)
    GET  /app      the demo app (upload an invoice, see the result)
    GET  /health   liveness check
    POST /extract  upload one invoice (PDF/PNG/JPG), get the ExtractionResult as JSON

Run with `invoice-agent serve` (or `uvicorn invoice_agent.api:app`). Every route except
/health and the login pages needs a logged-in user (see invoice_agent.auth.routes).
"""

import tempfile
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from invoice_agent import __version__
from invoice_agent.auth.routes import CurrentUser, SessionGate
from invoice_agent.auth.routes import router as auth_router
from invoice_agent.config import get_settings
from invoice_agent.extraction.router import process_document
from invoice_agent.llm_cache import ResponseCache

log = structlog.get_logger()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB: plenty for invoices, blocks abuse
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
WEB = Path(__file__).parent / "web"

app = FastAPI(title="Invoice Agent", version=__version__)
app.add_middleware(SessionGate)
app.include_router(auth_router)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the public home page.

    Returns:
        The page describing the product, with 'Try the demo' opening a login pop-up;
        marked not cacheable so a browser always shows the latest version.
    """
    return FileResponse(
        WEB / "home.html", media_type="text/html", headers={"Cache-Control": "no-store"}
    )


@app.get("/app", include_in_schema=False)
def demo_app() -> FileResponse:
    """Serve the demo app (logged-in users only, enforced by SessionGate).

    Returns:
        The single-page HTML app (upload form + result view), not cacheable.
    """
    return FileResponse(
        WEB / "index.html", media_type="text/html", headers={"Cache-Control": "no-store"}
    )


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
    user: CurrentUser,
    use_agent: Annotated[bool, Form(description="Let the agent fix gate failures")] = True,
) -> dict[str, Any]:
    """Extract one uploaded invoice: fast path, then the agent if the gate fails.

    This is a plain (sync) endpoint: FastAPI runs it in a worker thread, so a slow LLM
    call doesn't block other requests.

    Args:
        file: The uploaded invoice.
        user: The logged-in user (logged with the run).
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
    log.info(
        "api.extract",
        user_id=user.id,
        filename=file.filename,
        status=result.status,
        seconds=round(seconds, 1),
    )
    return {
        "filename": file.filename,
        "seconds": round(seconds, 1),
        "result": result.model_dump(mode="json"),
    }
