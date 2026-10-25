from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invoice_agent import api
from invoice_agent.auth.routes import COOKIE
from invoice_agent.auth.store import AuthStore
from invoice_agent.schemas import ExtractionResult, Invoice

SAMPLE = Path("data/samples/inv-004-minimal-IN.pdf")


def fake_process(
    path: Path, settings: Any, *, mode: str, cache: Any
) -> tuple[ExtractionResult, float]:
    """Stand-in for the real pipeline: records its inputs, returns a fixed result."""
    fake_process.calls.append((path.suffix, path.exists(), mode))  # type: ignore[attr-defined]
    result = ExtractionResult(
        status="accepted",
        route="fast_path",
        invoice=Invoice(invoice_number="INV-1"),
        score=0.99,
        model="fake/model",
        prompt_version="v",
        run_id="r1",
    )
    return result, 1.23


@pytest.fixture(autouse=True)
def _fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never call a real LLM from API tests."""
    fake_process.calls = []  # type: ignore[attr-defined]
    monkeypatch.setattr(api, "process_document", fake_process)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A client logged in as an approved user, against a throwaway user database."""
    store = AuthStore(tmp_path / "auth.sqlite")
    api.app.state.auth = store
    user = store.create_admin("tester@example.com", "Tester", "a-long-password")
    with TestClient(api.app) as c:
        c.cookies.set(COOKIE, store.create_session(user.id))
        yield c
    api.app.state.auth = None
    store.close()


def test_page_and_health(client: TestClient) -> None:
    """The demo page is served and /health reports the configured models."""
    page = client.get("/")
    assert page.status_code == 200 and "Invoice Agent" in page.text
    health = client.get("/health").json()
    assert health["status"] == "ok" and "extractor_model" in health


def test_extract_returns_the_result_and_passes_the_agent_switch(client: TestClient) -> None:
    """A valid upload reaches the pipeline as a real file; the agent checkbox sets the mode."""
    with SAMPLE.open("rb") as fh:
        resp = client.post(
            "/extract",
            files={"file": ("my invoice.pdf", fh, "application/pdf")},
            data={"use_agent": "false"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert (
        body["result"]["status"] == "accepted"
        and body["result"]["invoice"]["invoice_number"] == "INV-1"
    )
    assert body["filename"] == "my invoice.pdf" and body["seconds"] == 1.2
    assert fake_process.calls == [(".pdf", True, "fast")]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("name", "size", "status"),
    [("notes.txt", 5, 415), ("empty.pdf", 0, 400), ("big.pdf", 11 * 2**20, 413)],
    ids=["wrong-type", "empty", "too-large"],
)
def test_bad_uploads_are_rejected_before_any_llm_call(
    client: TestClient, name: str, size: int, status: int
) -> None:
    """Wrong type, empty file and oversized file are refused, and the pipeline never runs."""
    content = b"x" * size  # built here: as a parameter it would become an 11 MB test name
    resp = client.post("/extract", files={"file": (name, content, "application/octet-stream")})
    assert resp.status_code == status
    assert fake_process.calls == []  # type: ignore[attr-defined]
