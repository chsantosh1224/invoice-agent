"""Documents the agent is working on, looked up by id.

Agent state only carries a short `doc_id` (small and serializable, so it can be
checkpointed for human-in-the-loop review); the heavy objects - parsed document and page
images - live here while the agent runs.
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from invoice_agent.ingest.document import Document


@dataclass
class Workspace:
    """One document under investigation.

    Attributes:
        path: The source file (re-rendered at high DPI when the agent zooms in).
        doc: The ingested document (layout text + word positions).
        images: Page images, as sent to vision models.
        zoom_cache: Re-read results already computed, keyed by (page, row band).
    """

    path: Path
    doc: Document
    images: list[Image.Image]
    zoom_cache: dict[tuple[int, int], str] = field(default_factory=dict)


_REGISTRY: dict[str, Workspace] = {}


def register(ws: Workspace) -> str:
    """Make a workspace available to the agent's tools.

    Args:
        ws: The workspace.

    Returns:
        A new doc_id to put in the agent's state.
    """
    doc_id = uuid.uuid4().hex[:12]
    _REGISTRY[doc_id] = ws
    return doc_id


def get(doc_id: str) -> Workspace:
    """Look up a registered workspace.

    Args:
        doc_id: Id returned by `register`.

    Returns:
        The workspace.

    Raises:
        KeyError: If the id is unknown or already released.
    """
    return _REGISTRY[doc_id]


def release(doc_id: str) -> None:
    """Forget a workspace once the agent is done (frees images and OCR results).

    Args:
        doc_id: Id returned by `register`.

    Returns:
        None.
    """
    _REGISTRY.pop(doc_id, None)
