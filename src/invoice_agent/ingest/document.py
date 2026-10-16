"""Engine-independent document model shared by every ingestion engine.

Coordinates are normalized to 0..1 of the page (origin top-left), so spans from a PDF
text layer, OCR on a 200-DPI render, or Docling are directly comparable.
"""

from dataclasses import dataclass, field
from typing import Literal

DocKind = Literal["digital", "scanned"]


@dataclass(frozen=True)
class Span:
    """A piece of text and where it sits on the page.

    Attributes:
        text: The text (a word from a PDF text layer, or a line segment from OCR/Docling).
        x0: Left edge, 0..1 of page width.
        y0: Top edge, 0..1 of page height.
        x1: Right edge, 0..1 of page width.
        y1: Bottom edge, 0..1 of page height.
        conf: Recognition confidence 0..1 (1.0 for text read from a PDF text layer).
    """

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0

    @property
    def cy(self) -> float:
        """Vertical center of the span.

        Returns:
            (y0 + y1) / 2, in normalized page units.
        """
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        """Height of the span.

        Returns:
            y1 - y0, in normalized page units.
        """
        return self.y1 - self.y0


@dataclass
class Page:
    """One page of an ingested document.

    Attributes:
        number: 1-based page number.
        width: Page width in the engine's native units (points or pixels).
        height: Page height in the same units.
        spans: Text pieces with normalized positions, used for grounding/validation.
        text: Layout-aware text for the LLM (layout-preserving text or Markdown).
    """

    number: int
    width: float
    height: float
    spans: list[Span] = field(default_factory=list)
    text: str = ""


@dataclass
class Document:
    """The result of ingesting one file with one engine.

    Attributes:
        source: Path of the ingested file.
        engine: Which engine produced it, e.g. 'light' or 'docling'.
        kind: 'digital' (text layer read) or 'scanned' (text recognized by OCR).
        pages: The pages in order.
    """

    source: str
    engine: str
    kind: DocKind
    pages: list[Page] = field(default_factory=list)

    @property
    def text(self) -> str:
        """All pages' layout text, with page markers, ready to put in an LLM prompt.

        Returns:
            The concatenated text; pages are separated by '--- page N ---' lines.
        """
        return "\n\n".join(f"--- page {p.number} ---\n{p.text}" for p in self.pages)

    @property
    def ocr_confidence(self) -> float:
        """Average recognition confidence over all spans.

        Returns:
            Mean span confidence (1.0 for digital documents or documents with no spans).
        """
        spans = [s for p in self.pages for s in p.spans]
        return sum(s.conf for s in spans) / len(spans) if spans else 1.0
