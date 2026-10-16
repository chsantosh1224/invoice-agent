"""Lightweight ingestion: PDF text layer (pdfplumber) or RapidOCR, plus our layout rebuild.

No PyTorch: ~100 MB of dependencies and well under 1 GB of RAM, so it fits a small server.
"""

from pathlib import Path
from typing import Any

import pdfplumber

from invoice_agent.ingest.document import Document, Page, Span
from invoice_agent.ingest.layout import deskew_quad, estimate_skew, layout_text
from invoice_agent.ingest.pages import detect_kind, page_images


class LightIngestor:
    """Ingests documents without any deep-learning layout model."""

    name = "light"

    def __init__(self, *, dpi: int = 200) -> None:
        """Configure the engine; the OCR model is loaded lazily on first scanned page.

        Args:
            dpi: Render resolution used before OCR on PDFs.
        """
        self._dpi = dpi
        self._ocr: Any = None

    def ingest(self, path: Path) -> Document:
        """Read a document into words-with-positions and layout text.

        Args:
            path: A PDF or image file.

        Returns:
            A Document with one Page per page. Digital PDFs use their text layer
            (exact characters, conf 1.0); scans and images go through RapidOCR with
            skew correction.
        """
        kind = detect_kind(path)
        pages = self._digital_pages(path) if kind == "digital" else self._scanned_pages(path)
        for page in pages:
            page.text = layout_text(page.spans)
        return Document(source=str(path), engine=self.name, kind=kind, pages=pages)

    def _digital_pages(self, path: Path) -> list[Page]:
        """Extract words and positions from a PDF's text layer.

        Args:
            path: A PDF with a text layer.

        Returns:
            Pages with one Span per word, coordinates normalized to the page.
        """
        pages = []
        with pdfplumber.open(path) as pdf:
            for number, p in enumerate(pdf.pages, start=1):
                w, h = float(p.width), float(p.height)
                spans = [
                    Span(
                        word["text"],
                        word["x0"] / w,
                        word["top"] / h,
                        word["x1"] / w,
                        word["bottom"] / h,
                    )
                    for word in p.extract_words(keep_blank_chars=False, use_text_flow=False)
                ]
                pages.append(Page(number=number, width=w, height=h, spans=spans))
        return pages

    def _scanned_pages(self, path: Path) -> list[Page]:
        """Recognize text on page images with RapidOCR, correcting page skew.

        Args:
            path: A scanned PDF or an image.

        Returns:
            Pages with one Span per OCR text segment, straightened and normalized.
        """
        if self._ocr is None:
            from rapidocr import RapidOCR

            self._ocr = RapidOCR()
        pages = []
        for number, img in enumerate(page_images(path, dpi=self._dpi), start=1):
            w, h = img.size
            result = self._ocr(img)
            boxes = [] if result.boxes is None else result.boxes  # numpy array or None
            quads = [[(float(x), float(y)) for x, y in box] for box in boxes]
            texts = () if result.txts is None else result.txts
            scores = () if result.scores is None else result.scores
            angle = estimate_skew(quads)
            spans = []
            for quad, text, score in zip(quads, texts, scores, strict=True):
                x0, y0, x1, y1 = deskew_quad(quad, angle, w / 2, h / 2)
                spans.append(Span(text, x0 / w, y0 / h, x1 / w, y1 / h, float(score)))
            pages.append(Page(number=number, width=w, height=h, spans=spans))
        return pages
