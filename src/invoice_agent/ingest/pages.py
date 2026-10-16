"""File-type detection and page images, shared by all engines and by the LLM prompt."""

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from invoice_agent.ingest.document import DocKind

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MIN_CHARS_PER_PAGE = 20  # fewer characters than this in the text layer = treat as scanned


def detect_kind(path: Path) -> DocKind:
    """Decide whether a file has a usable text layer or needs OCR.

    Args:
        path: A PDF or image file.

    Returns:
        'scanned' for images and for PDFs whose pages carry (almost) no text layer;
        'digital' for PDFs with a real text layer.

    Raises:
        ValueError: If the file type is not supported.
    """
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "scanned"
    if suffix != ".pdf":
        raise ValueError(f"unsupported file type {suffix!r}")
    pdf = pdfium.PdfDocument(path)
    try:
        chars = sum(pdf[i].get_textpage().count_chars() for i in range(len(pdf)))
        return "digital" if chars >= MIN_CHARS_PER_PAGE * len(pdf) else "scanned"
    finally:
        pdf.close()


def page_images(path: Path, *, dpi: int = 200) -> list[Image.Image]:
    """Render every page of a document as an RGB image.

    Args:
        path: A PDF or image file.
        dpi: Render resolution for PDFs (ignored for images, which are used as-is).
            200 DPI is a good balance for OCR accuracy vs speed.

    Returns:
        One RGB image per page.
    """
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return [Image.open(path).convert("RGB")]
    pdf = pdfium.PdfDocument(path)
    try:
        return [pdf[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(len(pdf))]
    finally:
        pdf.close()
