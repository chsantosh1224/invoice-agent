"""Turn a clean digital PDF into a realistic "scanned" copy with no text layer.

The degradations mimic an office scanner or a phone photo: grayscale, slight skew,
blur, sensor noise, contrast loss and JPEG artifacts. All randomness comes from the
passed-in generator, so the same seed always produces the same scan.
"""

import io
import random
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageEnhance, ImageFilter


def degrade_page(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply scanner-like damage to one page image.

    Args:
        img: A clean page image (any mode).
        rng: Seeded random generator; controls the strength of every effect.

    Returns:
        A new grayscale ('L') image: rotated up to ±2.5°, blurred, noisy, lower
        contrast, and passed through a low-quality JPEG round trip.
    """
    page = img.convert("L")
    page = page.rotate(
        rng.uniform(-2.5, 2.5), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255
    )
    page = page.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    noise = Image.frombytes("L", page.size, rng.randbytes(page.width * page.height))
    page = Image.blend(page, noise, rng.uniform(0.04, 0.10))
    page = ImageEnhance.Contrast(page).enhance(rng.uniform(0.75, 1.0))
    page = ImageEnhance.Brightness(page).enhance(rng.uniform(1.0, 1.12))
    buf = io.BytesIO()
    page.save(buf, "JPEG", quality=rng.randint(30, 65))
    buf.seek(0)
    return Image.open(buf).convert("L")


def scan_pdf(src: Path, dst: Path, rng: random.Random, *, dpi: int = 150) -> int:
    """Rasterize a PDF, degrade every page, and save the result as PDF or PNG.

    Args:
        src: The clean digital PDF.
        dst: Output path. '.pdf' writes an image-only PDF (all pages); '.png' writes
            only the first page as an image (use for single-page invoices).
        rng: Seeded random generator.
        dpi: Scan resolution; 150 is typical for office scanners.

    Returns:
        The number of pages in the source document.

    Raises:
        ValueError: If `dst` is neither .pdf nor .png.
    """
    if dst.suffix not in (".pdf", ".png"):
        raise ValueError(f"unsupported scan format {dst.suffix!r}")
    pdf = pdfium.PdfDocument(src)
    try:
        pages = [degrade_page(pdf[i].render(scale=dpi / 72).to_pil(), rng) for i in range(len(pdf))]
    finally:
        pdf.close()
    if dst.suffix == ".png":
        pages[0].save(dst, "PNG")
    else:
        pages[0].save(dst, "PDF", resolution=dpi, save_all=True, append_images=pages[1:])
    return len(pages)


def page_count(path: Path) -> int:
    """Count pages in a PDF.

    Args:
        path: A PDF file.

    Returns:
        The number of pages.
    """
    pdf = pdfium.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()
