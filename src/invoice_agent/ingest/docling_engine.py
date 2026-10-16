"""Docling ingestion: deep-learning layout analysis + TableFormer table structure.

Uses RapidOCR for scanned pages - the same OCR engine as the lightweight path - so a
comparison measures Docling's layout/table models, not a different OCR engine.
Optional dependency: `uv sync --extra docling` (pulls PyTorch, ~700 MB of models).
"""

from pathlib import Path
from typing import Any

from invoice_agent.ingest.document import Document, Page, Span
from invoice_agent.ingest.pages import detect_kind


class DoclingIngestor:
    """Ingests documents with Docling and exposes them as our engine-independent Document."""

    def __init__(self, *, keep_furniture: bool = True) -> None:
        """Build the Docling converter (models load on the first conversion, not here).

        Args:
            keep_furniture: Include page headers/footers (letterhead, footer with VAT ID).
                True by default for invoices; False reproduces Docling's default output.

        Raises:
            ImportError: If the optional 'docling' extra is not installed.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import (
            DocumentConverter,
            ImageFormatOption,
            PdfFormatOption,
        )

        self._keep_furniture = keep_furniture
        self.name = "docling" if keep_furniture else "docling-default"
        opts = PdfPipelineOptions(
            do_ocr=True, do_table_structure=True, ocr_options=RapidOcrOptions()
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
            }
        )

    def ingest(self, path: Path) -> Document:
        """Convert a document with Docling.

        Args:
            path: A PDF or image file.

        Returns:
            A Document whose first page carries Docling's Markdown for the whole file
            (Docling orders content across pages itself) and whose pages carry Docling's
            text blocks and table cells as Spans for grounding.
        """
        from docling_core.types.doc.common.content_layer import ContentLayer

        # Docling files letterheads and footers under the "furniture" layer and skips them
        # by default. On invoices that is where the vendor name and VAT ID often are.
        layers = {ContentLayer.BODY, ContentLayer.FURNITURE} if self._keep_furniture else None
        doc = self._converter.convert(path).document
        sizes = {no: (p.size.width, p.size.height) for no, p in doc.pages.items()}
        pages = {no: Page(number=no, width=w, height=h) for no, (w, h) in sizes.items()}

        for item, _level in doc.iterate_items(included_content_layers=layers):
            text = getattr(item, "text", None)
            if text:
                for prov in getattr(item, "prov", []):
                    pages[prov.page_no].spans.append(_span(text, prov.bbox, sizes[prov.page_no]))
        for table in doc.tables:
            page_no = table.prov[0].page_no if table.prov else 1
            for cell in table.data.table_cells:
                if cell.text and cell.bbox is not None:
                    pages[page_no].spans.append(_span(cell.text, cell.bbox, sizes[page_no]))

        ordered = [pages[no] for no in sorted(pages)]
        if ordered:
            ordered[0].text = doc.export_to_markdown(included_content_layers=layers)
        return Document(source=str(path), engine=self.name, kind=detect_kind(path), pages=ordered)


def _span(text: str, bbox: Any, size: tuple[float, float]) -> Span:
    """Convert a Docling bounding box to a normalized, top-left-origin Span.

    Args:
        text: The item's text.
        bbox: A docling-core BoundingBox (bottom-left or top-left origin).
        size: (width, height) of its page in the same units.

    Returns:
        A Span with coordinates normalized to 0..1 of the page.
    """
    w, h = size
    box = bbox.to_top_left_origin(page_height=h)
    return Span(text, box.l / w, box.t / h, box.r / w, box.b / h)
