"""Render invoice views to HTML (Jinja2) and HTML to PDF (headless Chromium via Playwright)."""

from pathlib import Path
from types import TracebackType
from typing import Any, Self

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from playwright.sync_api import Browser, Playwright, sync_playwright

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATES = ("classic", "gst_india", "modern", "minimal", "euro")

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html"]),
    undefined=StrictUndefined,  # a typo in a template fails loudly instead of printing ''
)


def render_html(template: str, view: dict[str, Any]) -> str:
    """Fill one invoice template with a view's strings.

    Args:
        template: Template name without extension, one of TEMPLATES.
        view: The printable strings from `build.build_invoice` (InvoiceSpec.view).

    Returns:
        A complete HTML document.

    Raises:
        jinja2.TemplateNotFound: If the template does not exist.
        jinja2.UndefinedError: If the template uses a key missing from `view`.
    """
    return _env.get_template(f"{template}.html").render(**view)


class PdfRenderer:
    """Converts HTML to PDF with one headless browser reused for every document.

    Use as a context manager so the browser is always closed:

        with PdfRenderer(channel="msedge") as r:
            r.to_pdf(html, Path("out.pdf"))
    """

    def __init__(self, channel: str | None = None) -> None:
        """Configure which browser to launch (nothing starts until `__enter__`).

        Args:
            channel: None for Playwright's bundled Chromium (needs
                `playwright install chromium`), or an installed browser such as
                'msedge' or 'chrome' to skip that ~150 MB download.
        """
        self._channel = channel
        self._pw: Playwright | None = None
        self._browser: Browser | None = None

    def __enter__(self) -> Self:
        """Start Playwright and launch the headless browser.

        Returns:
            This renderer, ready for `to_pdf`.
        """
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(channel=self._channel)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the browser and stop Playwright, even if rendering failed.

        Args:
            exc_type: Exception type raised inside the `with` block, if any.
            exc: The exception instance, if any.
            tb: Its traceback, if any.

        Returns:
            None (exceptions are not suppressed).
        """
        if self._browser is not None:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()

    def to_pdf(self, html: str, path: Path) -> None:
        """Print an HTML document to an A4 PDF, honouring the template's @page CSS.

        Args:
            html: A complete HTML document (from `render_html`).
            path: Where to write the PDF. Parent dirs must exist.

        Returns:
            None. Writes the file.

        Raises:
            RuntimeError: If called outside the `with` block.
        """
        if self._browser is None:
            raise RuntimeError("PdfRenderer must be used as a context manager")
        page = self._browser.new_page()
        try:
            page.set_content(html, wait_until="load")
            page.pdf(path=str(path), prefer_css_page_size=True, print_background=True)
        finally:
            page.close()
