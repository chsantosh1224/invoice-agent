"""Rebuild readable, layout-preserving text from positioned spans.

Spans are grouped into rows by vertical position, then placed on a character grid by
horizontal position, like `pdftotext -layout`. Table columns stay aligned and each
table row stays on one line, which keeps qty / price / amount together for the LLM.
"""

import math
from operator import attrgetter
from statistics import median

from invoice_agent.ingest.document import Span


def estimate_skew(quads: list[list[tuple[float, float]]]) -> float:
    """Estimate how much a scanned page is rotated, from OCR text-line boxes.

    Args:
        quads: OCR boxes as 4 corner points [(x, y) top-left, top-right, bottom-right,
            bottom-left] in pixels.

    Returns:
        The median text-line angle in radians (positive = text slopes downward to the
        right). 0.0 if there are no wide boxes to measure.
    """
    angles = []
    for q in quads:
        (x0, y0), (x1, y1) = q[0], q[1]
        width = math.hypot(x1 - x0, y1 - y0)
        height = math.hypot(q[3][0] - x0, q[3][1] - y0)
        if width > 3 * height > 0:  # only long lines give a reliable angle
            angles.append(math.atan2(y1 - y0, x1 - x0))
    return median(angles) if angles else 0.0


def deskew_quad(
    quad: list[tuple[float, float]], angle: float, cx: float, cy: float
) -> tuple[float, float, float, float]:
    """Rotate an OCR box back by the page skew and return its axis-aligned bounds.

    Args:
        quad: Four (x, y) corner points in pixels.
        angle: Page skew in radians (from `estimate_skew`).
        cx: X of the rotation center (page center), pixels.
        cy: Y of the rotation center, pixels.

    Returns:
        (x0, y0, x1, y1) of the straightened box, in pixels.
    """
    cos, sin = math.cos(-angle), math.sin(-angle)
    pts = [
        (cx + (x - cx) * cos - (y - cy) * sin, cy + (x - cx) * sin + (y - cy) * cos)
        for x, y in quad
    ]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def group_rows(spans: list[Span]) -> list[list[Span]]:
    """Group spans into visual rows (one row = one printed line across the page).

    Args:
        spans: Spans of one page, in any order.

    Returns:
        Rows from top to bottom; spans inside each row sorted left to right. A span joins
        a row when its vertical center is within half a typical text height of the row's.
    """
    if not spans:
        return []
    tolerance = 0.5 * median(s.height for s in spans)
    rows: list[list[Span]] = []
    centers: list[float] = []
    for s in sorted(spans, key=attrgetter("cy")):  # top to bottom
        if rows and abs(s.cy - centers[-1]) <= tolerance:
            rows[-1].append(s)
            centers[-1] = sum(x.cy for x in rows[-1]) / len(rows[-1])
        else:
            rows.append([s])
            centers.append(s.cy)
    return [sorted(r, key=attrgetter("x0")) for r in rows]  # left to right


def layout_text(spans: list[Span], *, columns: int = 140) -> str:
    """Render spans as plain text that keeps the page's columns and rows.

    Args:
        spans: Spans of one page.
        columns: Character width of the virtual grid that maps to the page width.

    Returns:
        Multi-line text: one line per visual row, spans padded to their horizontal
        position; a blank line marks a large vertical gap (a new block on the page).
    """
    rows = group_rows(spans)
    if not rows:
        return ""
    typical = median(s.height for s in spans)
    lines: list[str] = []
    prev_bottom: float | None = None
    for row in rows:
        top = min(s.y0 for s in row)
        if prev_bottom is not None and top - prev_bottom > 1.5 * typical:
            lines.append("")
        line = ""
        for s in row:
            col = int(s.x0 * columns)
            line += " " * (col - len(line)) if col > len(line) else (" " if line else "")
            line += s.text
        lines.append(line.rstrip())
        prev_bottom = max(s.y1 for s in row)
    return "\n".join(lines)
