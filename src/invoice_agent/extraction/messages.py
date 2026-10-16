"""Build the chat messages for one extraction request (images and/or layout text)."""

import base64
import io
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from PIL import Image

from invoice_agent.extraction.prompts import SYSTEM_PROMPT, TEXT_INTRO, user_instructions

MAX_IMAGE_SIDE = 1600  # px; enough to read 8pt text on A4, keeps image tokens moderate


def image_to_data_url(img: Image.Image, *, max_side: int = MAX_IMAGE_SIDE) -> str:
    """Encode a page image as a JPEG data URL, downscaled to a sensible size.

    Args:
        img: A page image.
        max_side: Longest side in pixels after downscaling (never upscales).

    Returns:
        'data:image/jpeg;base64,...' suitable for an OpenAI-style image_url content block.
    """
    page = img.convert("RGB")
    scale = max_side / max(page.size)
    if scale < 1:
        page = page.resize(
            (round(page.width * scale), round(page.height * scale)), Image.Resampling.LANCZOS
        )
    buf = io.BytesIO()
    page.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_messages(
    *, images: list[Image.Image] | None = None, text: str | None = None
) -> list[BaseMessage]:
    """Assemble the system + user messages for extracting one invoice.

    Args:
        images: Page images to attach (for vision models), or None.
        text: Layout-preserving document text (from an ingestion engine), or None.

    Returns:
        [SystemMessage(rules), HumanMessage(instruction + images + text)].

    Raises:
        ValueError: If neither images nor text are given.
    """
    content: list[dict[str, Any]] = [
        {"type": "text", "text": user_instructions(has_image=bool(images), has_text=bool(text))}
    ]
    for img in images or []:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
    if text:
        hint = "; prefer the page images when they disagree" if images else ""
        content.append(
            {
                "type": "text",
                "text": f"{TEXT_INTRO.format(image_hint=hint)}\n\n<document>\n{text}\n</document>",
            }
        )
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(content=content)]  # type: ignore[arg-type]
