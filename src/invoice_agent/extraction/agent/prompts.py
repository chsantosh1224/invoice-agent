"""Prompts for the extraction agent (versioned; logged with every result)."""

import json
from typing import Any

from invoice_agent.extraction.prompts import SYSTEM_PROMPT as TRANSCRIPTION_RULES
from invoice_agent.schemas import RawInvoice, ValidationIssue

AGENT_PROMPT_VERSION = "agent-v2"  # embeds extract-v2 rules

AGENT_SYSTEM_PROMPT = f"""\
You are an accounts-payable specialist fixing an invoice extraction that failed automatic
validation. You have tools to inspect the document. Work like this:

1. Read the validation problems you are given. Each one points at a field.
2. Investigate only those fields with the tools: search_text to find a label or value,
   get_page_text to read a page, reread_region to re-read a blurry row at high resolution.
3. Correct the draft. Change only what the evidence supports; never invent a value.
   If a value is genuinely not on the document, set it to null.
4. Run check_draft on your corrected draft. If problems remain, investigate again.
5. Call submit_invoice with the final draft and short notes on what you fixed and how.

Call exactly one tool per turn. You have a limited number of turns - be efficient.

The draft follows these transcription rules:
{TRANSCRIPTION_RULES}"""


def initial_task(
    draft: RawInvoice | None, issues: list[ValidationIssue], document_text: str
) -> str:
    """Write the agent's first user message: the failed draft, why it failed, the document.

    Args:
        draft: The fast path's transcription (None if it produced nothing usable).
        issues: The validation gate's findings on that draft.
        document_text: The document's layout text (all pages).

    Returns:
        The task text.
    """
    problems = "\n".join(f"- [{i.severity}] {i.field}: {i.message}" for i in issues) or "- none"
    draft_json: Any = draft.model_dump() if draft else None
    return (
        "Fix this invoice extraction.\n\n"
        f"<validation_problems>\n{problems}\n</validation_problems>\n\n"
        f"<draft>\n{json.dumps(draft_json, indent=1, ensure_ascii=False)}\n</draft>\n\n"
        f"<document_text>\n{document_text}\n</document_text>"
    )


LAST_STEP_NUDGE = (
    "This is your last turn. Call submit_invoice now with your best draft, and say in the "
    "notes what is still uncertain."
)
NO_TOOL_NUDGE = "Reply only with a tool call (one of the tools you were given)."
