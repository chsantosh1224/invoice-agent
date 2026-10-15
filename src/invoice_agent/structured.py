"""Get validated structured output from any model, including free ones without schema support.

One path for every model:
    call model -> pull JSON out of the reply -> validate with Pydantic
        -> on failure, show the model its errors and ask again (max `max_repairs` times)

What changes per model is only how hard the provider helps (`OutputMode`):
- json_schema: the provider enforces our schema (strongest)
- json_mode:   the provider guarantees valid JSON; we describe the schema in the prompt
- prompt:      we describe the schema in the prompt and parse JSON out of free text
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from invoice_agent.llm_cache import ResponseCache, cache_key
from invoice_agent.models_catalog import OutputMode

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class StructuredOutputError(RuntimeError):
    """Raised when the model never produced valid output within the repair budget."""


@dataclass
class StructuredResult[T: BaseModel]:
    """The outcome of one `invoke_structured` call.

    Attributes:
        value: The validated Pydantic object.
        raw_text: The model's final reply text (what was validated).
        attempts: Model calls made (1 = first try succeeded; 0 = served from cache).
        cached: True if the reply came from the response cache.
        usage: Token counts summed over all attempts, e.g. {'input_tokens': ...}.
    """

    value: T
    raw_text: str
    attempts: int
    cached: bool
    usage: dict[str, int]


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Free models often wrap JSON in prose, ```json fences or <think> reasoning blocks.
    This strips those and decodes the outermost {...} object.

    Args:
        text: The raw reply text.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: If no JSON object can be found or decoded.
    """
    s = _THINK.sub("", text).strip()
    fenced = _FENCE.search(s)
    if fenced:
        s = fenced.group(1).strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


def schema_instruction(schema: type[BaseModel]) -> str:
    """Describe the required output format for models without schema enforcement.

    Args:
        schema: The Pydantic model the reply must match.

    Returns:
        A system-prompt paragraph containing the JSON schema and output rules.
    """
    return (
        "Reply with ONLY one JSON object that matches this JSON schema. No prose, no "
        "markdown fences. Include every property; use null when a value is absent.\n\n"
        + json.dumps(schema.model_json_schema(), indent=1)
    )


def response_format(schema: type[BaseModel], mode: OutputMode) -> dict[str, Any] | None:
    """Build the OpenAI-style `response_format` request option for a mode.

    Args:
        schema: The Pydantic model the reply must match.
        mode: How much the provider can enforce.

    Returns:
        The `response_format` dict for json_schema / json_mode, or None for prompt mode.
    """
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }
    if mode == "json_mode":
        return {"type": "json_object"}
    return None


def _validation_feedback(error: Exception) -> str:
    """Turn a parse/validation error into a correction request for the model.

    Args:
        error: A ValueError from `extract_json` or a pydantic ValidationError.

    Returns:
        A short message listing what was wrong (at most 10 errors).
    """
    if isinstance(error, ValidationError):
        lines = [
            f"- {'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
            for e in error.errors()[:10]
        ]
        detail = "\n".join(lines)
    else:
        detail = f"- {error}"
    return (
        "Your reply did not match the required JSON schema:\n"
        f"{detail}\n\nReply again with ONLY the corrected JSON object."
    )


def _add_usage(total: dict[str, int], message: AIMessage) -> None:
    """Add one reply's token counts into a running total (in place).

    Args:
        total: Running totals keyed by 'input_tokens', 'output_tokens', 'total_tokens'.
        message: The model reply; its `usage_metadata` may be None.

    Returns:
        None.
    """
    usage = message.usage_metadata
    if usage is None:
        return
    total["input_tokens"] = total.get("input_tokens", 0) + usage["input_tokens"]
    total["output_tokens"] = total.get("output_tokens", 0) + usage["output_tokens"]
    total["total_tokens"] = total.get("total_tokens", 0) + usage["total_tokens"]


def invoke_structured[T: BaseModel](
    llm: BaseChatModel,
    schema: type[T],
    messages: Sequence[BaseMessage],
    *,
    model: str,
    mode: OutputMode,
    max_repairs: int = 2,
    cache: ResponseCache | None = None,
) -> StructuredResult[T]:
    """Call a model and return its reply as a validated Pydantic object.

    Works the same for every model; `mode` only decides how much the provider enforces.
    Invalid replies trigger up to `max_repairs` follow-up turns in which the model sees
    its exact errors. Only replies that pass validation are cached.

    Args:
        llm: The chat model to call (from `llm.get_chat_model`, or a fake in tests).
        schema: The Pydantic model the reply must match, e.g. RawInvoice.
        messages: The conversation (system/human messages, may include images).
        model: OpenRouter slug, used in the cache key.
        mode: Structured-output mode from `models_catalog.output_mode_for`.
        max_repairs: Extra attempts allowed after the first invalid reply.
        cache: Response cache to read/write; None disables caching.

    Returns:
        A StructuredResult with the validated value, final raw text, attempt count,
        cache flag and summed token usage.

    Raises:
        StructuredOutputError: If no valid reply was produced within the repair budget.
    """
    schema_json = schema.model_json_schema()
    conversation: list[BaseMessage] = list(messages)
    if mode != "json_schema":
        conversation.insert(0, SystemMessage(schema_instruction(schema)))

    key = cache_key(model, mode, schema_json, conversation)
    if cache is not None and (hit := cache.get(key)) is not None:
        return StructuredResult(
            value=schema.model_validate(extract_json(hit)),
            raw_text=hit,
            attempts=0,
            cached=True,
            usage={},
        )

    fmt = response_format(schema, mode)
    runnable = llm.bind(response_format=fmt) if fmt else llm
    usage: dict[str, int] = {}
    last_error: Exception | None = None

    for attempt in range(1, max_repairs + 2):
        reply = runnable.invoke(conversation)
        assert isinstance(reply, AIMessage)
        _add_usage(usage, reply)
        text = str(reply.text)  # joins text blocks; ignores reasoning/tool blocks
        try:
            value = schema.model_validate(extract_json(text))
        except (ValueError, ValidationError) as e:  # ValidationError subclasses ValueError
            last_error = e
            conversation += [AIMessage(text), HumanMessage(_validation_feedback(e))]
            continue
        if cache is not None:
            cache.put(key, text)
        return StructuredResult(
            value=value, raw_text=text, attempts=attempt, cached=False, usage=usage
        )

    raise StructuredOutputError(
        f"{model} gave no valid {schema.__name__} after {max_repairs + 1} attempts: {last_error}"
    )
