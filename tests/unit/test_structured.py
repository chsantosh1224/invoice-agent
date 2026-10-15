import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from invoice_agent.llm_cache import ResponseCache, cache_key
from invoice_agent.structured import (
    StructuredOutputError,
    extract_json,
    invoke_structured,
    response_format,
)


class Pet(BaseModel):
    name: str
    age: int


def fake(*replies: str) -> GenericFakeChatModel:
    """A chat model that returns `replies` in order, one per call."""
    return GenericFakeChatModel(messages=iter(AIMessage(r) for r in replies))


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "Rex", "age": 3}',
        'Sure! Here it is:\n```json\n{"name": "Rex", "age": 3}\n```',
        '<think>The user wants JSON {not this}.</think>{"name": "Rex", "age": 3}',
        'Result: {"name": "Rex", "age": 3} -- done',
    ],
)
def test_extract_json_handles_free_model_noise(text: str) -> None:
    """Prose, ```json fences and <think> blocks around the JSON are stripped."""
    assert extract_json(text) == {"name": "Rex", "age": 3}


@pytest.mark.parametrize("text", ["no json here", "{broken", "[1, 2]"])
def test_extract_json_rejects_non_objects(text: str) -> None:
    """Missing, malformed or non-object JSON raises ValueError."""
    with pytest.raises(ValueError):
        extract_json(text)


def test_response_format_per_mode() -> None:
    """json_schema sends the strict schema, json_mode asks for JSON, prompt sends nothing."""
    strict = response_format(Pet, "json_schema")
    assert strict is not None and strict["json_schema"]["strict"] is True
    assert response_format(Pet, "json_mode") == {"type": "json_object"}
    assert response_format(Pet, "prompt") is None


def test_first_valid_reply_is_returned() -> None:
    """A valid first reply is accepted in one attempt."""
    result = invoke_structured(
        fake('{"name": "Rex", "age": 3}'), Pet, [HumanMessage("x")], model="m", mode="prompt"
    )
    assert result.value == Pet(name="Rex", age=3)
    assert result.attempts == 1 and not result.cached


def test_invalid_reply_is_repaired_with_feedback() -> None:
    """A schema violation triggers one repair turn that shows the model its errors."""
    llm = fake('{"name": "Rex", "age": "three"}', '{"name": "Rex", "age": 3}')
    result = invoke_structured(llm, Pet, [HumanMessage("x")], model="m", mode="json_mode")
    assert result.value.age == 3
    assert result.attempts == 2


def test_gives_up_after_repair_budget() -> None:
    """After 1 + max_repairs invalid replies, StructuredOutputError is raised."""
    llm = fake("nope", "still nope", "nope again")
    with pytest.raises(StructuredOutputError, match="after 3 attempts"):
        invoke_structured(llm, Pet, [HumanMessage("x")], model="m", mode="prompt", max_repairs=2)


def test_valid_reply_is_cached_and_reused(tmp_path: Path) -> None:
    """The second identical request is served from disk without calling the model."""
    cache = ResponseCache(tmp_path / "llm.sqlite")
    msgs = [HumanMessage("x")]
    first = invoke_structured(
        fake('{"name": "Rex", "age": 3}'), Pet, msgs, model="m", mode="prompt", cache=cache
    )
    second = invoke_structured(fake(), Pet, msgs, model="m", mode="prompt", cache=cache)
    cache.close()
    assert not first.cached and second.cached
    assert second.value == first.value and second.attempts == 0


def test_cache_key_changes_with_model_schema_or_messages() -> None:
    """Any change to model, schema or messages produces a different cache key."""
    schema = Pet.model_json_schema()
    base = cache_key("m", "prompt", schema, [SystemMessage("s"), HumanMessage("x")])
    assert base == cache_key("m", "prompt", schema, [SystemMessage("s"), HumanMessage("x")])
    assert base != cache_key("other", "prompt", schema, [SystemMessage("s"), HumanMessage("x")])
    assert base != cache_key("m", "prompt", {**schema, "x": 1}, [HumanMessage("x")])
    assert base != cache_key("m", "prompt", schema, [SystemMessage("s"), HumanMessage("y")])
    json.dumps(base)  # keys are plain strings
