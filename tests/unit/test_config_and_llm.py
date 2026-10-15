import pytest
from pydantic import SecretStr

from invoice_agent.config import Settings
from invoice_agent.llm import MissingAPIKeyError, get_chat_model, provider_preferences
from invoice_agent.models_catalog import parse_catalog
from invoice_agent.schemas import RawInvoice


def test_provider_order_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_ORDER", "anthropic, google-vertex")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.provider_order == ["anthropic", "google-vertex"]
    assert provider_preferences(s) == {
        "data_collection": "deny",
        "order": ["anthropic", "google-vertex"],
        "allow_fallbacks": False,
    }


def test_no_pinning_by_default() -> None:
    s = Settings(_env_file=None, provider_order=[])  # type: ignore[call-arg]
    assert provider_preferences(s) == {"data_collection": "deny"}


def test_missing_key_fails_fast() -> None:
    s = Settings(_env_file=None, openrouter_api_key=SecretStr(""))  # type: ignore[call-arg]
    with pytest.raises(MissingAPIKeyError):
        get_chat_model("any/model", s)


def test_chat_model_has_determinism_controls() -> None:
    s = Settings(_env_file=None, openrouter_api_key=SecretStr("test"), seed=42)  # type: ignore[call-arg]
    llm = get_chat_model("anthropic/claude-haiku-5.5", s)
    assert llm.temperature == 0
    assert llm.seed == 42
    assert llm.extra_body == {"provider": {"data_collection": "deny"}}


def test_raw_invoice_schema_is_strict_mode_friendly() -> None:
    """Strict JSON-schema mode needs every property required and no extra keys."""
    schema = RawInvoice.model_json_schema()

    def check(node: dict[str, object]) -> None:
        if node.get("type") == "object" and "properties" in node:
            props = node["properties"]
            assert isinstance(props, dict)
            assert set(node.get("required", [])) == set(props), node.get("title")
            assert node.get("additionalProperties") is False, node.get("title")

    check(schema)
    for definition in schema.get("$defs", {}).values():
        check(definition)


def test_parse_catalog_roles() -> None:
    payload = {
        "data": [
            {
                "id": "v/vision",
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": ["tools", "structured_outputs"],
                "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
            },
            {
                "id": "t/text",
                "architecture": {"input_modalities": ["text"]},
                "supported_parameters": ["tools"],
                "pricing": {},
            },
        ]
    }
    cat = parse_catalog(payload)
    assert cat["v/vision"].can_extract and cat["v/vision"].can_agent
    assert round(cat["v/vision"].usd_in_per_m, 2) == 0.10
    assert not cat["t/text"].can_extract and cat["t/text"].can_agent
