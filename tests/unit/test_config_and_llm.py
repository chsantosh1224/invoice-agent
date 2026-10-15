import pytest
from pydantic import SecretStr

from invoice_agent.config import Settings
from invoice_agent.llm import MissingAPIKeyError, get_chat_model, provider_preferences
from invoice_agent.models_catalog import parse_catalog
from invoice_agent.schemas import RawInvoice


def test_provider_order_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """PROVIDER_ORDER='a, b' in the env becomes a list and pins providers with no fallback."""
    monkeypatch.setenv("PROVIDER_ORDER", "anthropic, google-vertex")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.provider_order == ["anthropic", "google-vertex"]
    assert provider_preferences(s) == {
        "data_collection": "deny",
        "order": ["anthropic", "google-vertex"],
        "allow_fallbacks": False,
    }


def test_no_pinning_by_default() -> None:
    """Without a provider order, only the no-training policy is sent."""
    s = Settings(_env_file=None, provider_order=[])  # type: ignore[call-arg]
    assert provider_preferences(s) == {"data_collection": "deny"}


def test_prompt_logging_is_opt_in_for_free_models() -> None:
    """Free endpoints need data_collection=allow, which is off unless explicitly enabled."""
    assert Settings(_env_file=None).allow_prompt_logging is False  # type: ignore[call-arg]
    s = Settings(_env_file=None, allow_prompt_logging=True)  # type: ignore[call-arg]
    assert provider_preferences(s) == {"data_collection": "allow"}


def test_default_models_are_free_tier() -> None:
    """All three roles default to ':free' models; any can be swapped via env."""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert all(m.endswith(":free") for m in (s.extractor_model, s.agent_model, s.judge_model))


def test_missing_key_fails_fast() -> None:
    """Creating an LLM without an API key raises immediately, not on the first call."""
    s = Settings(_env_file=None, openrouter_api_key=SecretStr(""))  # type: ignore[call-arg]
    with pytest.raises(MissingAPIKeyError):
        get_chat_model("any/model", s)


def test_chat_model_has_determinism_controls() -> None:
    """Every chat model gets temperature 0, the configured seed and provider prefs."""
    s = Settings(_env_file=None, openrouter_api_key=SecretStr("test"), seed=42)  # type: ignore[call-arg]
    llm = get_chat_model("anthropic/claude-haiku-5.5", s)
    assert llm.temperature == 0
    assert llm.seed == 42
    assert llm.extra_body == {"provider": {"data_collection": "deny"}}


def test_raw_invoice_schema_is_strict_mode_friendly() -> None:
    """RawInvoice's JSON schema meets strict mode: all keys required, no extra keys."""
    schema = RawInvoice.model_json_schema()

    def check(node: dict[str, object]) -> None:
        """Assert one object node of the schema is strict-mode compliant.

        Args:
            node: A JSON-schema object definition (top level or from $defs).

        Returns:
            None. Fails the test if a property is optional or extra keys are allowed.
        """
        if node.get("type") == "object" and "properties" in node:
            props = node["properties"]
            assert isinstance(props, dict)
            assert set(node.get("required", [])) == set(props), node.get("title")
            assert node.get("additionalProperties") is False, node.get("title")

    check(schema)
    for definition in schema.get("$defs", {}).values():
        check(definition)


def test_parse_catalog_roles_and_output_modes() -> None:
    """Catalog flags map to roles and to the strongest structured-output mode."""
    payload = {
        "data": [
            {
                "id": "v/vision",
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": ["tools", "structured_outputs", "response_format"],
                "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
            },
            {
                "id": "g/gemma:free",
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": ["tools", "response_format"],
                "pricing": {"prompt": "0", "completion": "0"},
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
    assert cat["v/vision"].output_mode == "json_schema"
    assert round(cat["v/vision"].usd_in_per_m, 2) == 0.10
    assert cat["g/gemma:free"].output_mode == "json_mode" and cat["g/gemma:free"].is_free
    assert cat["g/gemma:free"].can_extract and cat["g/gemma:free"].can_agent
    assert cat["t/text"].output_mode == "prompt"
    assert not cat["t/text"].can_extract and cat["t/text"].can_agent
