"""OpenRouter chat-model factory with the determinism controls applied in one place."""

from typing import Any

from langchain_openai import ChatOpenAI

from invoice_agent.config import Settings, get_settings


class MissingAPIKeyError(RuntimeError):
    pass


def provider_preferences(settings: Settings) -> dict[str, Any]:
    prefs: dict[str, Any] = {"data_collection": "deny"}  # never let providers train on invoices
    if settings.provider_order:
        prefs["order"] = settings.provider_order
        prefs["allow_fallbacks"] = False  # same provider every run -> less variance
    return prefs


def get_chat_model(model: str, settings: Settings | None = None) -> ChatOpenAI:
    settings = settings or get_settings()
    if not settings.openrouter_api_key.get_secret_value():
        raise MissingAPIKeyError("OPENROUTER_API_KEY is not set. Copy .env.example to .env.")
    return ChatOpenAI(
        model=model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=0,
        seed=settings.seed,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        extra_body={"provider": provider_preferences(settings)},
    )
