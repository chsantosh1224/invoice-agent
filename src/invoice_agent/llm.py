"""OpenRouter chat-model factory with the determinism controls applied in one place."""

from typing import Any

from langchain_openai import ChatOpenAI

from invoice_agent.config import Settings, get_settings


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM is requested but OPENROUTER_API_KEY is not configured."""


def provider_preferences(settings: Settings) -> dict[str, Any]:
    """Build OpenRouter's `provider` routing options from the settings.

    By default tells OpenRouter to use only providers that don't log or train on prompts.
    Most free endpoints do log, so `allow_prompt_logging` relaxes this for development on
    synthetic data. When a provider order is configured, pins requests to those providers
    with no fallback, so every run hits the same hardware and quantization (less variance).

    Args:
        settings: The app settings; `allow_prompt_logging` and `provider_order` are read.

    Returns:
        A dict for the request body's `provider` field, e.g.
        {'data_collection': 'deny', 'order': ['anthropic'], 'allow_fallbacks': False}.
    """
    prefs: dict[str, Any] = {
        "data_collection": "allow" if settings.allow_prompt_logging else "deny",
    }
    if settings.provider_order:
        prefs["order"] = settings.provider_order
        prefs["allow_fallbacks"] = False  # same provider every run -> less variance
    return prefs


def get_chat_model(model: str, settings: Settings | None = None) -> ChatOpenAI:
    """Create a chat model that talks to OpenRouter with all determinism controls set.

    Every LLM in the project should be created here, so temperature, seed, timeouts,
    retries and provider pinning are applied consistently.

    Args:
        model: OpenRouter model slug, e.g. 'anthropic/claude-haiku-5.5'.
        settings: Settings to use; defaults to the cached app settings. Pass your own
            in tests.

    Returns:
        A LangChain ChatOpenAI client pointed at OpenRouter (temperature 0, fixed seed).
        No network call is made until the model is invoked.

    Raises:
        MissingAPIKeyError: If OPENROUTER_API_KEY is empty.
    """
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
