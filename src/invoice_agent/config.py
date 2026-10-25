"""Runtime settings, loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the app.

    Each field can be set by an environment variable of the same name in upper case
    (e.g. `EXTRACTOR_MODEL`) or by a line in `.env`. Environment variables win over `.env`.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Per-role models (free tier by default). The extractor must accept images; the agent
    # must call tools well; the judge should be a different family from the extractor.
    # Chosen by the Day 3 bake-off (reports/model_selection.md, docs/decisions.md).
    extractor_model: str = "dots-studio/dots-3-note-preview:free"
    extractor_fallback_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    agent_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    # Used when the agent model's provider fails (seen: NVIDIA 404 on Nemotron Super).
    agent_fallback_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    judge_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"

    # Most free endpoints may log prompts. Allowed only for synthetic data in development;
    # must stay False for real invoices.
    allow_prompt_logging: bool = False

    # Determinism controls.
    provider_order: Annotated[list[str], NoDecode] = []  # CSV in env, not JSON
    seed: int = 7
    llm_timeout_s: float = 120.0  # free endpoints queue; give them time
    llm_max_retries: int = 5  # the OpenAI client backs off on 429 rate limits

    # Response cache: identical requests are answered from disk (saves free-tier quota and
    # makes re-runs reproducible).
    llm_cache_enabled: bool = True
    cache_dir: Path = Path(".cache")

    # Validation gate: minimum confidence score to auto-accept (calibrated on dev).
    gate_threshold: float = 0.8

    agent_max_steps: int = 8
    log_level: str = "INFO"

    # Hosted demo: when DEMO_PASSWORD is set, every page except /health asks for this
    # user name and password (HTTP Basic auth). Empty = no login (local development).
    demo_user: str = "demo"
    demo_password: SecretStr = SecretStr("")

    @field_validator("provider_order", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Let PROVIDER_ORDER be written as a comma-separated string in the environment.

        Args:
            v: The raw value: a string like 'anthropic, google-vertex' from the
                environment, or an already-built list when passed in code.

        Returns:
            A list of trimmed, non-empty provider names when given a string;
            any other value unchanged (pydantic then validates it as list[str]).
        """
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    """Load the settings once and reuse them for the rest of the process.

    Args:
        None.

    Returns:
        The cached Settings instance, built from the environment and `.env` on first call.
    """
    return Settings()
