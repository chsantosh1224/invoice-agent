"""Runtime settings, loaded from environment variables and `.env`."""

from functools import lru_cache
from typing import Annotated

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Per-role models. The extractor must accept images; the agent must call tools well.
    extractor_model: str = "anthropic/claude-haiku-5.5"
    agent_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    judge_model: str = "openai/gpt-5.6-luna"

    # Determinism controls.
    provider_order: Annotated[list[str], NoDecode] = []  # CSV in env, not JSON
    seed: int = 7
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    agent_max_steps: int = 8
    log_level: str = "INFO"

    @field_validator("provider_order", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
