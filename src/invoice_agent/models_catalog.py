"""Check candidate models against the role requirements using OpenRouter's public catalog."""

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

CATALOG_URL = "https://openrouter.ai/api/v1/models"

CANDIDATES = [
    "anthropic/claude-haiku-5.5",
    "anthropic/claude-sonnet-5.5",
    "google/gemini-3.5-flash-lite",
    "openai/gpt-5.6-luna",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning",
    "nvidia/nemotron-3-super-120b-a12b",
]


@dataclass(frozen=True)
class ModelInfo:
    id: str
    vision: bool
    tools: bool
    json_schema: bool
    usd_in_per_m: float
    usd_out_per_m: float

    @property
    def can_extract(self) -> bool:
        """Fast-path extractor: needs images and strict JSON schema."""
        return self.vision and self.json_schema

    @property
    def can_agent(self) -> bool:
        """Extraction agent: needs tool calling (vision is a bonus)."""
        return self.tools


def parse_catalog(payload: dict[str, Any]) -> dict[str, ModelInfo]:
    out: dict[str, ModelInfo] = {}
    for m in payload.get("data", []):
        params = set(m.get("supported_parameters") or [])
        modalities = set((m.get("architecture") or {}).get("input_modalities") or [])
        pricing = m.get("pricing") or {}
        out[m["id"]] = ModelInfo(
            id=m["id"],
            vision="image" in modalities,
            tools="tools" in params,
            json_schema="structured_outputs" in params,
            usd_in_per_m=float(pricing.get("prompt") or 0) * 1e6,
            usd_out_per_m=float(pricing.get("completion") or 0) * 1e6,
        )
    return out


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
)
def fetch_catalog(timeout: float = 30.0) -> dict[str, ModelInfo]:
    resp = httpx.get(CATALOG_URL, timeout=timeout)
    resp.raise_for_status()
    return parse_catalog(resp.json())
