"""Check models against the role requirements using OpenRouter's public catalog."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

CATALOG_URL = "https://openrouter.ai/api/v1/models"
CATALOG_MAX_AGE_S = 24 * 3600

OutputMode = Literal["json_schema", "json_mode", "prompt"]

CANDIDATES = [
    # free
    "google/gemma-4-31b-it:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    # paid, for comparison
    "anthropic/claude-haiku-5.5",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "anthropic/claude-sonnet-5.5",
]


@dataclass(frozen=True)
class ModelInfo:
    """What one OpenRouter model supports and costs, as reported by the catalog.

    Attributes:
        id: OpenRouter slug, e.g. 'google/gemma-4-31b-it:free'.
        vision: Accepts image input.
        tools: Supports tool/function calling.
        json_schema: Supports strict JSON-schema structured output.
        json_mode: Supports `response_format: json_object` (valid JSON, schema not enforced).
        usd_in_per_m: Price in USD per million input tokens (0 for free models).
        usd_out_per_m: Price in USD per million output tokens (0 for free models).
    """

    id: str
    vision: bool
    tools: bool
    json_schema: bool
    json_mode: bool
    usd_in_per_m: float
    usd_out_per_m: float

    @property
    def is_free(self) -> bool:
        """Whether the model is on OpenRouter's free tier.

        Returns:
            True for ':free' variants (rate-limited, may log prompts).
        """
        return self.id.endswith(":free")

    @property
    def output_mode(self) -> OutputMode:
        """The strongest structured-output mechanism this model offers.

        Returns:
            'json_schema' if the provider enforces our schema, 'json_mode' if it only
            guarantees valid JSON, otherwise 'prompt' (schema described in the prompt and
            the reply parsed from text). Pydantic validates the result in every mode.
        """
        if self.json_schema:
            return "json_schema"
        if self.json_mode:
            return "json_mode"
        return "prompt"

    @property
    def can_extract(self) -> bool:
        """Whether this model can be the fast-path extractor.

        Returns:
            True if it accepts images. Structured output is handled by `output_mode`
            plus Pydantic validation and a repair loop, so it is not a hard requirement.
        """
        return self.vision

    @property
    def can_agent(self) -> bool:
        """Whether this model can drive the extraction agent.

        Returns:
            True if it supports tool calling (vision is a bonus, not required).
        """
        return self.tools


def parse_catalog(payload: dict[str, Any]) -> dict[str, ModelInfo]:
    """Turn OpenRouter's raw `/models` JSON into ModelInfo records.

    Kept separate from the HTTP call so it can be unit-tested with a fake payload.

    Args:
        payload: The decoded JSON body of GET /api/v1/models, shaped like
            {'data': [{'id': ..., 'architecture': {...}, 'supported_parameters': [...],
            'pricing': {...}}, ...]}. Missing keys are treated as "not supported" / free.

    Returns:
        A dict mapping each model slug to its ModelInfo.
    """
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
            json_mode="response_format" in params,
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
def fetch_catalog_payload(timeout: float = 30.0) -> dict[str, Any]:
    """Download OpenRouter's public model catalog (no API key needed).

    Network failures (connection reset, timeout) are retried up to 4 times with
    exponential backoff.

    Args:
        timeout: Seconds to wait for each HTTP attempt.

    Returns:
        The decoded JSON body of GET /api/v1/models.

    Raises:
        httpx.TransportError: If all retries fail to connect.
        httpx.HTTPStatusError: If OpenRouter returns a 4xx/5xx response.
    """
    resp = httpx.get(CATALOG_URL, timeout=timeout)
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    return payload


def load_catalog(cache_dir: Path, *, refresh: bool = False) -> dict[str, ModelInfo]:
    """Return the model catalog, from a disk cache when it is less than 24 hours old.

    Args:
        cache_dir: Directory for `openrouter_models.json` (created if missing).
        refresh: True to ignore the cache and download a fresh copy.

    Returns:
        A dict mapping every model slug on OpenRouter to its ModelInfo.

    Raises:
        httpx.HTTPError: If a download is needed and OpenRouter cannot be reached.
    """
    path = cache_dir / "openrouter_models.json"
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < CATALOG_MAX_AGE_S:
        return parse_catalog(json.loads(path.read_text(encoding="utf-8")))
    payload = fetch_catalog_payload()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return parse_catalog(payload)


def output_mode_for(model: str, cache_dir: Path) -> OutputMode:
    """Look up how to get structured output from a model.

    Args:
        model: OpenRouter slug.
        cache_dir: Where the catalog cache lives.

    Returns:
        The model's `output_mode`. Falls back to 'prompt' (works with every model) if the
        model is unknown or the catalog cannot be loaded.
    """
    try:
        info = load_catalog(cache_dir).get(model)
    except httpx.HTTPError:
        return "prompt"
    return info.output_mode if info else "prompt"
