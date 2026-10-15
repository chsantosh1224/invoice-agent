"""Disk cache for LLM responses.

Identical requests (same model, mode, schema and messages) are answered from SQLite
instead of the API. This saves free-tier quota, makes re-runs free and reproducible, and
lets evals be re-scored without re-calling the model.
"""

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage


def cache_key(
    model: str, mode: str, schema: dict[str, Any], messages: Sequence[BaseMessage]
) -> str:
    """Build a stable fingerprint for one structured-output request.

    Args:
        model: OpenRouter slug.
        mode: Structured-output mode ('json_schema', 'json_mode' or 'prompt').
        schema: The JSON schema the reply must match (changes to it invalidate the cache).
        messages: The conversation sent to the model, including any images.

    Returns:
        A hex SHA-256 digest; equal inputs always give equal keys.
    """
    body = {
        "model": model,
        "mode": mode,
        "schema": schema,
        "messages": [{"type": m.type, "content": m.content} for m in messages],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


class ResponseCache:
    """A tiny key -> text store backed by one SQLite file."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the cache database.

        Args:
            path: SQLite file path, e.g. `.cache/llm.sqlite`. Parent dirs are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, text TEXT)")

    def get(self, key: str) -> str | None:
        """Look up a cached response.

        Args:
            key: A digest from `cache_key`.

        Returns:
            The cached response text, or None on a miss.
        """
        row = self._db.execute("SELECT text FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, text: str) -> None:
        """Store (or overwrite) a response.

        Args:
            key: A digest from `cache_key`.
            text: The raw model reply that passed validation.

        Returns:
            None.
        """
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, text))

    def close(self) -> None:
        """Close the database connection.

        Returns:
            None.
        """
        self._db.close()
