"""Slow down password guessing: too many failed logins for one email -> wait."""

import threading
import time


class LoginThrottle:
    """Count recent failed logins per key (an email address) in memory.

    In memory is enough for one server: a restart forgets the counts, which only
    gives an attacker a fresh handful of guesses.
    """

    def __init__(self, max_failures: int = 5, window_s: float = 15 * 60) -> None:
        """Set the limit.

        Args:
            max_failures: Failed attempts allowed within the window.
            window_s: The window length in seconds.
        """
        self.max_failures = max_failures
        self.window_s = window_s
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> list[float]:
        """Drop expired failures for a key and return the ones left (lock held).

        Args:
            key: The email address.
            now: The current time (time.monotonic()).

        Returns:
            Times of failures still inside the window.
        """
        recent = [t for t in self._failures.get(key, []) if now - t < self.window_s]
        if recent:
            self._failures[key] = recent
        else:
            self._failures.pop(key, None)
        return recent

    def blocked(self, key: str) -> bool:
        """Whether this key has used up its attempts.

        Args:
            key: The email address.

        Returns:
            True if the next attempt should be refused.
        """
        with self._lock:
            return len(self._recent(key, time.monotonic())) >= self.max_failures

    def failed(self, key: str) -> None:
        """Record a failed attempt.

        Args:
            key: The email address.

        Returns:
            None.
        """
        with self._lock:
            now = time.monotonic()
            self._failures[key] = [*self._recent(key, now), now]

    def succeeded(self, key: str) -> None:
        """Forget the failures after a successful login.

        Args:
            key: The email address.

        Returns:
            None.
        """
        with self._lock:
            self._failures.pop(key, None)
