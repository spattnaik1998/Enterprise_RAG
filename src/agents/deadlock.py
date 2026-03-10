"""
DeadlockDetector -- tracks council retry counts per (session_id, query_hash).

Maintains an in-memory counter (suitable for single-process deployments).
When retries exceed max_retries, should_escalate() returns True to trigger
an escalated CouncilVerdict.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict


class DeadlockDetector:
    """
    Tracks retry counts per session + query pair.

    Args:
        max_retries: Maximum retries before triggering escalation.
    """

    def __init__(self, max_retries: int = 1) -> None:
        self._max_retries = max_retries
        self._counters: dict[str, int] = defaultdict(int)

    def _key(self, session_id: str, query: str) -> str:
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{session_id}:{query_hash}"

    def should_escalate(self, session_id: str, query: str) -> bool:
        """
        Increment the retry counter and return True if max_retries exceeded.
        The counter auto-resets to 0 after escalation.
        """
        key = self._key(session_id, query)
        self._counters[key] += 1
        if self._counters[key] > self._max_retries:
            self._counters[key] = 0  # reset after escalation
            return True
        return False

    def reset(self, session_id: str, query: str) -> None:
        """Manually reset the counter for a session+query pair."""
        key = self._key(session_id, query)
        self._counters.pop(key, None)
