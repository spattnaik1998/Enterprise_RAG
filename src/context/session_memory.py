"""
Session Memory
---------------
Loads recent conversation turns from the Supabase chat_logs table and
formats them as a history block for injection into the generator prompt.

Usage:
    memory = SessionMemory()
    history = memory.load_history(session_id, max_turns=5)
    formatted = memory.format_history(history)
    # Pass formatted string to generator as session context
"""
from __future__ import annotations

import os
from typing import Optional

from loguru import logger


_sb = None


def _client():
    """Lazy-init Supabase client (reuses the module-level singleton)."""
    global _sb
    if _sb is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            return None
        _sb = create_client(url, key)
    return _sb


class SessionMemory:
    """
    Loads and formats recent conversation history for a session.

    The history is injected into the generator prompt so follow-up
    queries have context of prior turns.
    """

    def __init__(self, max_history_tokens: int = 800) -> None:
        self.max_history_tokens = max_history_tokens

    def load_history(
        self, session_id: str, max_turns: int = 5
    ) -> list[dict]:
        """
        Load recent conversation turns from chat_logs.

        Returns list of {query, answer_preview} dicts, oldest first.
        """
        if not session_id:
            return []

        try:
            sb = _client()
            if sb is None:
                return []

            result = (
                sb.table("chat_logs")
                .select("query, answer_preview, created_at")
                .eq("session_id", session_id)
                .order("created_at", desc=True)
                .limit(max_turns)
                .execute()
            )
            rows = result.data or []
            # Reverse to get chronological order (oldest first)
            rows.reverse()
            return [
                {
                    "query": r.get("query", ""),
                    "answer_preview": r.get("answer_preview") or "",
                }
                for r in rows
                if r.get("query")
            ]
        except Exception as exc:
            logger.warning(f"[SessionMemory] Failed to load history: {exc}")
            return []

    def format_history(self, turns: list[dict]) -> str:
        """
        Format conversation history as a text block for the generator prompt.

        Returns empty string if no history. Caps at max_history_tokens
        (rough word-based estimate).
        """
        if not turns:
            return ""

        lines = []
        total_words = 0
        word_limit = self.max_history_tokens * 3 // 4  # rough tokens -> words

        for turn in turns:
            q = turn.get("query", "").strip()
            a = turn.get("answer_preview", "").strip()
            if not q:
                continue
            entry = f"User: {q}"
            if a:
                entry += f"\nAssistant: {a}"
            entry_words = len(entry.split())
            if total_words + entry_words > word_limit:
                break
            lines.append(entry)
            total_words += entry_words

        if not lines:
            return ""

        return "CONVERSATION HISTORY (prior turns in this session):\n" + "\n---\n".join(lines) + "\n"
