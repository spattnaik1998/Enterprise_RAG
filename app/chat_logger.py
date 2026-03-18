"""
Chat History Logger
--------------------
Writes each RAG interaction to the Supabase chat_logs table and
reads back aggregated stats for the monitoring dashboard.

Public API (unchanged from the JSONL version -- server.py needs no edits):
  log_interaction(**kwargs) -> None
  load_logs(limit, offset)  -> list[dict]
  compute_stats()           -> dict
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

_sb = None  # module-level singleton


def _client():
    global _sb
    if _sb is None:
        from supabase import create_client
        if not _SUPABASE_URL or not _SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env"
            )
        _sb = create_client(_SUPABASE_URL, _SUPABASE_KEY)
    return _sb


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def log_interaction(
    *,
    session_id: Optional[str],
    query: str,
    answer: str,
    provider: str,
    model: str,
    blocked: bool,
    blocked_reason: str,
    citations: list[dict],
    retrieval_ms: float,
    rerank_ms: float,
    generation_ms: float,
    total_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    estimated_cost_usd: float,
    pii_redacted: list[str],
) -> None:
    """Insert one interaction record into chat_logs (non-blocking best-effort)."""
    citation_meta = [
        {
            "title":           c.get("title", ""),
            "source_type":     c.get("source_type", ""),
            "relevance_score": round(float(c.get("relevance_score", 0.0)), 4),
        }
        for c in citations
    ]
    source_types = list({c.get("source_type", "") for c in citations if c.get("source_type")})

    record = {
        "session_id":             session_id or "anonymous",
        "query":                  query,
        "answer_length":          len(answer),
        "answer_preview":         answer[:300] if answer else "",
        "provider":               provider,
        "model":                  model,
        "blocked":                blocked,
        "blocked_reason":         blocked_reason if blocked else "",
        "citation_count":         len(citations),
        "source_types":           source_types,
        "citations":              citation_meta,
        "latency_retrieval_ms":   round(retrieval_ms,   1),
        "latency_rerank_ms":      round(rerank_ms,      1),
        "latency_generation_ms":  round(generation_ms,  1),
        "latency_total_ms":       round(total_ms,       1),
        "tokens_prompt":          prompt_tokens,
        "tokens_completion":      completion_tokens,
        "tokens_total":           total_tokens,
        "estimated_cost_usd":     round(estimated_cost_usd, 6),
        "pii_redacted_count":     len(pii_redacted),
    }

    try:
        _client().table("chat_logs").insert(record).execute()
    except Exception as exc:
        logger.warning(f"[ChatLogger] Insert failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Read — paginated raw records
# ---------------------------------------------------------------------------

def load_logs(limit: int = 50, offset: int = 0) -> list[dict]:
    """
    Return log records newest-first.

    Args:
        limit:  Max records (1-500).
        offset: Skip this many records from the newest end.
    """
    try:
        result = (
            _client()
            .table("chat_logs")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return result.data or []
    except Exception as exc:
        logger.warning(f"[ChatLogger] load_logs failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Feedback — update user rating on an existing log entry
# ---------------------------------------------------------------------------

def update_feedback(log_id: str, rating: int, feedback: str = "") -> bool:
    """
    Update user rating and feedback on an existing chat log entry.

    Args:
        log_id:   The id (primary key) of the chat_logs row.
        rating:   1-5 star rating.
        feedback: Optional free-form text comment.

    Returns:
        True if the update succeeded, False otherwise.
    """
    if rating < 1 or rating > 5:
        logger.warning(f"[ChatLogger] Invalid rating {rating} (must be 1-5)")
        return False
    try:
        updates = {"user_rating": rating}
        if feedback:
            updates["user_feedback"] = feedback[:1000]  # cap at 1000 chars
        _client().table("chat_logs").update(updates).eq("id", log_id).execute()
        logger.info(f"[ChatLogger] Feedback updated: log_id={log_id} rating={rating}")
        return True
    except Exception as exc:
        logger.warning(f"[ChatLogger] update_feedback failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Stats — single RPC call returns the full dashboard payload
# ---------------------------------------------------------------------------

def compute_stats() -> dict:
    """
    Return aggregated statistics for the monitoring dashboard.

    Delegates all aggregation to the get_chat_stats() PostgreSQL function
    so the Python layer does zero computation.
    """
    try:
        result = _client().rpc("get_chat_stats", {}).execute()
        data = result.data

        # PostgREST wraps scalar-returning functions in a list
        if isinstance(data, list):
            data = data[0] if data else {}

        # Some Supabase versions nest the result under the function name
        if isinstance(data, dict) and "get_chat_stats" in data:
            data = data["get_chat_stats"]

        return data if isinstance(data, dict) else {"total_queries": 0}

    except Exception as exc:
        logger.warning(f"[ChatLogger] compute_stats failed: {exc}")
        return {"total_queries": 0}
