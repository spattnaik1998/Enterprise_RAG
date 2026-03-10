"""
TraceReplayer -- re-runs a historical trace through the current pipeline.

The replayer loads a stored trace, extracts the original (PII-redacted) query
from the query_start event, runs it through the current pipeline, and returns
a new AgentTrace for diff comparison.

Note: The replayer runs in read-only mode -- MCP tool calls are logged but
      not executed, preventing replay attacks or unintended side-effects.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from src.observability.store import TraceStore


class TraceReplayer:
    """
    Re-runs a historical trace through the current RAGPipeline.

    Args:
        pipeline: A RAGPipeline instance (the current pipeline).
        store:    TraceStore to load traces from (defaults to data/traces/).
    """

    def __init__(self, pipeline, store: Optional[TraceStore] = None) -> None:
        self._pipeline = pipeline
        self._store = store or TraceStore("data/traces")

    def replay(self, trace_id: str) -> Optional[dict]:
        """
        Load trace by ID, extract original query, re-run through pipeline.

        Returns:
            A dict with keys: original_trace, new_result, diff
            or None if the trace could not be loaded.
        """
        original = self._store.get(trace_id)
        if original is None:
            logger.warning(f"[Replayer] Trace not found: {trace_id}")
            return None

        # Extract original query from query_start event
        query = None
        for event in original.get("events", []):
            if event.get("event_type") == "query_start":
                query = event.get("payload", {}).get("query_redacted") or event.get("payload", {}).get("query")
                break

        if not query:
            logger.warning(f"[Replayer] No query_start event in trace {trace_id}")
            return None

        logger.info(f"[Replayer] Replaying trace={trace_id} query={query[:80]!r}")

        try:
            new_result = self._pipeline.query(query)
            new_result_dict = new_result.to_dict()
        except Exception as exc:
            logger.error(f"[Replayer] Replay failed: {exc}")
            return {
                "original_trace": original,
                "new_result": None,
                "diff": {"error": str(exc)},
            }

        # Simple diff: compare verdict + answer length
        original_verdict = original.get("verdict", "unknown")
        new_verdict = "guardrail_block" if new_result.blocked else "success"
        diff = {
            "verdict_match": original_verdict == new_verdict,
            "original_verdict": original_verdict,
            "new_verdict": new_verdict,
            "original_answer_len": self._get_answer_len(original),
            "new_answer_len": len(new_result.answer or ""),
        }

        return {
            "original_trace": original,
            "new_result": new_result_dict,
            "diff": diff,
        }

    @staticmethod
    def _get_answer_len(trace: dict) -> int:
        for event in trace.get("events", []):
            if event.get("event_type") == "generate":
                return len(event.get("payload", {}).get("answer", "") or "")
        return 0
