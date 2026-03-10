"""
TraceStore -- JSONL-backed trace storage with an index file.

Directory layout:
    data/traces/
        index.jsonl                 -- one line per trace (lightweight header)
        {trace_id}.json             -- full trace case file

The index allows fast queries without loading every case file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from src.observability.schemas import AgentTrace


class TraceStore:
    """Simple file-based trace store backed by JSONL files."""

    def __init__(self, traces_dir: str = "data/traces") -> None:
        self._dir = Path(traces_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.jsonl"

    # -- write -----------------------------------------------------------------

    def write(self, trace: AgentTrace) -> None:
        """Persist a full trace as {trace_id}.json and update the index."""
        # Write full case file
        case_path = self._dir / f"{trace.trace_id}.json"
        try:
            with open(case_path, "w", encoding="utf-8") as f:
                json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"[TraceStore] Failed to write {case_path}: {exc}")
            raise

        # Append index entry
        index_entry = {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "created_at": trace.created_at,
            "verdict": trace.verdict,
            "capture_reason": trace.capture_reason,
            "total_cost_usd": round(trace.total_cost_usd, 7),
            "model": trace.model,
            "user_role": trace.user_role,
        }
        try:
            with open(self._index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(index_entry) + "\n")
        except Exception as exc:
            logger.warning(f"[TraceStore] Index append failed (non-fatal): {exc}")

    # -- read ------------------------------------------------------------------

    def query(
        self,
        verdict: Optional[str] = None,
        capture_reason: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Scan the index and return matching trace headers (not full case files).

        Args:
            verdict:        Filter by verdict ("error", "success", etc.)
            capture_reason: Filter by sampler reason.
            limit:          Maximum number of results.
        Returns:
            List of index entry dicts, newest first.
        """
        if not self._index_path.exists():
            return []
        rows: list[dict] = []
        try:
            with open(self._index_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if verdict is not None and entry.get("verdict") != verdict:
                        continue
                    if capture_reason is not None and entry.get("capture_reason") != capture_reason:
                        continue
                    rows.append(entry)
        except Exception as exc:
            logger.error(f"[TraceStore] Index read error: {exc}")
        # Newest first
        rows.reverse()
        return rows[:limit]

    def get(self, trace_id: str) -> Optional[dict]:
        """Load a full trace case file by trace_id."""
        case_path = self._dir / f"{trace_id}.json"
        if not case_path.exists():
            return None
        try:
            with open(case_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"[TraceStore] Failed to read {case_path}: {exc}")
            return None
