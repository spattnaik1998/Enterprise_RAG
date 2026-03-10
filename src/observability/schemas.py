"""
Trace schemas for agent run observability.

AgentTrace is a "case file" for a single RAG pipeline invocation:
it captures all pipeline events (retrieval, rerank, generation, guardrails,
PII redaction) with timing and cost, so failures can be replayed and debugged.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceEvent:
    """
    A single timestamped event within an agent trace.

    event_type values:
        query_start       -- raw (PII-redacted) query received
        retrieval         -- HybridRetriever results
        rerank            -- LLMReranker results
        context_pack      -- ContextManager bundle
        generate          -- RAGGenerator answer
        guardrail_block   -- PromptGuard blocked the query
        pii_redact        -- PIIFilter found and redacted PII
        tool_call         -- MCP tool invocation
        verdict           -- final pipeline outcome
    """
    event_type: str
    payload: dict[str, Any]
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "cost_usd": round(self.cost_usd, 7),
            "error": self.error,
            "payload": self.payload,
        }


@dataclass
class AgentTrace:
    """
    Complete trace of a single RAG pipeline run.

    query_hash is SHA-256 of the raw query -- allows correlation with eval
    results without storing raw query text in two places.  The actual
    (PII-redacted) query text lives in the query_start event payload.
    """
    trace_id: str
    session_id: str
    query_hash: str
    model: str
    user_role: str
    events: list[TraceEvent]
    verdict: str                 # "success" | "guardrail_block" | "error" | "pii_redacted" | "escalated"
    total_cost_usd: float
    capture_reason: str | None   # set by FailureBiasedSampler; None = not sampled
    created_at: str

    @classmethod
    def new(cls, session_id: str, query: str, model: str, user_role: str) -> "AgentTrace":
        return cls(
            trace_id=str(uuid.uuid4()),
            session_id=session_id,
            query_hash=hashlib.sha256(query.encode()).hexdigest(),
            model=model,
            user_role=user_role,
            events=[],
            verdict="success",
            total_cost_usd=0.0,
            capture_reason=None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query_hash": self.query_hash,
            "model": self.model,
            "user_role": self.user_role,
            "verdict": self.verdict,
            "total_cost_usd": round(self.total_cost_usd, 7),
            "capture_reason": self.capture_reason,
            "created_at": self.created_at,
            "events": [e.to_dict() for e in self.events],
        }

    @property
    def event_types(self) -> list[str]:
        return [e.event_type for e in self.events]
