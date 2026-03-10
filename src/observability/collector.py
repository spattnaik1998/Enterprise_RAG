"""
TraceCollector -- context manager that assembles an AgentTrace from pipeline events.

Usage:
    with TraceCollector(session_id="abc", query="...", model="gpt-4o-mini") as tc:
        tc.add_event(TraceEvent(event_type="query_start", payload={...}))
        tc.set_verdict("success")
    # -> automatically sampled and written to TraceStore

get_active_collector() returns the collector bound to the current async/thread
context, or None if no collector is active.
"""
from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Optional

from loguru import logger

from src.observability.schemas import AgentTrace, TraceEvent
from src.observability.sampler import FailureBiasedSampler
from src.observability.store import TraceStore

_ACTIVE_COLLECTOR: ContextVar[Optional["TraceCollector"]] = ContextVar(
    "_ACTIVE_COLLECTOR", default=None
)

_sampler = FailureBiasedSampler()
_store = TraceStore("data/traces")


def get_active_collector() -> Optional["TraceCollector"]:
    """Return the TraceCollector bound to this execution context, or None."""
    return _ACTIVE_COLLECTOR.get()


class TraceCollector:
    """
    Context-manager that records pipeline events and writes sampled traces.

    The collector is bound to the current execution context via a ContextVar,
    so it is safely accessible from async tasks and thread-pool workers without
    threading issues.
    """

    def __init__(
        self,
        session_id: str,
        query: str,
        model: str = "gpt-4o-mini",
        user_role: str = "anonymous",
    ) -> None:
        self._trace = AgentTrace.new(
            session_id=session_id,
            query=query,
            model=model,
            user_role=user_role,
        )
        self._token = None
        self._t_start = time.perf_counter()

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> "TraceCollector":
        self._token = _ACTIVE_COLLECTOR.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self._trace.verdict = "error"
            self.add_event(TraceEvent(
                event_type="verdict",
                payload={"outcome": "error"},
                error=str(exc_val),
            ))

        # Finalise total cost
        self._trace.total_cost_usd = sum(e.cost_usd for e in self._trace.events)

        # Apply failure-biased sampling
        reason = _sampler.should_capture(self._trace)
        if reason is not None:
            self._trace.capture_reason = reason
            try:
                _store.write(self._trace)
                logger.debug(
                    f"[Trace] Saved trace_id={self._trace.trace_id} "
                    f"verdict={self._trace.verdict} reason={reason}"
                )
            except Exception as exc:
                logger.warning(f"[Trace] Store write failed (non-fatal): {exc}")

        # Unbind from context
        if self._token is not None:
            _ACTIVE_COLLECTOR.reset(self._token)

        return False  # do not suppress exceptions

    # -- event API -------------------------------------------------------------

    def add_event(self, event: TraceEvent) -> None:
        """Append an event to this trace."""
        self._trace.events.append(event)

    def set_verdict(self, verdict: str) -> None:
        """Set the final pipeline outcome (call before exiting the context)."""
        self._trace.verdict = verdict

    @property
    def trace_id(self) -> str:
        return self._trace.trace_id

    @property
    def trace(self) -> AgentTrace:
        return self._trace
