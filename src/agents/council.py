"""
CouncilOrchestrator -- 3-agent voting pattern for high-stakes MSP decisions.

Pattern:
  1. Run HybridRetriever + ContextManager to get shared ContextBundle (once)
  2. Dispatch FastCreative and ConservativeChecker in parallel (asyncio.gather)
  3. PolicyVerifier reviews both proposals + context
  4. Accept the winning proposal, or escalate if PolicyVerifier rejects both
  5. Retry once on "escalate"; if still unresolved, return escalated CouncilVerdict

Usage:
  council = CouncilOrchestrator(pipeline)
  verdict = await council.run("Should we escalate Alpine Financial?", ctx=abac_ctx)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

from loguru import logger

from src.agents.base import AgentProposal
from src.agents.deadlock import DeadlockDetector
from src.agents.roles import (
    ConservativeCheckerAgent,
    FastCreativeAgent,
    PolicyVerifierAgent,
)


@dataclass
class CouncilVerdict:
    """Final output from the CouncilOrchestrator."""
    accepted_answer: str
    winning_agent: str           # "FastCreative" | "ConservativeChecker" | "escalated"
    dissent_summary: str         # brief note on the losing proposal
    escalated: bool
    policy_reasons: list[str]
    total_cost_usd: float
    trace_id: str
    hallucination_detected: bool = False
    pii_concern: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "accepted_answer": self.accepted_answer,
            "winning_agent": self.winning_agent,
            "dissent_summary": self.dissent_summary,
            "escalated": self.escalated,
            "policy_reasons": self.policy_reasons,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "trace_id": self.trace_id,
            "hallucination_detected": self.hallucination_detected,
            "pii_concern": self.pii_concern,
            "latency_ms": round(self.latency_ms, 1),
        }


class CouncilOrchestrator:
    """
    Orchestrates 3 agents: FastCreative, ConservativeChecker, PolicyVerifier.

    Args:
        pipeline:   RAGPipeline instance for retrieval + context packing.
        max_retries: Max retries when PolicyVerifier returns "escalate" (default 1).
    """

    def __init__(self, pipeline: Any, max_retries: int = 1) -> None:
        self._pipeline = pipeline
        self._max_retries = max_retries
        self._creative    = FastCreativeAgent()
        self._conservative = ConservativeCheckerAgent()
        self._verifier    = PolicyVerifierAgent()
        self._deadlock    = DeadlockDetector(max_retries=max_retries)

    async def run(
        self,
        query: str,
        budget_tokens: int = 3000,
        session_id: Optional[str] = None,
    ) -> CouncilVerdict:
        """
        Run the full council pattern asynchronously.

        Returns:
            CouncilVerdict with the accepted answer or an escalation message.
        """
        session_id = session_id or str(uuid.uuid4())[:8]
        trace_id = str(uuid.uuid4())
        t_start = time.perf_counter()

        logger.info(f"[Council] Starting | session={session_id} query={query[:80]!r}")

        # -- Step 1: Retrieve + context pack (once, shared) --------------------
        loop = asyncio.get_event_loop()
        try:
            candidates = await loop.run_in_executor(
                None, self._pipeline.retriever.retrieve, query
            )
            # ContextManager expects plain Chunk objects, not (Chunk, float) tuples.
            candidate_chunks = [chunk for chunk, _ in candidates]
            candidate_scores = {chunk.chunk_id: score for chunk, score in candidates}

            context_bundle = await loop.run_in_executor(
                None,
                partial(
                    self._pipeline.context_manager.get_context,
                    query=query,
                    chunks=candidate_chunks,
                    fast_path=False,
                ),
            )
        except Exception as exc:
            logger.error(f"[Council] Retrieval failed: {exc}")
            return CouncilVerdict(
                accepted_answer=f"Council retrieval error: {exc}",
                winning_agent="escalated",
                dissent_summary="",
                escalated=True,
                policy_reasons=[f"retrieval_error: {exc}"],
                total_cost_usd=0.0,
                trace_id=trace_id,
                latency_ms=(time.perf_counter() - t_start) * 1000,
            )

        # Build (Chunk, float) tuples for agents — same format generators expect.
        if context_bundle.pieces:
            chunks_for_agents = [
                (
                    candidate_chunks[p.chunk_index],
                    candidate_scores.get(candidate_chunks[p.chunk_index].chunk_id, p.relevance_score),
                )
                for p in context_bundle.pieces
            ]
        else:
            chunks_for_agents = candidates[:10]

        # -- Step 2: Parallel proposals -----------------------------------------
        attempt = 0
        while attempt <= self._max_retries:
            try:
                creative_proposal, conservative_proposal = await asyncio.gather(
                    loop.run_in_executor(
                        None,
                        partial(self._creative.propose, query, chunks_for_agents),
                    ),
                    loop.run_in_executor(
                        None,
                        partial(self._conservative.propose, query, chunks_for_agents),
                    ),
                )
            except Exception as exc:
                logger.error(f"[Council] Agent proposal failed: {exc}")
                return CouncilVerdict(
                    accepted_answer=f"Agent error: {exc}",
                    winning_agent="escalated",
                    dissent_summary="",
                    escalated=True,
                    policy_reasons=[f"agent_error: {exc}"],
                    total_cost_usd=0.0,
                    trace_id=trace_id,
                    latency_ms=(time.perf_counter() - t_start) * 1000,
                )

            # -- Step 3: Policy verification ------------------------------------
            try:
                verifier_result = await loop.run_in_executor(
                    None,
                    partial(
                        self._verifier.verify,
                        query,
                        creative_proposal,
                        conservative_proposal,
                        chunks_for_agents,
                    ),
                )
            except Exception as exc:
                logger.warning(f"[Council] Verifier error: {exc} -- defaulting to conservative")
                verifier_result = {
                    "verdict": "accept_conservative",
                    "winning_agent": "ConservativeChecker",
                    "reasons": [f"verifier_error: {exc}"],
                    "hallucination_detected": False,
                    "pii_concern": False,
                }

            verdict_code = verifier_result.get("verdict", "accept_conservative")
            reasons = verifier_result.get("reasons", [])

            if verdict_code == "accept_creative":
                winner = creative_proposal
                loser = conservative_proposal
            elif verdict_code == "accept_conservative":
                winner = conservative_proposal
                loser = creative_proposal
            elif verdict_code in ("reject_both", "escalate"):
                # Check deadlock
                if self._deadlock.should_escalate(session_id, query):
                    logger.warning(f"[Council] Deadlock detected -- escalating session={session_id}")
                    latency_ms = (time.perf_counter() - t_start) * 1000
                    return CouncilVerdict(
                        accepted_answer=(
                            "This query requires human review. The council could not reach "
                            "consensus after the maximum number of retries. Please escalate to "
                            "a senior MSP engineer."
                        ),
                        winning_agent="escalated",
                        dissent_summary=f"Both proposals rejected: {'; '.join(reasons)}",
                        escalated=True,
                        policy_reasons=reasons,
                        total_cost_usd=0.0,
                        trace_id=trace_id,
                        hallucination_detected=verifier_result.get("hallucination_detected", False),
                        pii_concern=verifier_result.get("pii_concern", False),
                        latency_ms=latency_ms,
                    )
                attempt += 1
                logger.info(f"[Council] PolicyVerifier escalated -- retry {attempt}/{self._max_retries}")
                continue
            else:
                # Unknown verdict -- default to conservative
                winner = conservative_proposal
                loser = creative_proposal
                reasons.append("unknown_verdict_defaulted_to_conservative")

            # -- Step 4: Build verdict -----------------------------------------
            dissent = (
                f"{loser.agent_name} proposed an alternative answer "
                f"(confidence={loser.confidence:.2f})"
            ) if loser.answer else "No alternative proposal."

            latency_ms = (time.perf_counter() - t_start) * 1000
            logger.info(
                f"[Council] Complete | winner={winner.agent_name} "
                f"verdict={verdict_code} latency={latency_ms:.0f}ms"
            )
            return CouncilVerdict(
                accepted_answer=winner.answer,
                winning_agent=winner.agent_name,
                dissent_summary=dissent,
                escalated=False,
                policy_reasons=reasons,
                total_cost_usd=winner.cost_usd + loser.cost_usd,
                trace_id=trace_id,
                hallucination_detected=verifier_result.get("hallucination_detected", False),
                pii_concern=verifier_result.get("pii_concern", False),
                latency_ms=latency_ms,
            )

        # Should not reach here, but safeguard
        return CouncilVerdict(
            accepted_answer="Council escalated after maximum retries.",
            winning_agent="escalated",
            dissent_summary="",
            escalated=True,
            policy_reasons=["max_retries_exceeded"],
            total_cost_usd=0.0,
            trace_id=trace_id,
            latency_ms=(time.perf_counter() - t_start) * 1000,
        )
