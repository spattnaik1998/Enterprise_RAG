"""
Retrieval Quality Consensus System (Sprint 4, Use Case 2)
---------------------------------------------------------
Implements multi-agent consensus on retrieved context quality:

1. RetrieverAgent: executes HybridRetriever, outputs candidate_chunks + metadata
2. EvidenceVerifierAgent: validates chunk metadata, source_type match, freshness
3. HallucinationAuditorAgent: LLM-based check for faithfulness (uses haiku)
4. ConsensusAgent: combines signals, decides accept | retrieve_more | reject

Usage:
    consensus = RetrievalQualityConsensusSystem(retriever=retriever, generator=generator)
    result = await consensus.consensus(query="...", retrieved_chunks=[...])
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from src.context.freshness import FreshnessScorer
from src.context.tiers import TierClassifier
from src.observability.collector import get_active_collector, TraceEvent


@dataclass
class RetrieverAgentResult:
    """Output from RetrieverAgent."""
    candidate_chunks: list
    retrieval_latency_ms: float
    recall_score: float  # rough estimate based on candidate count


@dataclass
class EvidenceVerifierResult:
    """Output from EvidenceVerifierAgent."""
    valid_chunks: list
    evidence_validity: float    # 0.0-1.0: proportion of chunks passing metadata check
    source_coverage: float      # 0.0-1.0: diversity of sources in valid set
    confidence: float
    risk_flags: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class HallucinationAuditorResult:
    """Output from HallucinationAuditorAgent."""
    faithfulness_score: float   # 0.0-1.0: LLM judge score
    hallucination_probability: float  # 1.0 - faithfulness_score
    auditor_reasoning: str
    latency_ms: float


@dataclass
class ConsensusResult:
    """Final output from ConsensusAgent."""
    decision: Literal["accept", "retrieve_more", "reject"]
    final_context_set: list
    confidence_score: float
    evidence_validity: float
    source_coverage: float
    faithfulness_score: float
    hallucination_probability: float
    retrieval_ms: float
    verification_ms: float
    audit_ms: float
    consensus_ms: float
    total_ms: float
    recall_at_10: float
    reasoning: str


class RetrieverAgent:
    """Executes HybridRetriever and returns candidates with metadata."""

    def __init__(self, retriever) -> None:
        self._retriever = retriever

    async def run(
        self,
        query: str,
        top_k: int = 20,
    ) -> RetrieverAgentResult:
        """Execute retrieval."""
        start = time.time()

        try:
            # HybridRetriever.retrieve returns list of Chunk objects
            candidates = await self._retriever.retrieve(
                query=query,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"[RetrieverAgent] Retrieval failed: {e}")
            candidates = []

        latency_ms = (time.time() - start) * 1000

        # Rough recall estimate: candidate count / expected typical count
        recall_score = min(1.0, len(candidates) / max(1, top_k))

        return RetrieverAgentResult(
            candidate_chunks=candidates,
            retrieval_latency_ms=latency_ms,
            recall_score=recall_score,
        )


class EvidenceVerifierAgent:
    """
    Validates chunk metadata: completeness, source_type consistency, freshness.

    Checks:
      - Chunk has required fields (doc_id, source_type, content, metadata)
      - Source type is recognized
      - Freshness score (older data = lower confidence)
      - Content length > 0
    """

    REQUIRED_FIELDS = ["doc_id", "source_type", "content"]

    def __init__(self, freshness_scorer: FreshnessScorer | None = None) -> None:
        self._freshness_scorer = freshness_scorer or FreshnessScorer()
        self._tier_classifier = TierClassifier()

    async def run(
        self,
        chunks: list,
    ) -> EvidenceVerifierResult:
        """Verify chunk evidence quality."""
        start = time.time()

        valid_chunks = []
        risk_flags = []
        source_types_seen = set()

        for chunk in chunks:
            # Check required fields
            missing = [f for f in self.REQUIRED_FIELDS if f not in chunk]
            if missing:
                risk_flags.append(f"Chunk missing fields: {missing}")
                continue

            # Check content length
            content = chunk.get("content", "")
            if not content or len(content) < 10:
                risk_flags.append(f"Chunk {chunk['doc_id']} has insufficient content")
                continue

            # Check source type validity
            source_type = chunk.get("source_type", "unknown")
            if source_type not in [
                "billing",
                "psa",
                "crm",
                "contracts",
                "communications",
                "arxiv",
                "wikipedia",
                "rss",
            ]:
                risk_flags.append(f"Unrecognized source_type: {source_type}")
                continue

            # Freshness check
            freshness_score = self._freshness_scorer.score(chunk)
            if freshness_score < 0.3:
                risk_flags.append(
                    f"Chunk {chunk['doc_id']} is stale (freshness={freshness_score:.2f})"
                )

            # Chunk is valid
            valid_chunks.append(chunk)
            source_types_seen.add(source_type)

        # Calculate metrics
        evidence_validity = (
            len(valid_chunks) / max(1, len(chunks)) if chunks else 0.0
        )
        source_coverage = (
            len(source_types_seen) / 8.0
        )  # 8 possible sources

        confidence = evidence_validity * 0.7 + source_coverage * 0.3

        latency_ms = (time.time() - start) * 1000

        return EvidenceVerifierResult(
            valid_chunks=valid_chunks,
            evidence_validity=min(evidence_validity, 1.0),
            source_coverage=min(source_coverage, 1.0),
            confidence=min(confidence, 1.0),
            risk_flags=risk_flags,
            latency_ms=latency_ms,
        )


class HallucinationAuditorAgent:
    """
    LLM-based faithfulness check: does the generated answer appear in retrieved chunks?

    Uses claude-haiku-4-5-20251001 for cheap, fast evaluation.
    """

    def __init__(self, generator=None) -> None:
        self._generator = generator

    async def run(
        self,
        query: str,
        answer: str,
        chunks: list,
    ) -> HallucinationAuditorResult:
        """Check answer faithfulness against chunks."""
        start = time.time()

        if not chunks:
            # No context = high hallucination risk
            return HallucinationAuditorResult(
                faithfulness_score=0.0,
                hallucination_probability=1.0,
                auditor_reasoning="No context chunks available for grounding",
                latency_ms=(time.time() - start) * 1000,
            )

        # Prepare context
        context_text = "\n\n".join(
            [f"[{c.get('source_type', 'unknown')}] {c.get('content', '')[:500]}"
             for c in chunks[:10]]
        )

        # Use LLM to check faithfulness
        try:
            from anthropic import Anthropic

            client = Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Given the query and retrieved context, evaluate how faithfully
the answer is grounded in the context.

Query: {query}

Context:
{context_text}

Answer: {answer}

Rate faithfulness on a 0.0-1.0 scale:
- 1.0: All major claims are directly supported by context
- 0.5: Some claims supported, some inferred or unsupported
- 0.0: Answer contradicts context or makes up facts

Return ONLY JSON: {{"faithfulness": <0.0-1.0>, "reasoning": "<brief>"}}""",
                    }
                ],
            )

            result_text = response.content[0].text
            data = json.loads(result_text)
            faithfulness_score = float(data.get("faithfulness", 0.5))

        except Exception as e:
            logger.warning(f"[HallucinationAuditor] LLM check failed: {e}")
            faithfulness_score = 0.5  # Conservative fallback

        latency_ms = (time.time() - start) * 1000

        return HallucinationAuditorResult(
            faithfulness_score=min(max(faithfulness_score, 0.0), 1.0),
            hallucination_probability=1.0 - min(max(faithfulness_score, 0.0), 1.0),
            auditor_reasoning=data.get("reasoning", "Faithfulness evaluated by haiku"),
            latency_ms=latency_ms,
        )


class ConsensusAgent:
    """
    Combines all three signals to make final accept/retrieve_more/reject decision.

    Decision rules:
      - If evidence_validity >= 0.9 AND faithfulness >= 0.85 AND source_coverage >= 0.5
        → ACCEPT
      - If evidence_validity < 0.5 AND hallucination_probability > 0.3
        → REJECT (poor evidence + hallucination risk)
      - If confidence < 0.7
        → RETRIEVE_MORE (ambiguous, need better candidates)
      - Otherwise → ACCEPT with lower confidence
    """

    def __init__(self) -> None:
        pass

    async def run(
        self,
        retriever_result: RetrieverAgentResult,
        verifier_result: EvidenceVerifierResult,
        auditor_result: HallucinationAuditorResult,
    ) -> ConsensusResult:
        """Make consensus decision."""
        start = time.time()

        # Calculate overall confidence
        confidence_score = (
            verifier_result.confidence * 0.5
            + auditor_result.faithfulness_score * 0.5
        )

        # Decision logic
        decision = "accept"
        reasoning = ""

        if (
            verifier_result.evidence_validity >= 0.9
            and auditor_result.faithfulness_score >= 0.85
            and verifier_result.source_coverage >= 0.5
        ):
            decision = "accept"
            reasoning = "Strong evidence + high faithfulness"

        elif (
            verifier_result.evidence_validity < 0.5
            and auditor_result.hallucination_probability > 0.3
        ):
            decision = "reject"
            reasoning = "Poor evidence quality + hallucination risk"

        elif confidence_score < 0.7:
            decision = "retrieve_more"
            reasoning = f"Confidence below threshold ({confidence_score:.2f})"

        else:
            decision = "accept"
            reasoning = f"Accepted with confidence {confidence_score:.2f}"

        latency_ms = (time.time() - start) * 1000

        return ConsensusResult(
            decision=decision,
            final_context_set=verifier_result.valid_chunks,
            confidence_score=min(confidence_score, 1.0),
            evidence_validity=verifier_result.evidence_validity,
            source_coverage=verifier_result.source_coverage,
            faithfulness_score=auditor_result.faithfulness_score,
            hallucination_probability=auditor_result.hallucination_probability,
            retrieval_ms=retriever_result.retrieval_latency_ms,
            verification_ms=verifier_result.latency_ms,
            audit_ms=auditor_result.latency_ms,
            consensus_ms=latency_ms,
            total_ms=(
                retriever_result.retrieval_latency_ms
                + verifier_result.latency_ms
                + auditor_result.latency_ms
                + latency_ms
            ),
            recall_at_10=retriever_result.recall_score,
            reasoning=reasoning,
        )


class RetrievalQualityConsensusSystem:
    """Main orchestrator for consensus-based retrieval quality assurance."""

    def __init__(self, retriever, generator=None) -> None:
        self._retriever_agent = RetrieverAgent(retriever)
        self._verifier_agent = EvidenceVerifierAgent()
        self._auditor_agent = HallucinationAuditorAgent(generator)
        self._consensus_agent = ConsensusAgent()

    async def consensus(
        self,
        query: str,
        answer: str | None = None,
        top_k: int = 20,
    ) -> ConsensusResult:
        """
        Run full consensus pipeline.

        Args:
            query: User query
            answer: Generated answer (optional, for hallucination check)
            top_k: Candidate count

        Returns:
            ConsensusResult with decision + context set
        """
        # Record trace
        collector = get_active_collector()
        if collector:
            collector.record(
                TraceEvent(
                    event_type="consensus_start",
                    data={"query_length": len(query)},
                )
            )

        # Stage 1: Retrieve
        retriever_result = await self._retriever_agent.run(query, top_k)

        # Stage 2: Verify
        verifier_result = await self._verifier_agent.run(retriever_result.candidate_chunks)

        # Stage 3: Audit (if answer provided)
        if answer:
            auditor_result = await self._auditor_agent.run(
                query,
                answer,
                verifier_result.valid_chunks,
            )
        else:
            # Placeholder audit result
            auditor_result = HallucinationAuditorResult(
                faithfulness_score=0.5,
                hallucination_probability=0.5,
                auditor_reasoning="No answer provided; skipping audit",
                latency_ms=0.0,
            )

        # Stage 4: Consensus
        result = await self._consensus_agent.run(
            retriever_result,
            verifier_result,
            auditor_result,
        )

        if collector:
            collector.record(
                TraceEvent(
                    event_type="consensus_complete",
                    data={
                        "decision": result.decision,
                        "confidence": result.confidence_score,
                        "total_ms": result.total_ms,
                    },
                )
            )

        logger.info(
            f"[Consensus] {result.decision.upper():15s} "
            f"| confidence={result.confidence_score:.2f} | "
            f"faithfulness={result.faithfulness_score:.2f} | "
            f"latency={result.total_ms:.1f}ms"
        )

        return result
