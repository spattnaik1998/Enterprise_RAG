"""
DSPy Module Wrappers for RAG Pipeline Components
--------------------------------------------------
Wraps existing pipeline components (HybridRetriever, LLMReranker, RAGGenerator)
in DSPy-compatible modules for automated optimization.

Key design:
  - Retriever is NOT optimizable (FAISS+BM25 are index-based, not prompt-driven)
  - Generation step IS optimizable (uses dspy.ChainOfThought)
  - Reranking can be optimizable (replaces LLMReranker with dspy.Predict)
  - Council reimplemented synchronously with DSPy primitives (original stays async)
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import dspy
from loguru import logger

from dspy_module.signatures import (
    RAGSignature,
    CreativeProposalSignature,
    ConservativeProposalSignature,
    PolicyVerdictSignature,
    RerankerSignature,
)
from src.chunking.schemas import Chunk


# ─────────────────────────────────────────────────────────────────────────────
# DSPyRetrieverAdapter — Non-optimizable wrapper for HybridRetriever
# ─────────────────────────────────────────────────────────────────────────────


class DSPyRetrieverAdapter(dspy.Retrieve):
    """
    Wraps HybridRetriever as a DSPy Retrieve component.

    Note: This is NOT optimizable. Retrieval uses FAISS+BM25 indices,
    which are not prompt-driven. It exists only to integrate into the
    DSPy module graph.

    Returns:
        dspy.Prediction with:
          - passages: list of formatted passage strings [1]..[N]
          - chunks: list of Chunk objects for downstream processing
          - scores: list of float relevance scores
    """

    def __init__(self, retriever: Any, top_k: int = 10) -> None:
        """
        Args:
            retriever: HybridRetriever instance
            top_k: Number of candidates to return
        """
        super().__init__()
        self.retriever = retriever
        self.top_k = top_k

    def forward(self, query: str) -> dspy.Prediction:
        """
        Retrieve candidates via hybrid search.

        Args:
            query: User query string

        Returns:
            dspy.Prediction with passages (formatted), chunks (raw), scores
        """
        # Retrieve via FAISS+BM25
        results: list[tuple[Chunk, float]] = self.retriever.retrieve(query)

        # Format passages as numbered list for LLM consumption
        passages = []
        chunks = []
        scores = []
        for i, (chunk, score) in enumerate(results, 1):
            # Format: [1] source_type: source | text
            passage = f"[{i}] {chunk.source_type}: {chunk.source} | {chunk.text[:200]}..."
            passages.append(passage)
            chunks.append(chunk)
            scores.append(float(score))

        # Combine into single passage string for context
        passages_str = "\n".join(passages)

        return dspy.Prediction(
            passages=passages,
            passages_str=passages_str,
            chunks=chunks,
            scores=scores,
        )


# ─────────────────────────────────────────────────────────────────────────────
# DSPyRAGModule — Optimizable generation-only module
# ─────────────────────────────────────────────────────────────────────────────


class DSPyRAGModule(dspy.Module):
    """
    Wraps RAGPipeline components: retriever -> reranker -> context manager -> generator.

    Only the generation step is optimizable (dspy.ChainOfThought on RAGSignature).
    Retrieval, reranking, and context management use existing pipeline code.

    Usage:
        module = DSPyRAGModule(pipeline)
        prediction = module.forward(query="Which clients are overdue?")
        print(prediction.answer)
    """

    def __init__(self, pipeline: Any, top_k: int = 10, rerank_top_k: int = 5) -> None:
        """
        Args:
            pipeline: RAGPipeline instance (with retriever, reranker, context_manager, generator)
            top_k: Candidates before reranking
            rerank_top_k: Chunks kept after reranking
        """
        super().__init__()
        self._pipeline = pipeline
        # Note: Non-trainable components (retriever, reranker, context_manager)
        # are stored but not exposed as DSPy parameters to avoid serialization issues
        self._retriever_adapter = DSPyRetrieverAdapter(pipeline.retriever, top_k=top_k)
        self._reranker = pipeline.reranker
        self._context_manager = pipeline.context_manager

        # Optimizable node
        self.generate_answer = dspy.ChainOfThought(RAGSignature)

    def forward(self, query: str) -> dspy.Prediction:
        """
        Full RAG pipeline: retrieve -> rerank -> pack context -> generate.

        Args:
            query: User query string

        Returns:
            dspy.Prediction with answer, context, citations
        """
        # Step 1: Retrieve candidates (returns list of (Chunk, float) tuples)
        # Note: retriever.retrieve() returns list[tuple[Chunk, float]], but
        # DSPyRetrieverAdapter.forward() extracts and formats them
        # We need to reconstruct tuples for the reranker
        retrieval_pred = self._retriever_adapter.forward(query)
        chunks = retrieval_pred.chunks
        scores = retrieval_pred.scores
        candidates_with_scores = list(zip(chunks, scores))
        passages_str = retrieval_pred.passages_str

        # Step 2: Rerank candidates (expects list of (Chunk, float) tuples)
        if candidates_with_scores:
            reranked_with_scores = self._reranker.rerank(query, candidates_with_scores)[:5]
            reranked_chunks = [chunk for chunk, _ in reranked_with_scores]
        else:
            reranked_chunks = []

        # Step 3: Pack context with budget
        context_bundle = self._context_manager.get_context(
            query=query,
            chunks=reranked_chunks,
            budget_tokens=3000,
            fast_path=False,
        )
        # Format context pieces for the LLM
        if context_bundle and context_bundle.pieces:
            context_str = "\n\n".join(
                f"[{i}] {piece.text}"
                for i, piece in enumerate(context_bundle.pieces, 1)
            )
        else:
            context_str = passages_str

        # Step 4: Generate answer (optimizable)
        output = self.generate_answer(query=query, context=context_str)

        # Parse citations from generated answer
        citations = self._extract_citations(output.answer, reranked_chunks)

        return dspy.Prediction(
            answer=output.answer,
            context=context_str,
            citations=citations,
            chunks=reranked_chunks,
        )

    @staticmethod
    def _extract_citations(answer: str, chunks: list[Chunk]) -> list[dict]:
        """
        Extract source numbers [1]..[N] from answer text.

        Args:
            answer: Generated answer string
            chunks: Reranked Chunk objects

        Returns:
            list of {source_number, source, source_type} dicts
        """
        import re

        citations = []
        cited_numbers = set()

        # Find all [N] patterns
        for match in re.finditer(r"\[(\d+)\]", answer):
            num = int(match.group(1))
            if 1 <= num <= len(chunks) and num not in cited_numbers:
                chunk = chunks[num - 1]
                citations.append(
                    {
                        "source_number": num,
                        "source": chunk.source,
                        "source_type": chunk.source_type,
                    }
                )
                cited_numbers.add(num)

        return citations


# ─────────────────────────────────────────────────────────────────────────────
# DSPyRerankerModule — Replaces LLMReranker with dspy.Predict
# ─────────────────────────────────────────────────────────────────────────────


class DSPyRerankerModule(dspy.Module):
    """
    Replaces LLMReranker with a DSPy-optimizable reranking module.

    Uses dspy.Predict(RerankerSignature) to score candidates.
    Falls back to original order on JSON parse failure.

    Usage:
        module = DSPyRerankerModule()
        reranked = module.forward(query, candidates)
    """

    def __init__(self) -> None:
        super().__init__()
        self.score_candidates = dspy.Predict(RerankerSignature)

    def forward(
        self, query: str, candidates: list[Chunk]
    ) -> list[Chunk]:
        """
        Score and rerank candidates.

        Args:
            query: User query
            candidates: List of Chunk objects

        Returns:
            Reranked list of Chunk objects
        """
        if not candidates:
            return []

        # Format candidates as newline-separated with IDs
        candidates_str = "\n".join(
            f"[chunk_{i}] {chunk.text[:150]}..."
            for i, chunk in enumerate(candidates)
        )

        # Score via DSPy
        try:
            output = self.score_candidates(query=query, candidates=candidates_str)
            scores_json = output.scores_json

            # Parse JSON scores
            try:
                scores = json.loads(scores_json)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse scores JSON: {scores_json}")
                return candidates

            # Rerank by score
            scored = []
            for i, chunk in enumerate(candidates):
                chunk_id = f"chunk_{i}"
                score = scores.get(chunk_id, 0)
                scored.append((chunk, score))

            # Sort descending by score
            scored.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in scored]

        except Exception as e:
            logger.error(f"Reranker error: {e}")
            return candidates


# ─────────────────────────────────────────────────────────────────────────────
# DSPyCouncilModule — Synchronous 3-agent voting (reimplemented from CouncilOrchestrator)
# ─────────────────────────────────────────────────────────────────────────────


class DSPyCouncilModule(dspy.Module):
    """
    3-agent voting pattern (FastCreative + ConservativeChecker + PolicyVerifier).

    NOTE: This is a SYNCHRONOUS reimplementation using DSPy primitives, separate
    from the original async CouncilOrchestrator which serves the web API.

    Usage:
        module = DSPyCouncilModule(pipeline)
        prediction = module.forward(query=user_query)
        print(prediction.verdict_json)
    """

    def __init__(self, pipeline: Any) -> None:
        """
        Args:
            pipeline: RAGPipeline instance
        """
        super().__init__()
        self._pipeline = pipeline

        # Shared components
        self.retriever = DSPyRetrieverAdapter(pipeline.retriever)
        self.context_manager = pipeline.context_manager

        # Three optimizable agents
        self.creative_agent = dspy.ChainOfThought(CreativeProposalSignature)
        self.conservative_agent = dspy.ChainOfThought(ConservativeProposalSignature)
        self.policy_verifier = dspy.Predict(PolicyVerdictSignature)

    def forward(self, query: str) -> dspy.Prediction:
        """
        Run 3-agent voting.

        Args:
            query: User query

        Returns:
            dspy.Prediction with verdict_json, winning_agent, dissent_summary
        """
        # Shared retrieval + context
        retrieval_pred = self.retriever.forward(query)
        chunks = retrieval_pred.chunks
        passages_str = retrieval_pred.passages_str

        # Rerank
        reranked_chunks = self._pipeline.reranker.rerank(query, chunks)[:5]
        context_str = passages_str  # Simplified; could use context_manager

        # Run creative + conservative in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            creative_future = executor.submit(
                self.creative_agent, query=query, context=context_str
            )
            conservative_future = executor.submit(
                self.conservative_agent, query=query, context=context_str
            )

            creative_output = creative_future.result()
            conservative_output = conservative_future.result()

        creative_answer = creative_output.creative_answer
        conservative_answer = conservative_output.conservative_answer

        # PolicyVerifier chooses winner
        verifier_output = self.policy_verifier(
            query=query,
            creative_proposal=creative_answer,
            conservative_proposal=conservative_answer,
            context=context_str,
        )

        # Parse verdict
        try:
            verdict = json.loads(verifier_output.verdict_json)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse verifier JSON: {verifier_output.verdict_json}")
            verdict = {
                "decision": "escalate",
                "winning_agent": "error",
                "dissent_summary": "Policy verifier error",
                "policy_reasons": [],
            }

        return dspy.Prediction(
            verdict_json=verifier_output.verdict_json,
            decision=verdict.get("decision", "escalate"),
            winning_agent=verdict.get("winning_agent", "escalated"),
            dissent_summary=verdict.get("dissent_summary", ""),
            policy_reasons=verdict.get("policy_reasons", []),
            creative_answer=creative_answer,
            conservative_answer=conservative_answer,
        )
