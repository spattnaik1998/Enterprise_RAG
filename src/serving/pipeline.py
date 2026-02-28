"""
RAG Serving Pipeline
---------------------
Orchestrates the full Phase III query lifecycle:

    user query
        |
        v
    PromptGuard (injection detection)
        |
        v
    HybridRetriever (FAISS dense + BM25 sparse -> RRF fusion, top_k=10)
        |
        v
    LLMReranker (one OpenAI call scores all candidates, keep top_k=5)
        |
        v
    RAGGenerator (OpenAI gpt-4o-mini with grounded system prompt)
        |
        v
    PIIFilter (redact email/phone/SSN/CC/IP from output)
        |
        v
    QueryResult (answer + citations + timings + token cost)

The outer query() method is decorated with @traceable so LangSmith captures
the full chain -- retrieval, reranking, and generation -- in a single trace.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable
from loguru import logger

from src.embedding.embedder import Embedder
from src.embedding.faiss_index import FAISSIndex
from src.generation.generator import (
    RAGGenerator,
    RAGResponse,
    _MODEL_PRICING,
    _cost_usd,
)
from src.retrieval.guardrails import PIIFilter, PromptGuard
from src.retrieval.reranker import LLMReranker
from src.retrieval.retriever import HybridRetriever

# Windows cp1252 terminal fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """
    Full output from a single RAG query.

    Timing fields are in milliseconds. Token counts come from the
    OpenAI usage object.  estimated_cost_usd covers generation only
    (reranking tokens are small and tracked separately in logs).
    """

    query: str
    answer: str
    citations: list[dict]
    pii_redacted: list[str]

    # Guardrail
    blocked: bool = False
    blocked_reason: str = ""

    # Latency breakdown
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0

    # Token stats
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.rerank_ms + self.generation_ms

    @property
    def estimated_cost_usd(self) -> float:
        """Cost estimate using per-model pricing from _MODEL_PRICING table."""
        return _cost_usd(self.model, self.prompt_tokens, self.completion_tokens)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "pii_redacted": self.pii_redacted,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "latency_ms": {
                "retrieval": round(self.retrieval_ms, 1),
                "rerank": round(self.rerank_ms, 1),
                "generation": round(self.generation_ms, 1),
                "total": round(self.total_ms, 1),
            },
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.prompt_tokens + self.completion_tokens,
            },
            "model": self.model,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG serving pipeline.

    Loads the FAISS + BM25 index built by Phase II and exposes a single
    query() method that runs the complete retrieve -> rerank -> generate cycle.

    Usage:
        pipeline = RAGPipeline()
        result = pipeline.query("Which clients have overdue invoices?")
        print(result.answer)
        for cit in result.citations:
            print(cit["source"])
    """

    def __init__(
        self,
        index_dir: str = "data/index",
        top_k: int = 10,
        rerank_top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        generator_model: str = "gpt-4o-mini",
        reranker_model: str = "gpt-4o-mini",
        enable_reranking: bool = True,
        enable_pii_filter: bool = True,
    ) -> None:
        logger.info(f"[RAGPipeline] Loading index from {index_dir}...")
        self.index = FAISSIndex.load(Path(index_dir))

        self.embedder = Embedder()
        self.retriever = HybridRetriever(
            index=self.index,
            embedder=self.embedder,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        self.enable_reranking = enable_reranking
        self.reranker = (
            LLMReranker(model=reranker_model, rerank_top_k=rerank_top_k)
            if enable_reranking
            else None
        )
        self.rerank_top_k = rerank_top_k

        self.generator = RAGGenerator(model=generator_model)
        self.guard = PromptGuard()
        self.pii_filter = PIIFilter() if enable_pii_filter else None

        logger.info(
            f"[RAGPipeline] Ready | {self.index.faiss_index.ntotal} vectors | "
            f"model={generator_model} | rerank={enable_reranking}"
        )

    @traceable(name="rag_query", run_type="chain")
    def query(self, user_query: str, generator=None) -> QueryResult:
        """
        Run the full RAG pipeline for a single user query.

        Steps:
            1. Prompt injection guardrail check
            2. Hybrid retrieval (FAISS + BM25 + RRF)
            3. LLM reranking (top_k -> rerank_top_k)
            4. Answer generation with grounded context
            5. PII redaction on the generated answer

        Args:
            user_query: The raw question from the user.
            generator:  Optional generator instance (RAGGenerator or
                        AnthropicGenerator).  Falls back to self.generator
                        (gpt-4o-mini) when not provided.

        Returns:
            QueryResult with answer, citations, timing, and token stats.
        """
        logger.info(f"[RAGPipeline] Query: {user_query[:100]!r}")

        # -- 1. Guardrail -------------------------------------------------------
        guard_result = self.guard.check(user_query)
        if not guard_result.passed:
            return QueryResult(
                query=user_query,
                answer=guard_result.blocked_reason,
                citations=[],
                pii_redacted=[],
                blocked=True,
                blocked_reason=guard_result.blocked_reason,
            )

        # -- 2. Retrieve --------------------------------------------------------
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(user_query)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # -- 3. Rerank ----------------------------------------------------------
        t1 = time.perf_counter()
        if self.enable_reranking and self.reranker is not None:
            reranked = self.reranker.rerank(user_query, candidates)
        else:
            reranked = candidates[: self.rerank_top_k]
        rerank_ms = (time.perf_counter() - t1) * 1000

        # -- 4. Generate --------------------------------------------------------
        t2 = time.perf_counter()
        active_generator = generator if generator is not None else self.generator
        rag_response: RAGResponse = active_generator.generate(user_query, reranked)
        generation_ms = (time.perf_counter() - t2) * 1000

        # -- 5. PII Filter ------------------------------------------------------
        answer = rag_response.answer
        pii_redacted: list[str] = []
        if self.pii_filter:
            answer, pii_redacted = self.pii_filter.redact(answer)

        logger.info(
            f"[RAGPipeline] Complete | "
            f"retrieve={retrieval_ms:.0f}ms "
            f"rerank={rerank_ms:.0f}ms "
            f"generate={generation_ms:.0f}ms | "
            f"tokens={rag_response.total_tokens}"
        )

        return QueryResult(
            query=user_query,
            answer=answer,
            citations=rag_response.citations,
            pii_redacted=pii_redacted,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            model=rag_response.model,
            prompt_tokens=rag_response.prompt_tokens,
            completion_tokens=rag_response.completion_tokens,
        )
