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

from src.context.budget_classifier import classify_budget
from src.context.manager import ContextManager
from src.context.session_memory import SessionMemory
from src.observability.collector import TraceCollector, get_active_collector
from src.observability.schemas import TraceEvent
from src.observability.quality_monitor import QualityMonitor
from src.embedding.embedder import Embedder
from src.embedding.faiss_index import FAISSIndex  # used for local / CLI mode
from src.generation.generator import (
    RAGGenerator,
    RAGResponse,
    _MODEL_PRICING,
    _cost_usd,
)
from src.learning.hard_negative_miner import HardNegativeMiner
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

    # Context management (Feature 3)
    context_bundle: object = None   # ContextBundle | None

    # Observability (Feature 4)
    trace_id: str = ""

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
        index=None,                    # Pre-built index (SupabaseIndex or FAISSIndex).
                                       # When provided, index_dir is ignored.
        top_k: int = 10,
        rerank_top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        generator_model: str = "gpt-4o-mini",
        reranker_model: str = "gpt-4o-mini",
        enable_reranking: bool = True,
        enable_pii_filter: bool = True,
    ) -> None:
        if index is not None:
            logger.info("[RAGPipeline] Using pre-built index (Supabase mode).")
            self.index = index
        else:
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
        self.context_manager = ContextManager()

        # Tier 1 + 2: Quality monitoring and hard negative mining
        self.quality_monitor = QualityMonitor()
        self.hard_negative_miner = HardNegativeMiner()
        self.session_memory = SessionMemory()

        logger.info(
            f"[RAGPipeline] Ready | {self.index.ntotal} vectors | "
            f"model={generator_model} | rerank={enable_reranking}"
        )

    @traceable(name="rag_query", run_type="chain")
    def query(self, user_query: str, generator=None, fast_path: bool = False, session_id: str = "") -> QueryResult:
        """
        Run the full RAG pipeline for a single user query.
        """
        import hashlib
        logger.info(f"[RAGPipeline] Query: {user_query[:100]!r}")

        # Determine model for trace metadata
        active_gen = generator if generator is not None else self.generator
        model_name = getattr(active_gen, "model", "unknown")

        session_id = hashlib.sha256(user_query.encode()).hexdigest()[:12]

        with TraceCollector(
            session_id=session_id,
            query=user_query,
            model=model_name,
            user_role="msp",  # pipeline-level default; gateway sets accurate role
        ) as tc:
            # -- query_start event --
            tc.add_event(TraceEvent(
                event_type="query_start",
                payload={"query_redacted": user_query[:200], "fast_path": fast_path},
            ))

            # -- 1. Guardrail --------------------------------------------------
            guard_result = self.guard.check(user_query)
            if not guard_result.passed:
                tc.add_event(TraceEvent(
                    event_type="guardrail_block",
                    payload={"reason": guard_result.blocked_reason},
                ))
                tc.set_verdict("guardrail_block")
                return QueryResult(
                    query=user_query,
                    answer=guard_result.blocked_reason,
                    citations=[],
                    pii_redacted=[],
                    blocked=True,
                    blocked_reason=guard_result.blocked_reason,
                )

            # -- 1b. Session memory -----------------------------------------------
            session_history = ""
            if session_id:
                try:
                    turns = self.session_memory.load_history(session_id)
                    session_history = self.session_memory.format_history(turns)
                    if session_history:
                        tc.add_event(TraceEvent(
                            event_type="session_memory",
                            payload={"turns_loaded": len(turns), "history_chars": len(session_history)},
                        ))
                except Exception as mem_exc:
                    logger.warning(f"[RAGPipeline] Session memory failed (non-fatal): {mem_exc}")

            # -- 2. Retrieve ---------------------------------------------------
            t0 = time.perf_counter()
            candidates = self.retriever.retrieve(user_query)
            retrieval_ms = (time.perf_counter() - t0) * 1000
            tc.add_event(TraceEvent(
                event_type="retrieval",
                payload={
                    "n_candidates": len(candidates),
                    "chunk_ids": [getattr(c, "id", str(i)) for i, c in enumerate(candidates[:5])],
                },
                duration_ms=retrieval_ms,
            ))

            # -- 3. Rerank -----------------------------------------------------
            t1 = time.perf_counter()
            if self.enable_reranking and self.reranker is not None:
                reranked = self.reranker.rerank(user_query, candidates)
            else:
                reranked = candidates[: self.rerank_top_k]
            rerank_ms = (time.perf_counter() - t1) * 1000
            tc.add_event(TraceEvent(
                event_type="rerank",
                payload={
                    "n_reranked": len(reranked),
                    "skipped": not (self.enable_reranking and self.reranker is not None),
                },
                duration_ms=rerank_ms,
            ))

            # -- 4. Context management -----------------------------------------
            # ContextManager expects plain Chunk objects, not (Chunk, float) tuples.
            # Wrapped in try/except so any edge-case failure falls back to the
            # raw reranked list — keeping the RAG pipeline independent of the
            # observability / context-management layer.
            reranked_chunks_only = [chunk for chunk, _ in reranked]
            reranked_scores = {chunk.chunk_id: score for chunk, score in reranked}

            # -- Dynamic token budget ------------------------------------------
            budget_tokens = classify_budget(user_query) if not fast_path else 1024

            context_bundle = None
            try:
                context_bundle = self.context_manager.get_context(
                    query=user_query,
                    chunks=reranked_chunks_only,
                    budget_tokens=budget_tokens,
                    fast_path=fast_path,
                    scores=reranked_scores,
                )
                tc.add_event(TraceEvent(
                    event_type="context_pack",
                    payload={
                        "total_tokens": context_bundle.total_tokens,
                        "budget_tokens": context_bundle.budget_tokens,
                        "truncated": context_bundle.truncated,
                        "n_pieces": len(context_bundle.pieces),
                    },
                ))
            except Exception as ctx_exc:
                logger.warning(f"[RAGPipeline] ContextManager failed (using raw reranked): {ctx_exc}")
                context_bundle = None

            # Reconstruct (Chunk, float) tuples for generator.
            # If context_bundle succeeded and has pieces, use the LIM-reordered set.
            # Otherwise fall back to the reranker's order directly.
            if context_bundle is not None and context_bundle.pieces:
                chunks_for_gen = []
                for p in context_bundle.pieces:
                    chunk = reranked_chunks_only[p.chunk_index]
                    score = reranked_scores.get(chunk.chunk_id, p.relevance_score)
                    chunks_for_gen.append((chunk, score))
            else:
                chunks_for_gen = reranked

            # -- 5. Generate ---------------------------------------------------
            t2 = time.perf_counter()
            rag_response: RAGResponse = active_gen.generate(user_query, chunks_for_gen, session_history=session_history)
            generation_ms = (time.perf_counter() - t2) * 1000
            tc.add_event(TraceEvent(
                event_type="generate",
                payload={
                    "model": rag_response.model,
                    "answer_len": len(rag_response.answer or ""),
                    "n_citations": len(rag_response.citations),
                    "prompt_tokens": rag_response.prompt_tokens,
                    "completion_tokens": rag_response.completion_tokens,
                },
                duration_ms=generation_ms,
                cost_usd=_cost_usd(rag_response.model, rag_response.prompt_tokens, rag_response.completion_tokens),
            ))

            # -- 6. PII Filter -------------------------------------------------
            answer = rag_response.answer
            pii_redacted: list[str] = []
            if self.pii_filter:
                answer, pii_redacted = self.pii_filter.redact(answer)
                if pii_redacted:
                    tc.add_event(TraceEvent(
                        event_type="pii_redact",
                        payload={"redacted_types": pii_redacted},
                    ))
                    tc.set_verdict("pii_redacted")

            # -- verdict -------------------------------------------------------
            if tc.trace.verdict == "success":
                tc.add_event(TraceEvent(
                    event_type="verdict",
                    payload={"outcome": "success"},
                ))
            tc.set_verdict(tc.trace.verdict)

            logger.info(
                f"[RAGPipeline] Complete | "
                f"retrieve={retrieval_ms:.0f}ms "
                f"rerank={rerank_ms:.0f}ms "
                f"generate={generation_ms:.0f}ms | "
                f"tokens={rag_response.total_tokens}"
            )

            result = QueryResult(
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
                context_bundle=context_bundle,
                trace_id=tc.trace_id,
            )

            # Tier 1 + 2: Log to quality monitor and hard negative miner
            try:
                self.quality_monitor.log_result(
                    query=user_query,
                    answer=answer,
                    citations=rag_response.citations,
                    latency_ms=result.total_ms,
                    model=rag_response.model,
                )

                # Get current metrics and check for degradation
                alerts = self.quality_monitor.get_degradation_signal()
                if alerts:
                    for alert in alerts:
                        logger.warning(
                            f"[RAGPipeline] Quality alert: {alert.alert_type} "
                            f"({alert.current_value:.2f} < {alert.threshold:.2f})"
                        )
                        # Tier 2: Collect hard negatives when quality degrades
                        # (we collect a sample of failures for later retraining)
                        if alert.alert_type in ["recall_degradation", "source_hit_degradation"]:
                            self.hard_negative_miner.collect(
                                query=user_query,
                                answer=answer,
                                citations=rag_response.citations,
                                recall_score=0.0 if "recall" in alert.alert_type else 0.5,
                                source_hit_score=0.0 if "source_hit" in alert.alert_type else 0.5,
                                category="unknown",
                                model=rag_response.model,
                                latency_ms=result.total_ms,
                            )
            except Exception as monitor_exc:
                logger.error(f"[RAGPipeline] Monitoring error (non-fatal): {monitor_exc}")

            return result
