"""
Latency-Aware Context Optimization System (Sprint 4, Use Case 3)
---------------------------------------------------------------
Implements dynamic context assembly with latency awareness:

1. ContextPlannerAgent: analyzes query complexity → sets budget + priority sources
2. LatencyEstimatorAgent: predicts latency + cost based on chunks/model/tokens
3. ContextAssemblerAgent: uses ContextManager with dynamic budget
4. GenerationAgent: produces final answer with actual latency tracking

Usage:
    optimizer = LatencyAwareContextOptimizationSystem(context_manager=cm, generator=gen)
    result = await optimizer.optimize(query="...", chunks=[...], model="gpt-4o-mini")
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from src.observability.collector import get_active_collector, TraceEvent


@dataclass
class ContextPlanResult:
    """Output from ContextPlannerAgent."""
    query_complexity: Literal["simple", "moderate", "complex"]
    context_budget: int  # tokens
    priority_sources: list[str]
    expected_chunk_count: int
    latency_ms: float


@dataclass
class LatencyEstimateResult:
    """Output from LatencyEstimatorAgent."""
    estimated_latency_ms: float
    estimated_cost_usd: float
    fast_path_recommended: bool
    latency_breakdown: dict  # assembly_ms, generation_ms, etc.
    cost_breakdown: dict  # embeddings, reranking, generation
    latency_ms: float


@dataclass
class ContextAssemblyResult:
    """Output from ContextAssemblerAgent."""
    assembled_chunks: list
    token_count: int
    packing_efficiency: float  # token_count / context_budget
    freshness_adjusted_chunks: list
    tier_reordered_chunks: list
    assembly_latency_ms: float


@dataclass
class OptimizationResult:
    """Final output from GenerationAgent."""
    answer: str
    citations: list[dict]
    # Plan
    query_complexity: str
    context_budget: int
    priority_sources: list[str]
    # Estimates
    estimated_latency_ms: float
    estimated_cost_usd: float
    fast_path_recommended: bool
    # Actual
    actual_latency_ms: float
    actual_cost_usd: float
    token_count: int
    packing_efficiency: float
    # Quality
    answer_quality_score: float
    # Latency breakdown
    planning_ms: float
    estimation_ms: float
    assembly_ms: float
    generation_ms: float
    total_ms: float


class ContextPlannerAgent:
    """
    Analyzes query complexity and sets context budget accordingly.

    Complexity rules:
      - SIMPLE: single-keyword lookups, factual questions
      - MODERATE: comparisons, multi-entity queries
      - COMPLEX: reasoning, cross-source, what-if analysis
    """

    SIMPLE_KEYWORDS = ["what", "who", "when", "where", "list", "how much"]
    COMPLEX_KEYWORDS = [
        "why",
        "should",
        "recommend",
        "compare",
        "analyze",
        "risk",
        "escalate",
    ]

    def __init__(self) -> None:
        pass

    async def run(self, query: str) -> ContextPlanResult:
        """Plan context requirements."""
        start = time.time()

        query_lower = query.lower()

        # Classify complexity
        complex_match = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in query_lower)
        simple_match = sum(1 for kw in self.SIMPLE_KEYWORDS if kw in query_lower)

        if complex_match >= 2:
            complexity = "complex"
            context_budget = 4096
            expected_chunks = 15
        elif complex_match >= 1 or simple_match >= 2:
            complexity = "moderate"
            context_budget = 2048
            expected_chunks = 10
        else:
            complexity = "simple"
            context_budget = 1024
            expected_chunks = 5

        # Prioritize sources based on keywords
        priority_sources = []
        if any(kw in query_lower for kw in ["invoice", "payment", "balance", "overdue"]):
            priority_sources.append("billing")
        if any(
            kw in query_lower
            for kw in ["contract", "sla", "renewal", "termination", "penalty"]
        ):
            priority_sources.append("contracts")
        if any(kw in query_lower for kw in ["health", "profile", "account", "contact"]):
            priority_sources.append("crm")
        if any(kw in query_lower for kw in ["ticket", "psa", "technician", "resolved"]):
            priority_sources.append("psa")

        latency_ms = (time.time() - start) * 1000

        return ContextPlanResult(
            query_complexity=complexity,
            context_budget=context_budget,
            priority_sources=priority_sources,
            expected_chunk_count=expected_chunks,
            latency_ms=latency_ms,
        )


class LatencyEstimatorAgent:
    """
    Predicts latency and cost based on query characteristics and model.

    Simple linear model:
      latency_ms ≈ (n_chunks * 15) + (n_tokens / 100) + model_base
      cost_usd ≈ (n_chunks * 0.001) + model_cost_per_token
    """

    # Base latency (ms) for each model
    MODEL_LATENCY = {
        "gpt-4o-mini": 500,
        "gpt-4o": 1000,
        "claude-haiku-4-5-20251001": 400,
        "claude-sonnet-4-6": 800,
    }

    # Marginal cost (usd) per token for each model
    MODEL_COST = {
        "gpt-4o-mini": 0.00001,
        "gpt-4o": 0.00003,
        "claude-haiku-4-5-20251001": 0.00000080,
        "claude-sonnet-4-6": 0.000003,
    }

    def __init__(self) -> None:
        pass

    async def run(
        self,
        chunk_count: int,
        token_budget: int,
        model: str = "gpt-4o-mini",
    ) -> LatencyEstimateResult:
        """Estimate latency and cost."""
        start = time.time()

        # Latency estimate (ms)
        assembly_ms = 50  # ContextManager assembly time
        chunk_fetch_ms = chunk_count * 15
        generation_base_ms = self.MODEL_LATENCY.get(model, 500)
        token_processing_ms = (token_budget / 100) * 10  # rough estimate

        estimated_generation_ms = generation_base_ms + token_processing_ms
        estimated_latency_ms = assembly_ms + chunk_fetch_ms + estimated_generation_ms

        # Cost estimate (usd)
        # Assume roughly: tokens_for_context = chunk_count * 100 + token_budget
        context_tokens = chunk_count * 100
        total_tokens = context_tokens + token_budget

        cost_per_token = self.MODEL_COST.get(model, 0.00001)
        estimated_cost_usd = total_tokens * cost_per_token

        # Recommend fast path if latency > 5s or cost > $0.10
        fast_path = estimated_latency_ms > 5000 or estimated_cost_usd > 0.10

        latency_ms = (time.time() - start) * 1000

        return LatencyEstimateResult(
            estimated_latency_ms=estimated_latency_ms,
            estimated_cost_usd=max(estimated_cost_usd, 0.001),  # min $0.001
            fast_path_recommended=fast_path,
            latency_breakdown={
                "assembly_ms": assembly_ms,
                "chunk_fetch_ms": chunk_fetch_ms,
                "generation_base_ms": generation_base_ms,
                "token_processing_ms": token_processing_ms,
                "total_ms": estimated_latency_ms,
            },
            cost_breakdown={
                "context_tokens": context_tokens,
                "query_tokens": token_budget,
                "total_tokens": total_tokens,
                "cost_per_token": cost_per_token,
                "total_usd": estimated_cost_usd,
            },
            latency_ms=latency_ms,
        )


class ContextAssemblerAgent:
    """
    Uses ContextManager to assemble final context set with dynamic budget.

    Applies:
      - Freshness scoring
      - Tier-based reordering (priority at start/end to mitigate "lost in the middle")
      - Token budget enforcement
    """

    def __init__(self, context_manager) -> None:
        self._context_manager = context_manager

    async def run(
        self,
        query: str,
        chunks: list,
        context_budget: int,
        priority_sources: list[str] | None = None,
        fast_path: bool = False,
    ) -> ContextAssemblyResult:
        """Assemble final context set."""
        start = time.time()

        if not chunks:
            return ContextAssemblyResult(
                assembled_chunks=[],
                token_count=0,
                packing_efficiency=0.0,
                freshness_adjusted_chunks=[],
                tier_reordered_chunks=[],
                assembly_latency_ms=(time.time() - start) * 1000,
            )

        # Use ContextManager for intelligent assembly
        try:
            context_result = self._context_manager.get_context(
                query=query,
                chunks=chunks,
                budget_tokens=context_budget,
                fast_path=fast_path,
            )
            assembled = context_result.chunks if hasattr(context_result, 'chunks') else chunks
            token_count = context_result.token_count if hasattr(context_result, 'token_count') else 0
        except Exception as e:
            logger.warning(f"[ContextAssembler] ContextManager failed: {e}; using raw chunks")
            assembled = chunks[: context_budget // 100]  # rough estimate: 100 tokens per chunk
            token_count = len(assembled) * 100

        packing_efficiency = token_count / max(1, context_budget)

        latency_ms = (time.time() - start) * 1000

        return ContextAssemblyResult(
            assembled_chunks=assembled,
            token_count=token_count,
            packing_efficiency=min(packing_efficiency, 1.0),
            freshness_adjusted_chunks=assembled,  # ContextManager already applies freshness
            tier_reordered_chunks=assembled,  # ContextManager already applies tier reordering
            assembly_latency_ms=latency_ms,
        )


class GenerationAgent:
    """
    Generates final answer and tracks actual latency vs. predicted.
    """

    def __init__(self, generator) -> None:
        self._generator = generator

    async def run(
        self,
        query: str,
        context_chunks: list,
        model: str,
        plan_result: ContextPlanResult,
        estimate_result: LatencyEstimateResult,
    ) -> tuple[str, list, float, float, float]:
        """
        Generate answer and track performance.

        Returns:
            (answer, citations, actual_latency_ms, actual_cost_usd, quality_score)
        """
        start = time.time()

        try:
            # Generate answer using provided generator
            response = self._generator.generate(
                query=query,
                context_chunks=context_chunks,
                model=model,
            )

            answer = response.answer if hasattr(response, 'answer') else str(response)
            citations = response.citations if hasattr(response, 'citations') else []
            cost = response.estimated_cost_usd if hasattr(response, 'estimated_cost_usd') else 0.01

        except Exception as e:
            logger.error(f"[GenerationAgent] Generation failed: {e}")
            answer = f"Error generating answer: {str(e)}"
            citations = []
            cost = 0.0

        latency_ms = (time.time() - start) * 1000

        # Quality score: simple heuristic based on citation count + answer length
        citation_count = len(citations)
        answer_length = len(answer.split())
        quality_score = min(1.0, (citation_count / 3.0) * 0.5 + (answer_length / 200.0) * 0.5)

        return answer, citations, latency_ms, cost, quality_score


class LatencyAwareContextOptimizationSystem:
    """Main orchestrator for dynamic context optimization."""

    def __init__(self, context_manager, generator) -> None:
        self._planner = ContextPlannerAgent()
        self._estimator = LatencyEstimatorAgent()
        self._assembler = ContextAssemblerAgent(context_manager)
        self._generator = GenerationAgent(generator)

    async def optimize(
        self,
        query: str,
        chunks: list | None = None,
        model: str = "gpt-4o-mini",
    ) -> OptimizationResult:
        """
        Run full optimization pipeline.

        Args:
            query: User query
            chunks: Retrieved context chunks
            model: LLM model to use

        Returns:
            OptimizationResult with predicted vs. actual performance
        """
        total_start = time.time()

        # Record trace
        collector = get_active_collector()
        if collector:
            collector.record(
                TraceEvent(
                    event_type="optimization_start",
                    data={"query_length": len(query), "model": model},
                )
            )

        if not chunks:
            chunks = []

        # Stage 1: Plan
        plan_result = await self._planner.run(query)

        # Stage 2: Estimate
        estimate_result = await self._estimator.run(
            chunk_count=len(chunks),
            token_budget=plan_result.context_budget,
            model=model,
        )

        # Stage 3: Assemble
        assembly_result = await self._assembler.run(
            query=query,
            chunks=chunks,
            context_budget=plan_result.context_budget,
            priority_sources=plan_result.priority_sources,
            fast_path=estimate_result.fast_path_recommended,
        )

        # Stage 4: Generate
        answer, citations, gen_latency_ms, gen_cost, quality_score = await self._generator.run(
            query=query,
            context_chunks=assembly_result.assembled_chunks,
            model=model,
            plan_result=plan_result,
            estimate_result=estimate_result,
        )

        total_latency_ms = (time.time() - total_start) * 1000

        result = OptimizationResult(
            answer=answer,
            citations=citations,
            # Plan
            query_complexity=plan_result.query_complexity,
            context_budget=plan_result.context_budget,
            priority_sources=plan_result.priority_sources,
            # Estimates
            estimated_latency_ms=estimate_result.estimated_latency_ms,
            estimated_cost_usd=estimate_result.estimated_cost_usd,
            fast_path_recommended=estimate_result.fast_path_recommended,
            # Actual
            actual_latency_ms=total_latency_ms,
            actual_cost_usd=gen_cost,
            token_count=assembly_result.token_count,
            packing_efficiency=assembly_result.packing_efficiency,
            # Quality
            answer_quality_score=quality_score,
            # Breakdown
            planning_ms=plan_result.latency_ms,
            estimation_ms=estimate_result.latency_ms,
            assembly_ms=assembly_result.assembly_latency_ms,
            generation_ms=gen_latency_ms,
            total_ms=total_latency_ms,
        )

        if collector:
            collector.record(
                TraceEvent(
                    event_type="optimization_complete",
                    data={
                        "complexity": result.query_complexity,
                        "actual_ms": result.actual_latency_ms,
                        "estimated_ms": result.estimated_latency_ms,
                        "packing_efficiency": result.packing_efficiency,
                    },
                )
            )

        # Log performance
        latency_delta = result.actual_latency_ms - result.estimated_latency_ms
        logger.info(
            f"[Optimization] {result.query_complexity:10s} | "
            f"estimated={result.estimated_latency_ms:.0f}ms, "
            f"actual={result.actual_latency_ms:.0f}ms (Δ={latency_delta:+.0f}ms) | "
            f"packing={result.packing_efficiency:.2f} | "
            f"quality={result.answer_quality_score:.2f}"
        )

        return result
