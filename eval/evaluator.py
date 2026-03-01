"""
RAG Evaluation Framework
--------------------------
Measures hybrid search quality and LLM answer quality across all four
supported models, then issues a binary PASS/FAIL production decision.

Metrics (all must exceed thresholds for PRODUCTION READY):
  - Retrieval Recall@10    >= 0.80
  - Source Type Hit Rate   >= 0.85
  - Answer Faithfulness    >= 0.85  (LLM judge)
  - Answer Correctness     >= 0.75  (LLM judge)
  - Composite Score        >= 0.82  (mean of four metrics above)

Usage:
    from eval.evaluator import RAGEvaluator
    evaluator = RAGEvaluator()
    report = evaluator.run(models=["gpt-4o-mini"])
    evaluator.print_report(report)
    evaluator.save_report(report, "eval/results/run.json")
"""
from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from eval.judge import LLMJudge, JudgeResult

# Suppress noisy warnings from dependencies during eval
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, float] = {
    "recall_at_10":        0.80,
    "source_type_hit":     0.85,
    "faithfulness":        0.85,
    "correctness":         0.75,
    "composite":           0.82,
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "gpt-4o-mini": {
        "provider": "openai",
        "class": "RAGGenerator",
    },
    "gpt-4o": {
        "provider": "openai",
        "class": "RAGGenerator",
    },
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "class": "AnthropicGenerator",
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "class": "AnthropicGenerator",
    },
}

# Dataset files relative to this module
_DATASETS_DIR = Path(__file__).parent / "datasets"
_CATEGORY_FILES: dict[str, str] = {
    "billing":        "billing_queries.json",
    "contracts":      "contracts_queries.json",
    "crm":            "crm_queries.json",
    "psa":            "psa_queries.json",
    "communications": "communications_queries.json",
    "cross_source":   "cross_source_queries.json",
}
ALL_CATEGORIES = list(_CATEGORY_FILES.keys())

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryEvalResult:
    """Per-query evaluation result for one model."""

    query_id: str
    category: str
    model: str
    query: str

    # Retrieval metrics (binary)
    recall_hit: bool = False          # True if any expected keyword found
    source_type_hit: bool = False     # True if citation source_type matches

    # LLM judge scores
    faithfulness: float = 0.0
    correctness: float = 0.0

    # Raw pipeline output
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    blocked: bool = False

    # Token / cost accounting
    generation_prompt_tokens: int = 0
    generation_completion_tokens: int = 0
    generation_cost_usd: float = 0.0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.generation_cost_usd + self.judge_cost_usd


@dataclass
class CategoryMetrics:
    """Aggregated metrics for one (model, category) pair."""

    model: str
    category: str
    n_queries: int = 0
    recall_at_10: float = 0.0
    source_type_hit: float = 0.0
    faithfulness: float = 0.0
    correctness: float = 0.0
    composite: float = 0.0
    n_blocked: int = 0
    total_cost_usd: float = 0.0


@dataclass
class ModelMetrics:
    """Aggregated metrics for one model across all evaluated categories."""

    model: str
    n_queries: int = 0
    recall_at_10: float = 0.0
    source_type_hit: float = 0.0
    faithfulness: float = 0.0
    correctness: float = 0.0
    composite: float = 0.0
    n_blocked: int = 0
    total_cost_usd: float = 0.0
    category_breakdown: list[CategoryMetrics] = field(default_factory=list)

    def passes_all_thresholds(self) -> bool:
        """Return True if the model meets every production threshold."""
        return (
            self.recall_at_10   >= THRESHOLDS["recall_at_10"]
            and self.source_type_hit >= THRESHOLDS["source_type_hit"]
            and self.faithfulness    >= THRESHOLDS["faithfulness"]
            and self.correctness     >= THRESHOLDS["correctness"]
            and self.composite       >= THRESHOLDS["composite"]
        )

    @property
    def production_status(self) -> str:
        return "PRODUCTION READY" if self.passes_all_thresholds() else "NOT READY"


@dataclass
class EvalReport:
    """Full evaluation report across all tested models and categories."""

    models_tested: list[str]
    categories_tested: list[str]
    total_queries: int
    model_metrics: dict[str, ModelMetrics]      # model_name -> ModelMetrics
    query_results: list[QueryEvalResult]
    total_cost_usd: float
    thresholds: dict[str, float] = field(default_factory=lambda: dict(THRESHOLDS))

    def to_dict(self) -> dict:
        return {
            "models_tested": self.models_tested,
            "categories_tested": self.categories_tested,
            "total_queries": self.total_queries,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "thresholds": self.thresholds,
            "model_metrics": {
                model: {
                    "n_queries": m.n_queries,
                    "recall_at_10": round(m.recall_at_10, 4),
                    "source_type_hit": round(m.source_type_hit, 4),
                    "faithfulness": round(m.faithfulness, 4),
                    "correctness": round(m.correctness, 4),
                    "composite": round(m.composite, 4),
                    "n_blocked": m.n_blocked,
                    "total_cost_usd": round(m.total_cost_usd, 4),
                    "production_status": m.production_status,
                    "passes_all_thresholds": m.passes_all_thresholds(),
                    "category_breakdown": [
                        {
                            "category": c.category,
                            "n_queries": c.n_queries,
                            "recall_at_10": round(c.recall_at_10, 4),
                            "source_type_hit": round(c.source_type_hit, 4),
                            "faithfulness": round(c.faithfulness, 4),
                            "correctness": round(c.correctness, 4),
                            "composite": round(c.composite, 4),
                        }
                        for c in m.category_breakdown
                    ],
                }
                for model, m in self.model_metrics.items()
            },
            "query_results": [
                {
                    "id": r.query_id,
                    "category": r.category,
                    "model": r.model,
                    "query": r.query,
                    "recall_hit": r.recall_hit,
                    "source_type_hit": r.source_type_hit,
                    "faithfulness": round(r.faithfulness, 4),
                    "correctness": round(r.correctness, 4),
                    "blocked": r.blocked,
                    "total_cost_usd": round(r.total_cost_usd, 6),
                    "answer_preview": r.answer[:200] if r.answer else "",
                }
                for r in self.query_results
            ],
        }


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Runs the full evaluation suite against the RAG pipeline.

    Load the pipeline once at construction time; swap generators per model
    via pipeline.query(q, generator=...).
    """

    def __init__(
        self,
        index_dir: str = "data/index",
        top_k: int = 20,
        rerank_top_k: int = 10,
        enable_pii_filter: bool = False,
        judge_model: str = "gpt-4o-mini",
    ) -> None:
        self.index_dir = index_dir
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.enable_pii_filter = enable_pii_filter
        self.judge = LLMJudge(model=judge_model)
        self._pipeline = None  # lazy-loaded in run()

    def _load_pipeline(self, enable_reranking: bool = True):
        """Load the RAGPipeline (FAISS + BM25 index) once."""
        from src.serving.pipeline import RAGPipeline
        logger.info("[Evaluator] Loading RAGPipeline...")
        self._pipeline = RAGPipeline(
            index_dir=self.index_dir,
            top_k=self.top_k,
            rerank_top_k=self.rerank_top_k,
            enable_reranking=enable_reranking,
            enable_pii_filter=self.enable_pii_filter,
        )
        logger.info("[Evaluator] Pipeline loaded.")
        return self._pipeline

    def _load_dataset(self, category: str) -> list[dict]:
        """Load and return the query list for a given category."""
        filename = _CATEGORY_FILES.get(category)
        if filename is None:
            raise ValueError(
                f"Unknown category {category!r}. "
                f"Valid categories: {list(_CATEGORY_FILES.keys())}"
            )
        path = _DATASETS_DIR / filename
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["queries"]

    def _make_generator(self, model: str):
        """Instantiate the right generator class for a given model name."""
        info = MODEL_REGISTRY.get(model)
        if info is None:
            raise ValueError(
                f"Unknown model {model!r}. "
                f"Valid models: {list(MODEL_REGISTRY.keys())}"
            )
        if info["provider"] == "openai":
            from src.generation.generator import RAGGenerator
            return RAGGenerator(model=model)
        else:
            from src.generation.generator import AnthropicGenerator
            return AnthropicGenerator(model=model)

    def _check_recall(
        self,
        citations: list[dict],
        answer: str,
        expected_keywords: list[str],
    ) -> bool:
        """
        Return True if at least one expected keyword appears in:
          - any citation title or source field (retrieval signal), OR
          - the generated answer text (synthesis signal / fallback).

        Case-insensitive substring match.
        """
        if not expected_keywords:
            return True  # no keywords specified => pass

        # Build a single haystack from citations + answer
        haystack_parts = [answer.lower()]
        for cit in citations:
            haystack_parts.append(cit.get("title", "").lower())
            haystack_parts.append(cit.get("source", "").lower())
        haystack = " ".join(haystack_parts)

        return any(kw.lower() in haystack for kw in expected_keywords)

    def _check_source_type_hit(
        self,
        citations: list[dict],
        expected_source_types: list[str],
    ) -> bool:
        """
        Return True if at least one citation has a source_type that matches
        any of the expected_source_types (case-insensitive).
        """
        if not expected_source_types:
            return True
        retrieved_types = {
            cit.get("source_type", "").lower() for cit in citations
        }
        expected_lower = {t.lower() for t in expected_source_types}
        return bool(retrieved_types & expected_lower)

    def _format_context_for_judge(self, citations: list[dict]) -> str:
        """
        Format citation metadata as a readable context string for the judge.
        Chunk text is not available in QueryResult.citations, so we use
        title + source + relevance_score as a proxy.
        """
        if not citations:
            return "(no citations retrieved)"
        lines = []
        for cit in citations:
            lines.append(
                f"[{cit.get('index', '?')}] {cit.get('title', '')} "
                f"| source: {cit.get('source', '')} "
                f"| type: {cit.get('source_type', '')} "
                f"| score: {cit.get('relevance_score', 0.0):.4f}"
            )
        return "\n".join(lines)

    def run_single_query(
        self,
        query_item: dict,
        model: str,
        category: str,
        generator,
    ) -> QueryEvalResult:
        """
        Run the pipeline for one query item and return a QueryEvalResult.

        Exceptions are caught per-query so a single failure does not abort
        the entire evaluation run.
        """
        qid = query_item["id"]
        query_text = query_item["query"]
        ground_truth = query_item.get("ground_truth", "")
        expected_keywords = query_item.get("expected_keywords", [])
        expected_source_types = query_item.get("expected_source_types", [])

        result = QueryEvalResult(
            query_id=qid,
            category=category,
            model=model,
            query=query_text,
        )

        try:
            pipeline_result = self._pipeline.query(query_text, generator=generator)

            result.answer = pipeline_result.answer
            result.citations = pipeline_result.citations
            result.blocked = pipeline_result.blocked

            # Retrieval metrics
            result.recall_hit = self._check_recall(
                pipeline_result.citations,
                pipeline_result.answer,
                expected_keywords,
            )
            result.source_type_hit = self._check_source_type_hit(
                pipeline_result.citations,
                expected_source_types,
            )

            # Generation token cost
            result.generation_prompt_tokens = pipeline_result.prompt_tokens
            result.generation_completion_tokens = pipeline_result.completion_tokens
            result.generation_cost_usd = pipeline_result.estimated_cost_usd

            # LLM judge scores (skip if blocked)
            if not pipeline_result.blocked:
                context_str = self._format_context_for_judge(pipeline_result.citations)
                judge_result: JudgeResult = self.judge.score(
                    query=query_text,
                    answer=pipeline_result.answer,
                    ground_truth=ground_truth,
                    context_str=context_str,
                )
                result.faithfulness = judge_result.faithfulness
                result.correctness = judge_result.correctness
                result.judge_prompt_tokens = judge_result.prompt_tokens
                result.judge_completion_tokens = judge_result.completion_tokens
                result.judge_cost_usd = judge_result.cost_usd
            else:
                # Blocked queries score 0 on judge metrics
                result.faithfulness = 0.0
                result.correctness = 0.0
                logger.warning(f"[Evaluator] Blocked: {qid} | {pipeline_result.blocked_reason}")

        except Exception as exc:
            logger.warning(f"[Evaluator] Exception on {qid}: {exc}")
            result.blocked = True

        return result

    def run(
        self,
        models: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
        sample_n: int = 0,
        enable_reranking: bool = True,
        rng_seed: int = 42,
        progress_callback=None,
    ) -> EvalReport:
        """
        Run the full evaluation.

        Args:
            models:           List of model names to evaluate. Defaults to all 4.
            categories:       List of categories to evaluate. Defaults to all 6.
            sample_n:         If > 0, randomly sample this many queries per category.
            enable_reranking: Whether to use LLM reranking (True by default).
            rng_seed:         Random seed for sampling reproducibility.
            progress_callback: Optional callable(current, total, model, query_id)
                               called after each query completes.

        Returns:
            EvalReport with all metrics and raw query results.
        """
        if models is None:
            models = list(MODEL_REGISTRY.keys())
        if categories is None:
            categories = ALL_CATEGORIES

        # Validate inputs
        for m in models:
            if m not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model {m!r}")
        for c in categories:
            if c not in _CATEGORY_FILES:
                raise ValueError(f"Unknown category {c!r}")

        # Load pipeline once
        self._load_pipeline(enable_reranking=enable_reranking)

        # Load and optionally sample datasets
        rng = random.Random(rng_seed)
        category_queries: dict[str, list[dict]] = {}
        for cat in categories:
            queries = self._load_dataset(cat)
            if sample_n > 0 and sample_n < len(queries):
                queries = rng.sample(queries, sample_n)
            category_queries[cat] = queries

        total_queries_per_model = sum(len(q) for q in category_queries.values())
        total_queries_all = total_queries_per_model * len(models)
        logger.info(
            f"[Evaluator] Starting eval | "
            f"{len(models)} models x {total_queries_per_model} queries = "
            f"{total_queries_all} total pipeline calls"
        )

        all_results: list[QueryEvalResult] = []
        done = 0

        for model in models:
            logger.info(f"[Evaluator] === Model: {model} ===")
            generator = self._make_generator(model)

            for cat in categories:
                for query_item in category_queries[cat]:
                    result = self.run_single_query(
                        query_item=query_item,
                        model=model,
                        category=cat,
                        generator=generator,
                    )
                    all_results.append(result)
                    done += 1
                    if progress_callback:
                        progress_callback(done, total_queries_all, model, query_item["id"])

        # Aggregate metrics
        model_metrics = self._aggregate(all_results, models, categories)

        total_cost = sum(r.total_cost_usd for r in all_results)

        return EvalReport(
            models_tested=models,
            categories_tested=categories,
            total_queries=total_queries_all,
            model_metrics=model_metrics,
            query_results=all_results,
            total_cost_usd=total_cost,
        )

    def _aggregate(
        self,
        query_results: list[QueryEvalResult],
        models: list[str],
        categories: list[str],
    ) -> dict[str, ModelMetrics]:
        """Aggregate per-query results into per-model and per-category metrics."""
        model_metrics: dict[str, ModelMetrics] = {}

        for model in models:
            model_results = [r for r in query_results if r.model == model]
            cat_metrics_list: list[CategoryMetrics] = []

            for cat in categories:
                cat_results = [r for r in model_results if r.category == cat]
                if not cat_results:
                    continue

                n = len(cat_results)
                recall = sum(1 for r in cat_results if r.recall_hit) / n
                source_hit = sum(1 for r in cat_results if r.source_type_hit) / n
                faith = sum(r.faithfulness for r in cat_results) / n
                correct = sum(r.correctness for r in cat_results) / n
                composite = (recall + source_hit + faith + correct) / 4

                cat_metrics_list.append(
                    CategoryMetrics(
                        model=model,
                        category=cat,
                        n_queries=n,
                        recall_at_10=recall,
                        source_type_hit=source_hit,
                        faithfulness=faith,
                        correctness=correct,
                        composite=composite,
                        n_blocked=sum(1 for r in cat_results if r.blocked),
                        total_cost_usd=sum(r.total_cost_usd for r in cat_results),
                    )
                )

            # Overall model metrics
            n = len(model_results)
            if n == 0:
                model_metrics[model] = ModelMetrics(model=model)
                continue

            recall = sum(1 for r in model_results if r.recall_hit) / n
            source_hit = sum(1 for r in model_results if r.source_type_hit) / n
            faith = sum(r.faithfulness for r in model_results) / n
            correct = sum(r.correctness for r in model_results) / n
            composite = (recall + source_hit + faith + correct) / 4

            model_metrics[model] = ModelMetrics(
                model=model,
                n_queries=n,
                recall_at_10=recall,
                source_type_hit=source_hit,
                faithfulness=faith,
                correctness=correct,
                composite=composite,
                n_blocked=sum(1 for r in model_results if r.blocked),
                total_cost_usd=sum(r.total_cost_usd for r in model_results),
                category_breakdown=cat_metrics_list,
            )

        return model_metrics

    def print_report(self, report: EvalReport) -> None:
        """Print a rich summary table: rows=models, cols=metrics, PASS=green/FAIL=red."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()
            table = Table(
                title="RAG Evaluation Report",
                box=box.ROUNDED,
                show_lines=True,
                highlight=True,
            )

            # Columns
            table.add_column("Model", style="bold cyan", no_wrap=True)
            table.add_column(
                f"Recall@10\n(>={THRESHOLDS['recall_at_10']:.0%})",
                justify="center",
            )
            table.add_column(
                f"Src Hit\n(>={THRESHOLDS['source_type_hit']:.0%})",
                justify="center",
            )
            table.add_column(
                f"Faithfulness\n(>={THRESHOLDS['faithfulness']:.0%})",
                justify="center",
            )
            table.add_column(
                f"Correctness\n(>={THRESHOLDS['correctness']:.0%})",
                justify="center",
            )
            table.add_column(
                f"Composite\n(>={THRESHOLDS['composite']:.0%})",
                justify="center",
            )
            table.add_column("Blocked", justify="center")
            table.add_column("Cost $", justify="right")
            table.add_column("Status", justify="center")

            for model in report.models_tested:
                m = report.model_metrics.get(model)
                if m is None:
                    continue

                def _fmt(val: float, thresh: float) -> str:
                    color = "green" if val >= thresh else "red"
                    return f"[{color}]{val:.1%}[/{color}]"

                status_color = "bold green" if m.passes_all_thresholds() else "bold red"
                table.add_row(
                    model,
                    _fmt(m.recall_at_10, THRESHOLDS["recall_at_10"]),
                    _fmt(m.source_type_hit, THRESHOLDS["source_type_hit"]),
                    _fmt(m.faithfulness, THRESHOLDS["faithfulness"]),
                    _fmt(m.correctness, THRESHOLDS["correctness"]),
                    _fmt(m.composite, THRESHOLDS["composite"]),
                    str(m.n_blocked),
                    f"${m.total_cost_usd:.3f}",
                    f"[{status_color}]{m.production_status}[/{status_color}]",
                )

            console.print()
            console.print(table)
            console.print()

            # Per-category breakdown
            for model in report.models_tested:
                m = report.model_metrics.get(model)
                if m is None or not m.category_breakdown:
                    continue
                cat_table = Table(
                    title=f"Category Breakdown: {model}",
                    box=box.SIMPLE_HEAD,
                    show_lines=False,
                )
                cat_table.add_column("Category", style="cyan")
                cat_table.add_column("N", justify="right")
                cat_table.add_column("Recall", justify="center")
                cat_table.add_column("Src Hit", justify="center")
                cat_table.add_column("Faith", justify="center")
                cat_table.add_column("Correct", justify="center")
                cat_table.add_column("Composite", justify="center")

                for cat in m.category_breakdown:
                    cat_table.add_row(
                        cat.category,
                        str(cat.n_queries),
                        f"{cat.recall_at_10:.1%}",
                        f"{cat.source_type_hit:.1%}",
                        f"{cat.faithfulness:.1%}",
                        f"{cat.correctness:.1%}",
                        f"{cat.composite:.1%}",
                    )
                console.print(cat_table)
                console.print()

        except ImportError:
            # Fallback plain-text output if rich is not available
            print("\n=== RAG Evaluation Report ===")
            for model, m in report.model_metrics.items():
                print(f"\nModel: {model}")
                print(f"  Recall@10:    {m.recall_at_10:.1%}  (>= {THRESHOLDS['recall_at_10']:.0%})")
                print(f"  Source Hit:   {m.source_type_hit:.1%}  (>= {THRESHOLDS['source_type_hit']:.0%})")
                print(f"  Faithfulness: {m.faithfulness:.1%}  (>= {THRESHOLDS['faithfulness']:.0%})")
                print(f"  Correctness:  {m.correctness:.1%}  (>= {THRESHOLDS['correctness']:.0%})")
                print(f"  Composite:    {m.composite:.1%}  (>= {THRESHOLDS['composite']:.0%})")
                print(f"  Status:       {m.production_status}")

    def save_report(self, report: EvalReport, path: str) -> None:
        """Save the full report as JSON to the given path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"[Evaluator] Report saved to {out}")
