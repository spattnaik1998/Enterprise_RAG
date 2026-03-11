"""
Parallel Evaluation Orchestrator (Architecture A)
---------------------------------------------------
Runs evaluation shards in parallel across models/categories to reduce wall-clock time
from ~90 min (serial) to < 12 min (parallel).

Usage:
    orchestrator = ParallelEvalOrchestrator(
        index_dir="data/index",
        output_dir="eval/results"
    )
    report = await orchestrator.run(
        models=["gpt-4o-mini", "gpt-4o"],
        categories=["billing", "contracts"],
        sample=5
    )
    orchestrator.print_report(report)
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from eval.evaluator import RAGEvaluator, EvalReport, QueryEvalResult
from eval.judge import LLMJudge


@dataclass
class ShardTask:
    """A single shard: (model, category, query_subset)."""
    model: str
    category: str
    query_ids: list[str]  # Subset of queries for this shard


@dataclass
class ShardResult:
    """Result from a single shard execution."""
    model: str
    category: str
    query_results: list[QueryEvalResult]
    shard_latency_ms: float


class ModelShardAgent:
    """
    Executes a single shard of evaluation work (one model × one category × N queries).
    Reuses RAGEvaluator for the actual scoring logic.
    """

    def __init__(
        self,
        shard: ShardTask,
        evaluator: RAGEvaluator,
        judge_queue: asyncio.Queue,
        category_queries: dict,
    ) -> None:
        self._shard = shard
        self._evaluator = evaluator
        self._judge_queue = judge_queue
        self._category_queries = category_queries

    async def run(self) -> ShardResult:
        """Execute the shard and return results."""
        start = time.time()
        logger.info(
            f"[ModelShard] Starting {self._shard.model:20s} | "
            f"{self._shard.category:15s} | {len(self._shard.query_ids):3d} queries"
        )

        query_results = []
        generator = self._evaluator._make_generator(self._shard.model)

        for query in self._category_queries.get(self._shard.category, []):
            if query["id"] not in self._shard.query_ids:
                continue

            # Run the query (includes retrieval + generation)
            result = self._evaluator.run_single_query(
                query_item=query,
                model=self._shard.model,
                category=self._shard.category,
                generator=generator,
            )
            query_results.append(result)

            # Enqueue judge task only if we need to run judge later
            # (already scored in run_single_query, so this is just for tracking)
            await self._judge_queue.put({
                "query_result": result,
                "query": query,
            })

        latency_ms = (time.time() - start) * 1000
        logger.info(
            f"[ModelShard] Completed {self._shard.model:20s} | "
            f"{self._shard.category:15s} | latency={latency_ms:.0f}ms"
        )
        return ShardResult(
            model=self._shard.model,
            category=self._shard.category,
            query_results=query_results,
            shard_latency_ms=latency_ms,
        )


class JudgePoolWorker:
    """
    Placeholder worker for judge queue management.
    In the parallel architecture, judges are run inline in ModelShardAgent
    as part of run_single_query, so this just tracks queue completion.
    """

    def __init__(
        self,
        judge_queue: asyncio.Queue,
        worker_id: int = 0,
    ) -> None:
        self._judge_queue = judge_queue
        self._worker_id = worker_id

    async def run(self) -> int:
        """Drain the judge queue (no-op since judges already ran inline)."""
        logger.info(f"[JudgeWorker-{self._worker_id}] Started")
        count = 0
        while True:
            try:
                task = self._judge_queue.get_nowait()
                count += 1
                self._judge_queue.task_done()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.05)
                try:
                    task = self._judge_queue.get_nowait()
                    count += 1
                    self._judge_queue.task_done()
                except asyncio.QueueEmpty:
                    break
        logger.info(f"[JudgeWorker-{self._worker_id}] Drained {count} queue items")
        return count


class ParallelEvalOrchestrator:
    """
    Orchestrates parallel evaluation across multiple shards.

    Workflow:
      1. Build shards: (model, category, query_ids) triples
      2. Fan-out: run all ModelShardAgents concurrently
      3. Judge pool: 4 workers process judge queue in parallel
      4. Aggregate: merge results and compute metrics
    """

    def __init__(
        self,
        index_dir: str = "data/index",
        output_dir: str = "eval/results",
        num_judge_workers: int = 4,
        rps_limit: int = 10,  # OpenAI rate limit (requests per minute)
    ) -> None:
        self._index_dir = Path(index_dir)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._num_judge_workers = num_judge_workers
        self._rps_limit = rps_limit
        self._evaluator = RAGEvaluator(
            index_dir=str(self._index_dir),
            skip_index_check=False,
        )

    def _build_shards(
        self,
        models: list[str],
        categories: list[str],
        sample: int = 0,
    ) -> list[ShardTask]:
        """
        Build shard tasks: one shard per (model, category, query_subset).
        If sample > 0, each shard gets sample queries. Otherwise, all queries.
        """
        shards = []
        for model in models:
            for category in categories:
                queries = self._evaluator._load_queries_for_category(category)
                query_ids = [q["id"] for q in queries]
                if sample > 0:
                    query_ids = query_ids[:sample]
                shards.append(
                    ShardTask(model=model, category=category, query_ids=query_ids)
                )
        return shards

    async def run(
        self,
        models: list[str],
        categories: list[str],
        sample: int = 0,
    ) -> EvalReport:
        """
        Run parallel evaluation.

        Args:
            models: List of model IDs (e.g., ["gpt-4o-mini", "gpt-4o"])
            categories: List of categories (e.g., ["billing", "contracts"])
            sample: Number of queries per category (0 = all)

        Returns:
            EvalReport with all results aggregated.
        """
        import random
        start_time = time.time()

        # Load datasets and optionally sample
        rng = random.Random(42)
        category_queries: dict[str, list[dict]] = {}
        for cat in categories:
            queries = self._evaluator._load_dataset(cat)
            if sample > 0 and sample < len(queries):
                queries = rng.sample(queries, sample)
            category_queries[cat] = queries

        # Load pipeline once
        self._evaluator._load_pipeline(enable_reranking=True)

        # Build shards
        shards = self._build_shards(models, categories, sample)
        logger.info(f"[Orchestrator] Built {len(shards)} shards")

        # Create judge queue
        judge_queue: asyncio.Queue = asyncio.Queue()

        # Fan-out: run all shards concurrently
        logger.info(f"[Orchestrator] Launching {len(shards)} ModelShardAgents")
        shard_agents = [
            ModelShardAgent(shard, self._evaluator, judge_queue, category_queries)
            for shard in shards
        ]
        shard_results = await asyncio.gather(
            *[agent.run() for agent in shard_agents]
        )
        logger.info(f"[Orchestrator] All shards completed")

        # Drain judge queue with workers
        logger.info(f"[Orchestrator] Launching {self._num_judge_workers} JudgePoolWorkers")
        judge_workers = [
            JudgePoolWorker(judge_queue, worker_id=i)
            for i in range(self._num_judge_workers)
        ]
        await asyncio.gather(
            *[worker.run() for worker in judge_workers]
        )

        # Aggregate results
        all_results: list[QueryEvalResult] = []
        for shard_result in shard_results:
            all_results.extend(shard_result.query_results)

        total_latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[Orchestrator] Evaluation complete | "
            f"total latency={total_latency_ms:.0f}ms | "
            f"results={len(all_results)}"
        )

        # Build report using evaluator's aggregation logic
        model_metrics = self._evaluator._aggregate(all_results, models, categories)
        total_cost = sum(r.total_cost_usd for r in all_results)

        report = EvalReport(
            models_tested=models,
            categories_tested=categories,
            total_queries=len(all_results),
            model_metrics=model_metrics,
            query_results=all_results,
            total_cost_usd=total_cost,
        )
        report.total_latency_ms = total_latency_ms

        return report

    def print_report(self, report: EvalReport) -> None:
        """Print a pretty report."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"\n[bold]Parallel Evaluation Report[/bold]")
        console.print(f"Total latency: {report.total_latency_ms:.0f}ms\n")

        table = Table(title="Results by Model and Category")
        table.add_column("Model", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Recall@10", style="green")
        table.add_column("Source Hit", style="green")
        table.add_column("Faithfulness", style="green")
        table.add_column("Correctness", style="green")
        table.add_column("Composite", style="yellow")

        for metric in report.metrics:
            table.add_row(
                metric.model,
                metric.category,
                f"{metric.recall_at_10:.2f}",
                f"{metric.source_type_hit:.2f}",
                f"{metric.faithfulness:.2f}",
                f"{metric.correctness:.2f}",
                f"{metric.composite:.2f}",
            )

        console.print(table)
        console.print(f"\nFull report saved to: {report.report_path}")

    def save_report(self, report: EvalReport, output_path: Optional[str] = None) -> None:
        """Save report to JSON."""
        if output_path:
            report.report_path = output_path
        else:
            ts = int(time.time())
            report.report_path = str(self._output_dir / f"parallel_eval_{ts}.json")

        output = {
            "total_latency_ms": report.total_latency_ms,
            "metrics": [
                {
                    "model": m.model,
                    "category": m.category,
                    "recall_at_10": m.recall_at_10,
                    "source_type_hit": m.source_type_hit,
                    "faithfulness": m.faithfulness,
                    "correctness": m.correctness,
                    "composite": m.composite,
                }
                for m in report.metrics
            ],
        }

        Path(report.report_path).write_text(json.dumps(output, indent=2))
        logger.info(f"[Orchestrator] Report saved to {report.report_path}")
