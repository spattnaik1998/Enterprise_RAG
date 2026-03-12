#!/usr/bin/env python
"""
Latency Benchmark Suite for RAG and Agent Systems
Runs test queries and measures response times, costs, and quality metrics.

Usage:
    python -m eval.latency_benchmark --mode rag --complexity simple
    python -m eval.latency_benchmark --mode agent --complexity all --sample 5
    python -m eval.latency_benchmark --mode both --complexity moderate --output results.json
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.serving.pipeline import RAGPipeline
from src.embedding.supabase_index import SupabaseIndex
from src.agents.council import CouncilOrchestrator


console = Console()


class LatencyBenchmark:
    """Run latency benchmarks on RAG and Agent systems."""

    def __init__(self):
        self.pipeline = None
        self.council = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "rag_results": [],
            "agent_results": [],
            "summary": {},
        }

    async def initialize(self):
        """Initialize pipeline and agent."""
        with console.status("[bold cyan]Initializing systems..."):
            try:
                index = SupabaseIndex()
                self.pipeline = RAGPipeline(index=index)
                self.council = CouncilOrchestrator(self.pipeline)
                logger.info(f"[Benchmark] Initialized | {index.ntotal} vectors")
            except Exception as e:
                console.print(f"[red]ERROR initializing pipeline: {e}[/red]")
                raise

    async def query_rag(self, query: str, session_id: str) -> dict:
        """Run a single RAG query and measure latency."""
        t0 = time.perf_counter()
        try:
            result = self.pipeline.query(query)
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "success": True,
                "latency_ms": latency_ms,
                "answer_length": len(result.answer or ""),
                "citation_count": len(result.citations),
                "cost_usd": result.estimated_cost_usd,
                "tokens": result.prompt_tokens + result.completion_tokens,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"[RAG] Error: {e}")
            return {
                "success": False,
                "latency_ms": latency_ms,
                "error": str(e),
            }

    async def query_agent(self, query: str, session_id: str) -> dict:
        """Run a single Agent query and measure latency."""
        t0 = time.perf_counter()
        try:
            verdict = await self.council.run(
                query=query,
                budget_tokens=3000,
                session_id=session_id,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "success": True,
                "latency_ms": latency_ms,
                "answer_length": len(verdict.accepted_answer or ""),
                "winning_agent": verdict.winning_agent,
                "escalated": verdict.escalated,
                "cost_usd": verdict.total_cost_usd,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"[Agent] Error: {e}")
            return {
                "success": False,
                "latency_ms": latency_ms,
                "error": str(e),
            }

    def load_test_queries(self, complexity: str) -> list[dict]:
        """Load test queries from JSON file."""
        test_file = Path(__file__).parent / "test_queries.json"
        if not test_file.exists():
            console.print(f"[red]Test queries file not found: {test_file}[/red]")
            raise FileNotFoundError(str(test_file))

        with open(test_file) as f:
            data = json.load(f)

        queries = []
        if complexity == "all":
            for level in ["simple", "moderate", "complex", "edge_cases"]:
                queries.extend(data[level]["queries"])
        else:
            queries = data[complexity]["queries"]

        return queries

    async def run_benchmark(
        self,
        mode: str = "both",
        complexity: str = "simple",
        sample: Optional[int] = None,
    ):
        """Run the benchmark suite."""
        # Load queries
        queries = self.load_test_queries(complexity)
        if sample:
            queries = queries[:sample]

        console.print(f"\n[bold cyan]Running {mode.upper()} Benchmark[/bold cyan]")
        console.print(f"Complexity: {complexity} | Queries: {len(queries)}")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for query_obj in queries:
                query_id = query_obj["id"]
                query_text = query_obj["query"]
                task_desc = f"{query_id}: {query_text[:60]}..."

                if mode in ["rag", "both"]:
                    task = progress.add_task(
                        f"[cyan]{task_desc} (RAG)", total=None
                    )
                    result = await self.query_rag(query_text, query_id)
                    result["query_id"] = query_id
                    result["query"] = query_text
                    result["category"] = query_obj["category"]
                    self.results["rag_results"].append(result)
                    progress.remove_task(task)

                if mode in ["agent", "both"]:
                    task = progress.add_task(
                        f"[green]{task_desc} (Agent)", total=None
                    )
                    result = await self.query_agent(query_text, query_id)
                    result["query_id"] = query_id
                    result["query"] = query_text
                    result["category"] = query_obj["category"]
                    self.results["agent_results"].append(result)
                    progress.remove_task(task)

    def compute_summary(self):
        """Compute summary statistics."""
        for results_key, results_list in [
            ("rag", self.results["rag_results"]),
            ("agent", self.results["agent_results"]),
        ]:
            if not results_list:
                continue

            successful = [r for r in results_list if r.get("success", False)]
            if not successful:
                self.results["summary"][results_key] = {
                    "total_queries": len(results_list),
                    "successful": 0,
                    "success_rate": 0.0,
                }
                continue

            latencies = [r["latency_ms"] for r in successful]
            costs = [r.get("cost_usd", 0) for r in successful]

            self.results["summary"][results_key] = {
                "total_queries": len(results_list),
                "successful": len(successful),
                "success_rate": len(successful) / len(results_list),
                "latency_ms": {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": sum(latencies) / len(latencies),
                    "p50": sorted(latencies)[len(latencies) // 2],
                    "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                },
                "cost_usd": {
                    "total": sum(costs),
                    "mean": sum(costs) / len(costs) if costs else 0,
                },
            }

    def display_results(self):
        """Display results in a formatted table."""
        console.print()

        # RAG Results Table
        if self.results["rag_results"]:
            console.print("[bold cyan]RAG Benchmark Results[/bold cyan]")
            rag_table = Table(show_header=True, header_style="bold magenta")
            rag_table.add_column("Query ID", style="cyan")
            rag_table.add_column("Latency (ms)", style="green")
            rag_table.add_column("Citations", style="yellow")
            rag_table.add_column("Cost (USD)", style="red")
            rag_table.add_column("Status", style="blue")

            for result in self.results["rag_results"]:
                status = "[green]OK[/green]" if result.get("success") else "[red]FAIL[/red]"
                latency = f"{result.get('latency_ms', 0):.0f}"
                citations = str(result.get("citation_count", 0))
                cost = f"${result.get('cost_usd', 0):.6f}"
                rag_table.add_row(
                    result["query_id"],
                    latency,
                    citations,
                    cost,
                    status,
                )

            console.print(rag_table)

        # Agent Results Table
        if self.results["agent_results"]:
            console.print("\n[bold green]Agent Benchmark Results[/bold green]")
            agent_table = Table(show_header=True, header_style="bold magenta")
            agent_table.add_column("Query ID", style="cyan")
            agent_table.add_column("Latency (ms)", style="green")
            agent_table.add_column("Agent", style="yellow")
            agent_table.add_column("Cost (USD)", style="red")
            agent_table.add_column("Status", style="blue")

            for result in self.results["agent_results"]:
                status = "[green]OK[/green]" if result.get("success") else "[red]FAIL[/red]"
                latency = f"{result.get('latency_ms', 0):.0f}"
                agent = result.get("winning_agent", "ERROR")
                cost = f"${result.get('cost_usd', 0):.6f}"
                agent_table.add_row(
                    result["query_id"],
                    latency,
                    agent,
                    cost,
                    status,
                )

            console.print(agent_table)

        # Summary Statistics
        if self.results["summary"]:
            console.print("\n[bold cyan]Summary Statistics[/bold cyan]")
            summary_table = Table(show_header=True, header_style="bold magenta")
            summary_table.add_column("System", style="cyan")
            summary_table.add_column("Queries", style="yellow")
            summary_table.add_column("Success Rate", style="green")
            summary_table.add_column("Mean Latency (ms)", style="red")
            summary_table.add_column("P95 Latency (ms)", style="red")
            summary_table.add_column("Total Cost (USD)", style="blue")

            for system in ["rag", "agent"]:
                if system in self.results["summary"]:
                    summary = self.results["summary"][system]
                    success_rate = f"{summary['success_rate']*100:.1f}%"
                    mean_latency = f"{summary['latency_ms']['mean']:.0f}"
                    p95_latency = f"{summary['latency_ms']['p95']:.0f}"
                    total_cost = f"${summary['cost_usd']['total']:.6f}"
                    summary_table.add_row(
                        system.upper(),
                        str(summary["total_queries"]),
                        success_rate,
                        mean_latency,
                        p95_latency,
                        total_cost,
                    )

            console.print(summary_table)

    def save_results(self, output_path: Optional[str] = None):
        """Save results to JSON file."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"eval/results/latency_benchmark_{timestamp}.json"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        console.print(f"\n[green]Results saved to: {output_file}[/green]")


async def main(
    mode: str = typer.Option(
        "both",
        "--mode",
        "-m",
        help="Test mode: rag, agent, or both",
    ),
    complexity: str = typer.Option(
        "simple",
        "--complexity",
        "-c",
        help="Query complexity: simple, moderate, complex, edge_cases, or all",
    ),
    sample: Optional[int] = typer.Option(
        None,
        "--sample",
        "-s",
        help="Sample N queries from the selected complexity (all if not specified)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path for results (auto-generated if not specified)",
    ),
):
    """Run latency benchmark on RAG and Agent systems."""
    benchmark = LatencyBenchmark()
    await benchmark.initialize()

    try:
        await benchmark.run_benchmark(mode=mode, complexity=complexity, sample=sample)
        benchmark.compute_summary()
        benchmark.display_results()
        benchmark.save_results(output)
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Benchmark failed: {e}[/red]")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
