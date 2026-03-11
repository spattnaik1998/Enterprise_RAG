"""
Unified Benchmark CLI for Sprint 4 Use Cases
---------------------------------------------
Runs all three benchmarks:
  1. Security Benchmark
  2. Consensus Benchmark
  3. Latency Benchmark

Usage:
    # Run all benchmarks in mock mode
    python -m eval.run_benchmarks

    # Run single benchmark
    python -m eval.run_benchmarks --use-case security --sample 5

    # Run with real pipeline (requires data/index/)
    python -m eval.run_benchmarks --use-case consensus --mock false

    # Full suite
    python -m eval.run_benchmarks --use-case all
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

console = Console()

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


app = typer.Typer(help="Sprint 4 Benchmark Suite")


@app.command()
def run(
    use_case: Literal["security", "consensus", "latency", "all"] = typer.Option(
        "all",
        help="Which benchmark(s) to run",
    ),
    sample: int = typer.Option(
        0,
        help="Number of queries per category (0 = all)",
    ),
    mock: bool = typer.Option(
        True,
        help="Use mock mode (no actual pipeline required)",
    ),
    output: str = typer.Option(
        "",
        help="Output JSON file (auto-timestamped if omitted)",
    ),
) -> None:
    """
    Run Sprint 4 benchmarks.

    Args:
        use_case: Which benchmark to run
        sample: Query sample size (0 = all)
        mock: Use mock mode (no LLM calls)
        output: Output JSON file path
    """
    console.print(
        Panel(
            "[bold cyan]Sprint 4 Benchmark Suite[/bold cyan]\n"
            f"Use Case: {use_case} | Sample: {sample} | Mock: {mock}",
            title="🚀 Benchmarking",
        )
    )

    # Run benchmarks
    results = {}

    if use_case in ["security", "all"]:
        try:
            console.print("\n[bold]1. Running Security Benchmark...[/bold]")
            from eval.benchmarks.security_benchmark import run_security_benchmark

            security_result = asyncio.run(
                run_security_benchmark(sample=sample, mock=mock)
            )
            results["security"] = {
                "true_positive_rate": security_result.true_positive_rate,
                "false_positive_rate": security_result.false_positive_rate,
                "p95_total_ms": security_result.p95_total_ms,
                "avg_total_ms": security_result.avg_total_ms,
                "attack_precision": security_result.attack_precision,
            }
        except Exception as e:
            console.print(f"[red]Security benchmark failed: {e}[/red]")
            logger.error(f"Security benchmark error: {e}")

    if use_case in ["consensus", "all"]:
        try:
            console.print("\n[bold]2. Running Consensus Benchmark...[/bold]")
            from eval.benchmarks.consensus_benchmark import run_consensus_benchmark

            consensus_result = asyncio.run(
                run_consensus_benchmark(sample=sample, mock=mock)
            )
            results["consensus"] = {
                "baseline_recall_at_10": consensus_result.baseline_recall_at_10,
                "consensus_recall_at_10": consensus_result.consensus_recall_at_10,
                "recall_uplift": consensus_result.recall_uplift,
                "baseline_faithfulness": consensus_result.baseline_faithfulness,
                "consensus_faithfulness": consensus_result.consensus_faithfulness,
                "faithfulness_uplift": consensus_result.faithfulness_uplift,
                "consensus_latency_ms": consensus_result.consensus_latency_ms,
                "hallucination_mitigation_rate": consensus_result.hallucination_mitigation_rate,
            }
        except Exception as e:
            console.print(f"[red]Consensus benchmark failed: {e}[/red]")
            logger.error(f"Consensus benchmark error: {e}")

    if use_case in ["latency", "all"]:
        try:
            console.print("\n[bold]3. Running Latency Benchmark...[/bold]")
            from eval.benchmarks.latency_benchmark import run_latency_benchmark

            latency_result = asyncio.run(
                run_latency_benchmark(sample=sample, mock=mock)
            )
            results["latency"] = {
                "predicted_latency_mae_ms": latency_result.predicted_latency_mae_ms,
                "actual_avg_latency_ms": latency_result.actual_avg_latency_ms,
                "packing_efficiency_mean": latency_result.packing_efficiency_mean,
                "correctness_vs_baseline": latency_result.correctness_vs_baseline,
                "cost_per_query_usd": latency_result.cost_per_query_usd,
                "fast_path_usage_rate": latency_result.fast_path_usage_rate,
            }
        except Exception as e:
            console.print(f"[red]Latency benchmark failed: {e}[/red]")
            logger.error(f"Latency benchmark error: {e}")

    # Save results
    if results:
        if not output:
            from datetime import datetime
            output = f"eval/results/benchmarks_{datetime.utcnow().isoformat()}.json"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n✅ Results saved to {output_path}")
    else:
        console.print("\n[red]No benchmarks ran successfully[/red]")


if __name__ == "__main__":
    app()
