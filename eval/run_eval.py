"""
RAG Evaluation CLI
-------------------
Entry point for running the evaluation framework.

Usage:
    # Quick smoke test (1 model, billing only, 5 queries)
    python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5

    # Full production eval (all 4 models, all 80 queries)
    python -m eval.run_eval

    # Single-category deep-dive on two models
    python -m eval.run_eval --models gpt-4o-mini gpt-4o --category contracts crm

    # Skip reranking for faster/cheaper evaluation
    python -m eval.run_eval --no-rerank --models gpt-4o-mini

Exit codes:
    0  All tested models pass all production thresholds (PRODUCTION READY)
    1  One or more models fail at least one threshold (NOT READY)
"""
from __future__ import annotations

import sys

# Windows cp1252 terminal fix -- must come before any rich/loguru imports
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from eval.evaluator import (
    ALL_CATEGORIES,
    MODEL_REGISTRY,
    RAGEvaluator,
    EvalReport,
)

app = typer.Typer(
    name="eval",
    help="TechVault MSP Enterprise RAG -- Evaluation Framework",
    add_completion=False,
)
console = Console()


def _resolve_models(models: list[str]) -> list[str]:
    """Validate model names and return the resolved list."""
    valid = list(MODEL_REGISTRY.keys())
    for m in models:
        if m not in valid:
            console.print(
                f"[red]Unknown model: {m!r}. "
                f"Valid models: {valid}[/red]"
            )
            raise typer.Exit(code=2)
    return models


def _resolve_categories(categories: list[str]) -> list[str]:
    """Validate category names and return the resolved list."""
    if not categories:
        return ALL_CATEGORIES
    for c in categories:
        if c not in ALL_CATEGORIES:
            console.print(
                f"[red]Unknown category: {c!r}. "
                f"Valid categories: {ALL_CATEGORIES}[/red]"
            )
            raise typer.Exit(code=2)
    return categories


@app.command()
def main(
    models: Optional[List[str]] = typer.Option(
        None,
        "--models",
        "-m",
        help=(
            "Models to evaluate. Repeat for multiple: "
            "--models gpt-4o-mini --models gpt-4o. "
            "Default: all 4 models."
        ),
    ),
    category: Optional[List[str]] = typer.Option(
        None,
        "--category",
        "-c",
        help=(
            "Categories to evaluate. Repeat for multiple: "
            "--category billing --category crm. "
            "Default: all 6 categories."
        ),
    ),
    sample: int = typer.Option(
        0,
        "--sample",
        "-n",
        help="If > 0, randomly sample this many queries per category. 0 = use all queries.",
        min=0,
    ),
    no_rerank: bool = typer.Option(
        False,
        "--no-rerank",
        help="Skip LLM reranking (faster and cheaper, lower quality).",
    ),
    output: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Path to save JSON report. Defaults to eval/results/eval_YYYYMMDD_HHMMSS.json.",
    ),
    judge_model: str = typer.Option(
        "gpt-4o-mini",
        "--judge-model",
        help="OpenAI model to use as LLM judge.",
    ),
    top_k: int = typer.Option(
        20,
        "--top-k",
        help="Number of candidate chunks retrieved before reranking.",
        min=1,
    ),
    rerank_top_k: int = typer.Option(
        10,
        "--rerank-top-k",
        help="Number of chunks kept after LLM reranking.",
        min=1,
    ),
    index_dir: str = typer.Option(
        "data/index",
        "--index-dir",
        help="Directory containing the FAISS index and BM25 corpus.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducible query sampling.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress per-query progress output.",
    ),
) -> None:
    """Run the RAG evaluation suite and print a production readiness report."""

    # Defaults
    resolved_models = _resolve_models(models or list(MODEL_REGISTRY.keys()))
    resolved_categories = _resolve_categories(category or [])

    # Auto-generate output path
    if not output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"eval/results/eval_{ts}.json"

    # Print run configuration
    console.rule("[bold cyan]TechVault MSP RAG Evaluation")
    console.print(f"  Models:       {', '.join(resolved_models)}")
    console.print(f"  Categories:   {', '.join(resolved_categories)}")
    console.print(f"  Sample/cat:   {'all' if sample == 0 else sample}")
    console.print(f"  Reranking:    {'OFF' if no_rerank else 'ON'}")
    console.print(f"  Top-K:        {top_k} -> {rerank_top_k}")
    console.print(f"  Judge model:  {judge_model}")
    console.print(f"  Output:       {output}")
    console.print()

    # Build evaluator
    evaluator = RAGEvaluator(
        index_dir=index_dir,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        enable_pii_filter=False,
        judge_model=judge_model,
    )

    # Set up rich progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=quiet,
    )

    # Estimate total queries
    from eval.evaluator import _CATEGORY_FILES
    import json as _json
    total_estimate = 0
    for cat in resolved_categories:
        try:
            path = Path(__file__).parent / "datasets" / _CATEGORY_FILES[cat]
            with open(path) as f:
                data = _json.load(f)
            n = len(data["queries"])
            total_estimate += min(n, sample) if sample > 0 else n
        except Exception:
            total_estimate += 10  # fallback estimate

    total_estimate *= len(resolved_models)

    with progress:
        task_id = progress.add_task(
            f"[cyan]Evaluating ({len(resolved_models)} models)...",
            total=total_estimate,
        )

        def on_progress(done: int, total: int, model: str, qid: str) -> None:
            progress.update(
                task_id,
                completed=done,
                description=f"[cyan]{model} | {qid}",
            )

        report: EvalReport = evaluator.run(
            models=resolved_models,
            categories=resolved_categories,
            sample_n=sample,
            enable_reranking=not no_rerank,
            rng_seed=seed,
            progress_callback=on_progress,
        )

        progress.update(task_id, completed=total_estimate)

    # Print summary table
    evaluator.print_report(report)

    # Print overall verdict
    all_pass = all(
        m.passes_all_thresholds()
        for m in report.model_metrics.values()
    )

    if all_pass:
        console.print(
            "[bold green]VERDICT: ALL models pass production thresholds. "
            "PRODUCTION READY.[/bold green]"
        )
    else:
        failing = [
            name
            for name, m in report.model_metrics.items()
            if not m.passes_all_thresholds()
        ]
        console.print(
            f"[bold red]VERDICT: {len(failing)} model(s) FAILED: "
            f"{', '.join(failing)}. NOT READY FOR PRODUCTION.[/bold red]"
        )

    console.print(
        f"\n[dim]Total cost: ${report.total_cost_usd:.3f} USD | "
        f"Queries evaluated: {report.total_queries}[/dim]"
    )

    # Save JSON report
    try:
        evaluator.save_report(report, output)
        console.print(f"[green]Report saved:[/green] {output}")
    except Exception as exc:
        console.print(f"[red]Failed to save report: {exc}[/red]")

    # CI-friendly exit code
    raise typer.Exit(code=0 if all_pass else 1)


if __name__ == "__main__":
    app()
