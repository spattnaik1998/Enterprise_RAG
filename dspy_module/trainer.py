"""
DSPy Modular Training CLI
---------------------------
Automates prompt optimization and few-shot demonstration compilation
using BootstrapFewShot or MIPRO optimizers.

Commands:
  train-rag        — Optimize RAGModule generation prompts
  train-reranker   — Optimize RerankerModule scoring prompts
  train-council    — Optimize CouncilModule voting logic
  evaluate         — Run evaluation on compiled programs
  compare          — Side-by-side baseline vs compiled performance

Cost estimates:
  Bootstrap cheap (billing only):  ~$0.01
  Bootstrap full (all cats):       ~$0.08
  Bootstrap + LLM judge:           ~$0.35
  MIPRO full (20 trials):          ~$2.50

Usage:
  pip install dspy-ai>=2.5
  python -m dspy_module.trainer train-rag --optimizer bootstrap --cheap
  python -m dspy_module.trainer evaluate --program dspy_module/compiled/rag_latest.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import dspy
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from dspy_module.dataset import MSPDataset
from dspy_module.metrics import (
    keyword_recall_metric,
    source_type_metric,
    faithfulness_metric,
    correctness_metric,
    rag_composite_metric,
    cheap_metric,
)
from dspy_module.modules import (
    DSPyRAGModule,
    DSPyRerankerModule,
    DSPyCouncilModule,
)
from dspy_module.signatures import (
    RAGSignature,
    CreativeProposalSignature,
    ConservativeProposalSignature,
    PolicyVerdictSignature,
)
from src.serving.pipeline import RAGPipeline

# Windows UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(help="DSPy Modular Training Framework for Enterprise RAG")
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def _load_pipeline(index_dir: str) -> RAGPipeline:
    """Load RAGPipeline from local FAISS index."""
    index_path = Path(index_dir)
    if not (index_path / "faiss.index").exists():
        console.print(
            f"[red]ERROR[/red] Index not found at {index_dir}",
            style="bold red",
        )
        console.print("\nRun Phase II first:", style="yellow")
        console.print("  python -m src.main phase2", style="dim")
        raise typer.Exit(1)

    try:
        from src.embedding.faiss_index import FAISSIndex

        index = FAISSIndex.load(index_dir=index_path)  # Pass Path object
        # RAGPipeline creates its own embedder internally
        pipeline = RAGPipeline(index=index)
        console.print(f"[green]✓[/green] Loaded pipeline from {index_dir}")
        return pipeline
    except Exception as e:
        console.print(f"[red]Error loading pipeline:[/red] {e}", style="bold red")
        logger.exception("Pipeline load traceback")
        raise typer.Exit(1)


def _configure_lm(model: str) -> None:
    """Configure DSPy to use specified model."""
    dspy.configure(lm=dspy.LM(f"openai/{model}"))
    logger.info(f"[Trainer] DSPy LM configured: {model}")


def _save_compiled(program: dspy.Module, output_path: Path) -> None:
    """Save compiled program as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Only save the optimizable component (generate_answer)
    # The retriever, reranker, context_manager are not trainable
    try:
        program.generate_answer.save(str(output_path))
    except Exception as e:
        logger.warning(f"Failed to save generate_answer only, trying full module: {e}")
        program.save(str(output_path))

    logger.info(f"[Trainer] Saved compiled program to {output_path}")

    # Copy to latest
    latest_path = Path(__file__).parent / "compiled" / "rag_latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path) as src, open(latest_path, "w") as dst:
        dst.write(src.read())
    logger.info(f"[Trainer] Updated {latest_path}")


def _filter_examples(
    examples: list[dspy.Example], categories: Optional[list[str]] = None
) -> list[dspy.Example]:
    """Filter examples by category."""
    if not categories:
        return examples
    return [e for e in examples if e.category in categories]


# ─────────────────────────────────────────────────────────────────────────────
# train-rag Command
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def train_rag(
    optimizer: str = typer.Option(
        "bootstrap",
        "--optimizer",
        help="Optimizer: bootstrap | mipro",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        help="LLM model for generation",
    ),
    teacher_model: str = typer.Option(
        "gpt-4o-mini",
        "--teacher-model",
        help="Teacher model for MIPRO (default: gpt-4o-mini)",
    ),
    max_bootstrapped_demos: int = typer.Option(
        3,
        "--max-bootstrapped-demos",
        help="Max few-shot demos for BootstrapFewShot",
    ),
    num_candidates: int = typer.Option(
        10,
        "--num-candidates",
        help="MIPRO: num candidate programs",
    ),
    num_trials: int = typer.Option(
        20,
        "--num-trials",
        help="MIPRO: num optimization trials",
    ),
    cheap: bool = typer.Option(
        False,
        "--cheap",
        help="Use keyword metric only (no LLM judge, ~$0.08)",
    ),
    categories: Optional[list[str]] = typer.Option(
        None,
        "--categories",
        help="Category subset: billing|contracts|crm|psa|communications|cross_source (repeat for multiple)",
    ),
    index_dir: str = typer.Option(
        "data/index",
        "--index-dir",
        help="Path to FAISS index directory",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        help="Output path for compiled program (auto-timestamped if omitted)",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """
    Train DSPyRAGModule with BootstrapFewShot or MIPRO.

    Optimizes generation prompts and few-shot demonstrations.
    """
    console.print("\n" + "=" * 70, style="cyan")
    console.print("DSPy RAG Module Training", style="bold cyan")
    console.print("=" * 70)

    start_time = time.time()

    try:
        # Load pipeline + dataset
        pipeline = _load_pipeline(index_dir)
        dataset = MSPDataset(seed=seed)
        dataset.print_summary()

        # Filter by category
        train_examples = _filter_examples(dataset.train, categories)
        dev_examples = _filter_examples(dataset.dev, categories)

        console.print(f"[cyan]Using {len(train_examples)} train, {len(dev_examples)} dev examples[/cyan]")

        # Configure DSPy
        _configure_lm(model)

        # Instantiate module
        module = DSPyRAGModule(pipeline)

        # Select metric
        if cheap:
            metric_fn = cheap_metric
            console.print("[yellow]Using cheap metric (keyword recall)[/yellow]")
        else:
            metric_fn = rag_composite_metric
            console.print("[yellow]Using composite metric (recall + faithfulness + correctness)[/yellow]")

        # Run optimizer
        console.print(f"\n[cyan]Starting {optimizer} optimization...[/cyan]")

        if optimizer.lower() == "bootstrap":
            optimizer_instance = dspy.BootstrapFewShot(
                metric=metric_fn,
                max_bootstrapped_demos=max_bootstrapped_demos,
            )
        elif optimizer.lower() == "mipro":
            teacher_lm = dspy.LM(f"openai/{teacher_model}")
            optimizer_instance = dspy.MIPROv2(
                metric=metric_fn,
                num_candidates=num_candidates,
                num_trials=num_trials,
                teacher_lm=teacher_lm,
            )
        else:
            console.print(f"[red]Unknown optimizer: {optimizer}[/red]")
            raise typer.Exit(1)

        # Compile (BootstrapFewShot only uses trainset for demo selection)
        compiled = optimizer_instance.compile(
            student=module,
            trainset=train_examples,
        )

        # Save
        if not output:
            timestamp = int(time.time())
            output = f"dspy_module/compiled/rag_{timestamp}.json"

        _save_compiled(compiled, Path(output))

        # Evaluate on dev set
        console.print("\n[cyan]Evaluating on dev set...[/cyan]")
        dev_scores = []
        for example in dev_examples[:5]:  # Sample first 5 for quick eval
            pred = compiled.forward(query=example.query)
            score = metric_fn(example, pred)
            dev_scores.append(score)

        if dev_scores:
            avg_score = sum(dev_scores) / len(dev_scores)
            console.print(f"[green]✓[/green] Dev sample score: {avg_score:.4f}")

        elapsed = time.time() - start_time
        console.print(
            f"\n[green]✓ Training complete in {elapsed:.1f}s[/green]",
            style="bold green",
        )
        console.print(f"Compiled program: {output}\n", style="dim")

    except Exception as e:
        console.print(f"[red]ERROR:[/red] {e}", style="bold red")
        logger.exception("Training failed")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate Command
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def evaluate(
    program: str = typer.Argument(..., help="Path to compiled DSPy program JSON"),
    set_type: str = typer.Option("dev", "--set", help="Evaluate on: dev | gold"),
    index_dir: str = typer.Option("data/index", "--index-dir"),
) -> None:
    """
    Evaluate a compiled DSPy program on dev or gold set.
    """
    console.print("\n" + "=" * 70, style="cyan")
    console.print("DSPy Program Evaluation", style="bold cyan")
    console.print("=" * 70)

    try:
        program_path = Path(program)
        if not program_path.exists():
            console.print(f"[red]Program not found:[/red] {program}", style="bold red")
            raise typer.Exit(1)

        # Load compiled module
        compiled = dspy.Module.load(str(program_path))
        console.print(f"[green]✓[/green] Loaded {program}")

        # Load dataset
        dataset = MSPDataset()
        eval_set = dataset.dev if set_type == "dev" else dataset.gold
        console.print(f"[cyan]Evaluating {len(eval_set)} examples from {set_type} set[/cyan]\n")

        # Evaluate
        scores_by_metric = {
            "keyword_recall": [],
            "source_type": [],
        }

        for i, example in enumerate(eval_set):
            try:
                pred = compiled.forward(query=example.query)
                scores_by_metric["keyword_recall"].append(
                    keyword_recall_metric(example, pred)
                )
                scores_by_metric["source_type"].append(source_type_metric(example, pred))
            except Exception as e:
                logger.error(f"Error on example {i}: {e}")

        # Print results
        table = Table(title=f"Evaluation Results ({set_type})")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")

        for metric, scores in scores_by_metric.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            table.add_row(metric, f"{avg:.4f}")

        console.print(table)

    except Exception as e:
        console.print(f"[red]ERROR:[/red] {e}", style="bold red")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# compare Command
# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def compare(
    program: str = typer.Option(..., "--program", help="Path to compiled DSPy program"),
    num_queries: int = typer.Option(5, "--num", help="Number of queries to compare"),
    index_dir: str = typer.Option("data/index", "--index-dir"),
) -> None:
    """
    Compare baseline pipeline vs compiled DSPy program on sample queries.
    """
    console.print("\n" + "=" * 70, style="cyan")
    console.print("Baseline vs DSPy Comparison", style="bold cyan")
    console.print("=" * 70)

    try:
        # Load pipeline + compiled module
        pipeline = _load_pipeline(index_dir)
        compiled = dspy.Module.load(program)
        dataset = MSPDataset()

        # Sample queries
        sample = dataset.dev[:num_queries]

        table = Table(title="Query Comparison")
        table.add_column("Query", style="cyan", width=40)
        table.add_column("Baseline Recall", style="magenta")
        table.add_column("DSPy Recall", style="green")

        for example in sample:
            # Baseline
            baseline_result = pipeline.query(example.query)
            baseline_score = keyword_recall_metric(
                example,
                type("Pred", (), {"answer": baseline_result.answer, "citations": baseline_result.citations})(),
            )

            # DSPy
            dspy_pred = compiled.forward(query=example.query)
            dspy_score = keyword_recall_metric(example, dspy_pred)

            table.add_row(
                example.query[:40] + "...",
                f"{baseline_score:.3f}",
                f"{dspy_score:.3f}",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]ERROR:[/red] {e}", style="bold red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
