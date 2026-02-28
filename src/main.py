"""
Enterprise RAG Pipeline - CLI Entry Point
------------------------------------------
Exposes Typer commands for each pipeline phase.

Usage:
    python -m src.main phase1               # Run full collection + validation
    python -m src.main phase1 --dry-run     # Validate config only
    python -m src.main phase2               # Chunk, embed, index
    python -m src.main phase3               # Interactive RAG Q&A
    python -m src.main phase3 --query "..."  # Single-shot query
    python -m src.main status               # Show current checkpoint state
"""
from __future__ import annotations

import sys

# Windows cp1252 terminal fix: force UTF-8 so RSS/external content with
# emoji does not crash the Rich console renderer.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.checkpoint import PhaseICheckpoint
from src.collection.pipeline import CollectionPipeline
from src.embedding.pipeline import run_phase2
from src.schemas import CollectionStats
from src.utils.helpers import ensure_dirs, load_json
from src.utils.logger import setup_logger
from src.validation.report import ValidationReportGenerator
from src.validation.validator import DocumentValidator

app = typer.Typer(
    name="enterprise-rag",
    help="Enterprise AI Research Intelligence Hub - RAG Pipeline CLI",
    add_completion=False,
)
console = Console()


# --- Helpers ------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Commands -----------------------------------------------------------------

@app.command()
def phase1(
    config: str = typer.Option(
        "config/config.yaml", "--config", "-c", help="Path to pipeline config YAML"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Parse config and exit without collecting data"
    ),
    skip_health: bool = typer.Option(
        False, "--skip-health", help="Skip source health checks"
    ),
) -> None:
    """
    Phase I: Collect data from all sources and run validation.

    \b
    Steps:
      1. Source health checks
      2. Data collection  (ArXiv - Wikipedia - RSS)
      3. Document validation  (7 quality checks)
      4. Validation report generation
      5. Human checkpoint - pipeline pauses here
    """
    asyncio.run(_phase1_async(config, dry_run, skip_health))


@app.command()
def phase2(
    validated_dir: str = typer.Option(
        "data/validated", "--validated-dir", help="Directory with Phase I validated documents"
    ),
    index_dir: str = typer.Option(
        "data/index", "--index-dir", help="Output directory for FAISS index"
    ),
    batch_size: int = typer.Option(
        512, "--batch-size", help="Embedding API batch size (max 2048)"
    ),
) -> None:
    """
    Phase II: Chunk, embed, and index validated documents.

    \b
    Steps:
      1. Load ValidatedDocuments from data/validated/
      2. Adaptive chunking (keep-whole / sentence-window / fixed-overlap)
      3. OpenAI text-embedding-3-small embeddings
      4. Build FAISS + BM25 dual index -> data/index/
    """
    from dotenv import load_dotenv
    load_dotenv()
    setup_logger()
    run_phase2(
        validated_dir=validated_dir,
        index_dir=index_dir,
        batch_size=batch_size,
    )


@app.command()
def phase3(
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Single query (omit for interactive loop)"
    ),
    index_dir: str = typer.Option(
        "data/index", "--index-dir", help="FAISS index directory"
    ),
    top_k: int = typer.Option(
        10, "--top-k", help="Candidates to retrieve before reranking"
    ),
    rerank_top_k: int = typer.Option(
        5, "--rerank-top-k", help="Chunks kept after LLM reranking"
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", help="OpenAI model for generation and reranking"
    ),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="Skip LLM reranking (faster, lower cost)"
    ),
    no_pii: bool = typer.Option(
        False, "--no-pii", help="Disable PII redaction on output"
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Print result as JSON (single-query mode only)"
    ),
) -> None:
    """
    Phase III: Retrieve, rerank, and generate answers from the RAG index.

    \b
    Steps per query:
      1. Prompt injection guardrail check
      2. Hybrid retrieval  (FAISS dense + BM25 sparse -> RRF fusion)
      3. LLM reranking     (one OpenAI call scores all candidates)
      4. Answer generation (gpt-4o-mini with grounded context)
      5. PII redaction     (email / phone / SSN / CC / IP)
    """
    from dotenv import load_dotenv

    load_dotenv()
    setup_logger()

    from src.serving.pipeline import RAGPipeline

    # Verify index exists before loading
    if not Path(index_dir).exists():
        console.print(
            f"[red]Index directory not found: {index_dir}[/red]\n"
            "Run Phase II first: [bold]python -m src.main phase2[/bold]"
        )
        raise typer.Exit(1)

    console.print()
    console.print(
        Panel(
            "[bold cyan]Enterprise RAG Pipeline[/bold cyan]\n"
            "[white]Phase III - Retrieval, Reranking & Generation[/white]",
            box=box.DOUBLE_EDGE,
            expand=False,
        )
    )

    with console.status("[cyan]Loading FAISS + BM25 index...[/cyan]"):
        pipeline = RAGPipeline(
            index_dir=index_dir,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            generator_model=model,
            reranker_model=model,
            enable_reranking=not no_rerank,
            enable_pii_filter=not no_pii,
        )

    console.print(
        f"[green][OK] Index loaded[/green] "
        f"| {pipeline.index.faiss_index.ntotal:,} vectors "
        f"| rerank={'off' if no_rerank else 'on'} "
        f"| pii-filter={'off' if no_pii else 'on'}"
    )

    # --- Single-shot mode -----------------------------------------------------
    if query:
        result = pipeline.query(query)
        if json_out:
            console.print_json(json.dumps(result.to_dict(), indent=2))
        else:
            _print_result(result)
        return

    # --- Interactive loop -----------------------------------------------------
    console.print()
    console.print(
        "[bold]Ask anything about TechVault's operations or AI/ML research.[/bold]"
    )
    console.print("[dim]Type 'exit', 'quit', or press Ctrl+C to quit.[/dim]\n")

    while True:
        try:
            raw = console.input("[bold cyan]You[/bold cyan] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        with console.status("[cyan]Thinking...[/cyan]"):
            result = pipeline.query(raw)

        _print_result(result)


def _print_result(result) -> None:
    """Render a QueryResult to the terminal using Rich."""
    if result.blocked:
        console.print(
            Panel(
                f"[red]Blocked:[/red] {result.blocked_reason}",
                title="[red]Guardrail[/red]",
                border_style="red",
                expand=False,
            )
        )
        return

    # Answer panel
    console.print()
    console.print(
        Panel(
            Markdown(result.answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            expand=True,
        )
    )

    # Citations table
    if result.citations:
        table = Table(
            "No.", "Source Type", "Title", "Source", "Score",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
        )
        for cit in result.citations:
            table.add_row(
                str(cit["index"]),
                cit["source_type"],
                cit["title"][:55] + ("..." if len(cit["title"]) > 55 else ""),
                cit["source"][:50] + ("..." if len(cit["source"]) > 50 else ""),
                f"{cit['relevance_score']:.2f}",
            )
        console.print(table)

    # PII notice
    if result.pii_redacted:
        console.print(
            f"[yellow]PII redacted:[/yellow] {', '.join(result.pii_redacted)}"
        )

    # Stats footer
    total_s = result.total_ms / 1000
    console.print(
        f"[dim]"
        f"retrieve={result.retrieval_ms:.0f}ms  "
        f"rerank={result.rerank_ms:.0f}ms  "
        f"generate={result.generation_ms:.0f}ms  "
        f"total={total_s:.1f}s  |  "
        f"tokens={result.prompt_tokens}+{result.completion_tokens}  "
        f"cost=${result.estimated_cost_usd:.5f}"
        f"[/dim]\n"
    )


@app.command()
def status() -> None:
    """Show the current pipeline checkpoint state."""
    path = Path("data/checkpoint_phase1.json")
    if not path.exists():
        console.print("[yellow]No Phase I checkpoint found.  Run: python -m src.main phase1[/yellow]")
        raise typer.Exit(1)

    state = load_json(path)
    console.print()
    console.print("[bold]Phase I Checkpoint[/bold]")
    console.print(f"  Status    : [yellow]{state.get('status')}[/yellow]")
    console.print(f"  Run ID    : [dim]{state.get('run_id')}[/dim]")
    console.print(f"  Timestamp : {state.get('timestamp')}")
    console.print(f"  Validated : [green]{state.get('validated_count')}[/green]")
    console.print(f"  Rejected  : [red]{state.get('rejected_count')}[/red]")
    console.print()
    console.print(f"  Next phase: [cyan]{state.get('next_phase')}[/cyan]")
    console.print(f"  Command   : [bold]{state.get('next_command')}[/bold]")
    console.print()
    console.print("[dim]Instructions:[/dim]")
    for line in state.get("instructions", "").splitlines():
        console.print(f"  {line}")


# --- Async Phase I ------------------------------------------------------------

async def _phase1_async(config_path: str, dry_run: bool, skip_health: bool) -> None:
    cfg = _load_config(config_path)

    log_cfg = cfg.get("logging", {})
    setup_logger(
        log_level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/pipeline.log"),
    )

    ensure_dirs("data/raw", "data/validated", "data/rejected", "logs")

    project = cfg.get("project", {})
    logger.info("=" * 60)
    logger.info(f"Project : {project.get('name', 'Enterprise RAG')}")
    logger.info(f"Phase   : I - Collection & Validation")
    logger.info(f"Started : {datetime.utcnow().isoformat()}")
    logger.info("=" * 60)

    if dry_run:
        console.print(
            "[green][OK] Config loaded successfully.  "
            "Dry-run mode -- no data will be collected.[/green]"
        )
        return

    pipeline = CollectionPipeline(cfg)
    validator = DocumentValidator(cfg.get("validation", {}))
    reporter = ValidationReportGenerator(output_dir="data")
    checkpoint = PhaseICheckpoint(cfg)
    stats = CollectionStats()

    # -- Step 1: Health checks -------------------------------------------------
    if not skip_health:
        console.print("\n[bold cyan]Step 1 / 4 - Source health checks[/bold cyan]")
        health = await pipeline.run_health_checks()
        ok_count = sum(1 for v in health.values() if v)
        console.print(f"[green][OK] {ok_count}/{len(health)} sources healthy[/green]")
    else:
        console.print("\n[yellow]Health checks skipped[/yellow]")

    # -- Step 2: Collect -------------------------------------------------------
    console.print("\n[bold cyan]Step 2 / 4 - Collecting data from sources[/bold cyan]")
    raw_docs = await pipeline.collect_all()
    stats.total_collected = len(raw_docs)
    stats.by_source = pipeline.stats.by_source
    console.print(f"[green][OK] {len(raw_docs)} raw documents collected[/green]")

    # -- Step 3: Validate ------------------------------------------------------
    console.print("\n[bold cyan]Step 3 / 4 - Running validation checks[/bold cyan]")
    validated, rejected = validator.validate_batch(raw_docs)
    stats.total_validated = len(validated)
    stats.total_rejected = len(rejected)
    console.print(
        f"[green][OK] {len(validated)} passed[/green]  "
        f"[red]{len(rejected)} rejected[/red]"
    )

    # -- Step 4: Report + Checkpoint -------------------------------------------
    console.print("\n[bold cyan]Step 4 / 4 - Generating report & saving checkpoint[/bold cyan]")
    report = reporter.generate(raw_docs, validated, rejected, stats)
    reporter.print_report(report)
    await checkpoint.save(validated, rejected, stats)


# --- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    app()
