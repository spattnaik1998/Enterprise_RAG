"""
Council Orchestrator CLI demo.

Usage:
    python -m src.agents.council_cli --query "Should we escalate Alpine Financial?"
    python -m src.agents.council_cli  # interactive mode
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
import time

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

app_cli = typer.Typer(add_completion=False)
console = Console()


def _print_verdict(verdict) -> None:
    status = "[red]ESCALATED[/red]" if verdict.escalated else "[green]ACCEPTED[/green]"
    console.print(f"\n[bold cyan]Council Verdict[/bold cyan]  {status}")
    console.print(Panel(
        verdict.accepted_answer,
        title=f"[bold]{verdict.winning_agent}[/bold]",
        border_style="green" if not verdict.escalated else "red",
    ))

    t = Table(show_header=True, header_style="bold blue")
    t.add_column("Field")
    t.add_column("Value")
    t.add_row("Winning Agent",        verdict.winning_agent)
    t.add_row("Escalated",            str(verdict.escalated))
    t.add_row("Hallucination Detected", str(verdict.hallucination_detected))
    t.add_row("PII Concern",          str(verdict.pii_concern))
    t.add_row("Total Cost USD",       f"${verdict.total_cost_usd:.5f}")
    t.add_row("Latency ms",           f"{verdict.latency_ms:.0f}")
    t.add_row("Trace ID",             verdict.trace_id)
    console.print(t)

    if verdict.policy_reasons:
        console.print(f"[dim]Policy reasons: {'; '.join(verdict.policy_reasons)}[/dim]")
    if verdict.dissent_summary:
        console.print(f"[dim]Dissent: {verdict.dissent_summary}[/dim]")


async def _run_query(query: str) -> None:
    """Load pipeline and run a single council query."""
    with console.status("[yellow]Loading RAG pipeline...[/yellow]"):
        from src.serving.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    from src.agents.council import CouncilOrchestrator
    from src.observability.collector import TraceCollector
    from src.observability.schemas import TraceEvent

    council = CouncilOrchestrator(pipeline)

    # Set up observability tracking
    session_id = f"cli_{int(time.time() * 1000)}"

    with console.status(f"[yellow]Running council for: {query[:60]!r}...[/yellow]"):
        # Wrap council execution with TraceCollector for observability
        with TraceCollector(
            session_id=session_id,
            query=query,
            model="council-3-agent",
            user_role="msp",
        ) as tc:
            verdict = await council.run(query, session_id=session_id)

            # Record council verdict as a trace event
            tc.add_event(TraceEvent(
                event_type="council_verdict",
                payload={
                    "winning_agent": verdict.winning_agent,
                    "escalated": verdict.escalated,
                    "hallucination_detected": verdict.hallucination_detected,
                    "pii_concern": verdict.pii_concern,
                    "policy_reasons": verdict.policy_reasons[:3] if verdict.policy_reasons else [],
                },
                cost_usd=verdict.total_cost_usd,
            ))

            # Set final verdict based on council outcome
            if verdict.escalated:
                tc.set_verdict("escalated")
            elif verdict.hallucination_detected:
                tc.set_verdict("pii_redacted")
            else:
                tc.set_verdict("success")

    _print_verdict(verdict)


@app_cli.command()
def main(
    query: str = typer.Option("", "--query", "-q", help="Query to run (omit for interactive mode)"),
) -> None:
    """Council Orchestrator demo -- runs the 3-agent voting pattern."""
    if query.strip():
        asyncio.run(_run_query(query.strip()))
        return

    # Interactive mode
    console.print("[bold cyan]Red Key Sandbox Council Orchestrator[/bold cyan]  (type 'exit' to quit)")
    while True:
        try:
            q = console.input("[bold blue]council>[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            break
        asyncio.run(_run_query(q))


if __name__ == "__main__":
    app_cli()
