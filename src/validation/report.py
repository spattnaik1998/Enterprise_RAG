"""
Validation Report Generator
----------------------------
Produces a machine-readable JSON report + a Rich-formatted console
summary for the Phase I human checkpoint.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.schemas import CollectionStats, RejectedDocument, RawDocument, ValidatedDocument
from src.utils.helpers import save_json, truncate_text

console = Console()


class ValidationReportGenerator:
    """Builds and prints the Phase I validation report."""

    def __init__(self, output_dir: str = "data") -> None:
        self.output_dir = Path(output_dir)

    # --- Public API -----------------------------------------------------------

    def generate(
        self,
        raw_docs: list[RawDocument],
        validated_docs: list[ValidatedDocument],
        rejected_docs: list[RejectedDocument],
        stats: CollectionStats,
    ) -> dict:
        """Build the full report dict and save it to disk."""
        total = max(len(raw_docs), 1)
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "run_id": stats.run_id,
            "pipeline_phase": "I - Collection & Validation",
            "collection_summary": {
                "total_collected": len(raw_docs),
                "total_validated": len(validated_docs),
                "total_rejected": len(rejected_docs),
                "pass_rate": f"{len(validated_docs) / total * 100:.1f}%",
                "by_source_raw": stats.by_source,
            },
            "quality_statistics": self._quality_stats(validated_docs),
            "source_breakdown": self._source_breakdown(validated_docs),
            "rejection_analysis": self._rejection_analysis(rejected_docs),
            "data_profile": self._data_profile(validated_docs),
            "sample_documents": self._samples(validated_docs, n=5),
            "recommendations": self._recommendations(validated_docs, rejected_docs),
        }

        path = self.output_dir / "validation_report.json"
        save_json(report, path)
        logger.info(f"Validation report saved -> {path}")
        return report

    def print_report(self, report: dict) -> None:
        """Print a formatted summary to the console."""
        c = console

        c.print()
        c.print(
            Panel(
                "[bold cyan]Enterprise RAG Pipeline[/bold cyan]\n"
                "[white]Phase I - Collection & Validation Report[/white]",
                subtitle=f"[dim]{report['generated_at']}[/dim]",
                box=box.DOUBLE_EDGE,
                expand=False,
            )
        )

        # -- Collection Summary ------------------------------------------------
        s = report["collection_summary"]
        t = Table(title="Collection Summary", box=box.ROUNDED, show_header=True)
        t.add_column("Metric", style="cyan", no_wrap=True)
        t.add_column("Value", style="bold white")
        t.add_row("Total collected", str(s["total_collected"]))
        t.add_row("Passed validation", f"[green]{s['total_validated']}[/green]")
        t.add_row("Rejected", f"[red]{s['total_rejected']}[/red]")
        t.add_row("Pass rate", f"[bold]{s['pass_rate']}[/bold]")
        c.print(t)

        # -- By Source ---------------------------------------------------------
        c.print()
        t2 = Table(title="Validated Documents by Source", box=box.ROUNDED)
        t2.add_column("Source", style="cyan")
        t2.add_column("Count", style="bold white", justify="right")
        t2.add_column("Avg Quality Score", style="yellow", justify="right")
        for src, data in report["source_breakdown"].items():
            t2.add_row(src, str(data["count"]), f"{data['avg_quality']:.3f}")
        c.print(t2)

        # -- Quality Stats -----------------------------------------------------
        c.print()
        qs = report["quality_statistics"]
        c.print(
            Panel(
                f"Quality Scores - "
                f"Min: [red]{qs.get('min', 0):.3f}[/red]  "
                f"Max: [green]{qs.get('max', 0):.3f}[/green]  "
                f"Avg: [yellow]{qs.get('avg', 0):.3f}[/yellow]  "
                f"Median: [cyan]{qs.get('median', 0):.3f}[/cyan]",
                box=box.ROUNDED,
                expand=False,
            )
        )

        # -- Rejection Analysis ------------------------------------------------
        if report["rejection_analysis"]:
            c.print()
            t3 = Table(title="Rejection Reasons", box=box.ROUNDED)
            t3.add_column("Reason", style="red")
            t3.add_column("Count", style="bold white", justify="right")
            for reason, count in report["rejection_analysis"].items():
                t3.add_row(reason, str(count))
            c.print(t3)

        # -- Data Profile ------------------------------------------------------
        c.print()
        dp = report["data_profile"]
        if dp:
            c.print(
                Panel(
                    f"Total words: [bold]{dp.get('total_words', 0):,}[/bold]  "
                    f"Avg words/doc: [bold]{dp.get('avg_word_count', 0):.0f}[/bold]  "
                    f"Source types: [cyan]{', '.join(dp.get('source_types', []))}[/cyan]",
                    title="Data Profile",
                    box=box.ROUNDED,
                    expand=False,
                )
            )

        # -- Sample Documents --------------------------------------------------
        c.print()
        c.print("[bold cyan]Top-Quality Sample Documents:[/bold cyan]")
        for i, sample in enumerate(report["sample_documents"][:3], 1):
            c.print(
                Panel(
                    f"[bold]{sample['title']}[/bold]\n"
                    f"[dim]Source:[/dim] [cyan]{sample['source']}[/cyan]  "
                    f"[dim]Quality:[/dim] [yellow]{sample['quality_score']:.3f}[/yellow]  "
                    f"[dim]Words:[/dim] {sample['word_count']}\n\n"
                    f"{sample['preview']}",
                    title=f"#{i}",
                    box=box.ROUNDED,
                )
            )

        # -- Recommendations ---------------------------------------------------
        c.print()
        c.print("[bold cyan]Recommendations:[/bold cyan]")
        for rec in report["recommendations"]:
            c.print(f"  - {rec}")

        # -- Human Checkpoint Banner -------------------------------------------
        c.print()
        c.print(
            Panel(
                "[bold yellow]>>> HUMAN CHECKPOINT -- Phase I Complete[/bold yellow]\n\n"
                "Please review the data before proceeding to Phase II.\n\n"
                f"  [green]Validated docs :[/green]  data/validated/  ({report['collection_summary']['total_validated']} files)\n"
                f"  [red]Rejected docs  :[/red]  data/rejected/   ({report['collection_summary']['total_rejected']} files)\n"
                f"  [blue]Full report    :[/blue]  data/validation_report.json\n"
                f"  [blue]Raw documents  :[/blue]  data/raw/\n\n"
                "When satisfied, run Phase II:\n"
                "  [bold]python -m src.main phase2[/bold]",
                box=box.DOUBLE_EDGE,
                border_style="yellow",
            )
        )

    # --- Helpers --------------------------------------------------------------

    @staticmethod
    def _quality_stats(docs: list[ValidatedDocument]) -> dict:
        if not docs:
            return {}
        scores = sorted(d.quality_score for d in docs)
        n = len(scores)
        median = scores[n // 2] if n % 2 else (scores[n // 2 - 1] + scores[n // 2]) / 2
        return {
            "min": round(scores[0], 4),
            "max": round(scores[-1], 4),
            "avg": round(sum(scores) / n, 4),
            "median": round(median, 4),
        }

    @staticmethod
    def _source_breakdown(docs: list[ValidatedDocument]) -> dict:
        breakdown: dict[str, dict] = {}
        for doc in docs:
            key = doc.source_type.value
            if key not in breakdown:
                breakdown[key] = {"count": 0, "_scores": []}
            breakdown[key]["count"] += 1
            breakdown[key]["_scores"].append(doc.quality_score)
        for key, data in breakdown.items():
            scores = data.pop("_scores")
            data["avg_quality"] = round(sum(scores) / len(scores), 4)
        return breakdown

    @staticmethod
    def _rejection_analysis(docs: list[RejectedDocument]) -> dict:
        reasons: dict[str, int] = {}
        for doc in docs:
            for r in doc.rejection_reasons:
                reasons[r] = reasons.get(r, 0) + 1
        return dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def _data_profile(docs: list[ValidatedDocument]) -> dict:
        if not docs:
            return {}
        wc = [d.word_count for d in docs]
        return {
            "total_words": sum(wc),
            "total_chars": sum(d.char_count for d in docs),
            "avg_word_count": round(sum(wc) / len(wc), 1),
            "min_word_count": min(wc),
            "max_word_count": max(wc),
            "source_types": list({d.source_type.value for d in docs}),
        }

    @staticmethod
    def _safe(text: str) -> str:
        """Sanitize text for Windows cp1252 terminal - replace non-ASCII with '?'."""
        return text.encode("ascii", errors="replace").decode("ascii")

    def _samples(self, docs: list[ValidatedDocument], n: int = 5) -> list[dict]:
        top = sorted(docs, key=lambda d: d.quality_score, reverse=True)[:n]
        return [
            {
                "id": d.id,
                "title": self._safe(d.title),
                "source": d.source,
                "source_type": d.source_type.value,
                "quality_score": d.quality_score,
                "word_count": d.word_count,
                "url": d.url,
                "preview": self._safe(truncate_text(d.content, 350)),
            }
            for d in top
        ]

    @staticmethod
    def _recommendations(
        validated: list[ValidatedDocument],
        rejected: list[RejectedDocument],
    ) -> list[str]:
        recs: list[str] = []
        total = len(validated) + len(rejected)
        pass_rate = len(validated) / max(total, 1)

        if pass_rate < 0.50:
            recs.append(
                "Low pass rate (<50%). Consider relaxing quality thresholds "
                "or adding more reliable data sources."
            )
        if len(validated) < 20:
            recs.append(
                "Small corpus (<20 docs). Expand ArXiv queries or add more RSS feeds."
            )
        if validated:
            avg = sum(d.quality_score for d in validated) / len(validated)
            if avg < 0.60:
                recs.append(
                    f"Average quality score is {avg:.2f} (< 0.60). "
                    "Review data sources for richer content."
                )

        if not recs:
            recs.append("Data quality looks solid. Ready to proceed to Phase II.")

        recs.append(
            "Inspect data/validated/ manually before chunking & embedding."
        )
        recs.append(
            "Consider the embedding model based on document domain and length distribution."
        )
        return recs
