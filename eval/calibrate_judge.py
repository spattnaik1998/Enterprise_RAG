"""
LLM-Judge Calibration Harness
--------------------------------
Measures how well the LLMJudge in eval/judge.py tracks human labels.

Metrics computed:
  - Binary precision, recall, F1 (binarized at threshold 0.8 for both axes)
  - Pearson r (continuous agreement between human and judge scores)
  - Mean absolute error (faithfulness and correctness separately)
  - Confusion matrix (TP, FP, TN, FN)
  - Per-category breakdown

Usage:
  python -m eval.calibrate_judge                        # run calibration
  python -m eval.calibrate_judge --update-baseline       # update committed baseline
  python -m eval.calibrate_judge --judge-model gpt-4o    # use a different judge
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

# Windows cp1252 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from eval.judge import LLMJudge

app = typer.Typer(add_completion=False)
console = Console()

LABELS_PATH = Path("eval/datasets/judge_labels.json")
BASELINE_PATH = Path("eval/results/judge_calibration_baseline.json")
THRESHOLD = 0.8   # binarize both human and judge scores at this value


def _binarize(score: float, threshold: float = THRESHOLD) -> int:
    return 1 if score >= threshold else 0


def _compute_metrics(labels: list[dict], judge_results: list[dict]) -> dict:
    """Compute calibration metrics from paired human/judge scores."""
    n = len(labels)
    assert n == len(judge_results)

    # Continuous scores
    human_faith = [e["human_faithfulness"] for e in labels]
    human_corr  = [e["human_correctness"] for e in labels]
    judge_faith = [r["faithfulness"] for r in judge_results]
    judge_corr  = [r["correctness"] for r in judge_results]

    # Mean absolute error
    mae_faith = sum(abs(h - j) for h, j in zip(human_faith, judge_faith)) / n
    mae_corr  = sum(abs(h - j) for h, j in zip(human_corr,  judge_corr))  / n

    # Pearson r (faithfulness)
    def pearson(xs, ys):
        n_ = len(xs)
        mx, my = sum(xs)/n_, sum(ys)/n_
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx)**2 for x in xs) * sum((y - my)**2 for y in ys)) ** 0.5
        return num / den if den > 0 else 0.0

    pearson_faith = pearson(human_faith, judge_faith)
    pearson_corr  = pearson(human_corr,  judge_corr)

    # Binary classification: "accept" = both faith and corr >= threshold
    tp = fp = tn = fn = 0
    for lbl, jr in zip(labels, judge_results):
        h_accept = _binarize(lbl["human_faithfulness"]) and _binarize(lbl["human_correctness"])
        j_accept = _binarize(jr["faithfulness"]) and _binarize(jr["correctness"])
        if h_accept and j_accept:
            tp += 1
        elif not h_accept and j_accept:
            fp += 1
        elif h_accept and not j_accept:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n": n,
        "mae_faithfulness": round(mae_faith, 4),
        "mae_correctness":  round(mae_corr,  4),
        "pearson_r_faithfulness": round(pearson_faith, 4),
        "pearson_r_correctness":  round(pearson_corr,  4),
        "binary_threshold": THRESHOLD,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _compute_per_category(labels: list[dict], judge_results: list[dict]) -> dict:
    """Per-category F1 breakdown."""
    cats: dict[str, tuple[list, list]] = {}
    for lbl, jr in zip(labels, judge_results):
        cat = lbl.get("category", "unknown")
        if cat not in cats:
            cats[cat] = ([], [])
        cats[cat][0].append(lbl)
        cats[cat][1].append(jr)

    out = {}
    for cat, (lbls, jrs) in cats.items():
        m = _compute_metrics(lbls, jrs)
        out[cat] = {"n": m["n"], "f1": m["f1"], "mae_faithfulness": m["mae_faithfulness"]}
    return out


def _print_report(metrics: dict, per_category: dict, model: str) -> None:
    console.print(f"\n[bold cyan]LLM-Judge Calibration Report[/bold cyan]  model=[yellow]{model}[/yellow]  n={metrics['n']}")

    t = Table(title="Overall Metrics", show_header=True, header_style="bold blue")
    t.add_column("Metric", style="dim")
    t.add_column("Value", justify="right")
    rows = [
        ("F1",                 f"{metrics['f1']:.4f}"),
        ("Precision",          f"{metrics['precision']:.4f}"),
        ("Recall",             f"{metrics['recall']:.4f}"),
        ("MAE Faithfulness",   f"{metrics['mae_faithfulness']:.4f}"),
        ("MAE Correctness",    f"{metrics['mae_correctness']:.4f}"),
        ("Pearson r (faith)",  f"{metrics['pearson_r_faithfulness']:.4f}"),
        ("Pearson r (corr)",   f"{metrics['pearson_r_correctness']:.4f}"),
    ]
    cm = metrics["confusion_matrix"]
    rows += [
        ("TP / FP / TN / FN",  f"{cm['tp']} / {cm['fp']} / {cm['tn']} / {cm['fn']}"),
    ]
    for label, value in rows:
        t.add_row(label, value)
    console.print(t)

    t2 = Table(title="Per-Category F1", show_header=True, header_style="bold blue")
    t2.add_column("Category", style="dim")
    t2.add_column("n", justify="right")
    t2.add_column("F1", justify="right")
    t2.add_column("MAE Faith", justify="right")
    for cat, m in sorted(per_category.items()):
        t2.add_row(cat, str(m["n"]), f"{m['f1']:.4f}", f"{m['mae_faithfulness']:.4f}")
    console.print(t2)


@app.command()
def main(
    judge_model:      str  = typer.Option("gpt-4o-mini", help="OpenAI model to use as judge"),
    update_baseline:  bool = typer.Option(False, "--update-baseline", help="Save results as new baseline"),
    labels_path:      str  = typer.Option(str(LABELS_PATH), "--labels", help="Path to judge_labels.json"),
    quiet:            bool = typer.Option(False, help="Suppress rich output"),
) -> None:
    """Run LLM-judge calibration against human-labelled dataset."""
    labels_file = Path(labels_path)
    if not labels_file.exists():
        console.print(f"[red]Labels file not found: {labels_file}[/red]")
        raise typer.Exit(1)

    with open(labels_file, encoding="utf-8") as f:
        labels = json.load(f)

    console.print(f"[green]Loaded {len(labels)} human-labelled examples[/green]")

    judge = LLMJudge(model=judge_model)
    judge_results = []

    with console.status(f"[yellow]Running judge ({len(labels)} calls)...[/yellow]"):
        for entry in labels:
            result = judge.score(
                query=entry["query"],
                answer=entry["answer"],
                ground_truth=entry["ground_truth"],
                context_str=entry["context_str"],
            )
            judge_results.append({
                "id": entry["id"],
                "faithfulness": result.faithfulness,
                "correctness":  result.correctness,
                "cost_usd":     result.cost_usd,
            })

    total_cost = sum(r["cost_usd"] for r in judge_results)
    console.print(f"[green]Judge calls complete | total_cost=${total_cost:.4f}[/green]")

    metrics = _compute_metrics(labels, judge_results)
    per_category = _compute_per_category(labels, judge_results)

    if not quiet:
        _print_report(metrics, per_category, judge_model)

    # Drift detection vs baseline
    exit_code = 0
    if BASELINE_PATH.exists() and not update_baseline:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            baseline = json.load(f)
        baseline_f1 = baseline.get("f1", 0.0)
        if metrics["f1"] < baseline_f1 - 0.05:
            console.print(
                f"[red]DRIFT DETECTED: F1 dropped from {baseline_f1:.4f} to {metrics['f1']:.4f} "
                f"(threshold: -0.05)[/red]"
            )
            exit_code = 1
        else:
            console.print(
                f"[green]Calibration OK: F1={metrics['f1']:.4f} baseline={baseline_f1:.4f}[/green]"
            )

    # Save results
    output = {
        "model": judge_model,
        "n": metrics["n"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mae_faithfulness": metrics["mae_faithfulness"],
        "mae_correctness": metrics["mae_correctness"],
        "pearson_r_faithfulness": metrics["pearson_r_faithfulness"],
        "pearson_r_correctness": metrics["pearson_r_correctness"],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_category": per_category,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    results_dir = Path("eval/results")
    results_dir.mkdir(exist_ok=True)

    if update_baseline:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        console.print(f"[green]Baseline updated: {BASELINE_PATH}[/green]")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"judge_calibration_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        console.print(f"[dim]Results saved to {out_path}[/dim]")

    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
