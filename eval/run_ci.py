"""
Gold-Task CI Runner — Sprint 1, Feature 2
------------------------------------------
Lightweight regression gate that runs 20 curated gold tasks against
gpt-4o-mini and compares composite score to a committed baseline.

Usage:
    python -m eval.run_ci                     # Normal CI run (exits 0/1)
    python -m eval.run_ci --update-baseline   # Overwrite ci_baseline.json
    python -m eval.run_ci --tolerance 0.05    # Widen regression window

Exit codes:
    0  All metrics within tolerance of baseline (or no baseline yet)
    1  Regression detected OR any metric below absolute threshold
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Windows cp1252 terminal fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import typer
from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EVAL_DIR      = Path(__file__).parent
_DATASETS_DIR  = _EVAL_DIR / "datasets"
_RESULTS_DIR   = _EVAL_DIR / "results"
_GOLD_FILE     = _DATASETS_DIR / "gold_tasks.json"
_BASELINE_FILE = _RESULTS_DIR  / "ci_baseline.json"

# ---------------------------------------------------------------------------
# Absolute thresholds (same as main eval, but applied to gold-task subset)
# ---------------------------------------------------------------------------

_THRESHOLDS = {
    "recall_at_10":   0.75,   # slightly relaxed for 20-query subset
    "source_type_hit": 0.80,
    "faithfulness":   0.82,
    "correctness":    0.72,
    "composite":      0.78,
}

# Regression tolerance: fail CI if metric drops more than this vs baseline
_DEFAULT_TOLERANCE = 0.03


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gold_tasks() -> list[dict]:
    if not _GOLD_FILE.exists():
        raise FileNotFoundError(f"Gold tasks file not found: {_GOLD_FILE}")
    with open(_GOLD_FILE, encoding="utf-8") as f:
        return json.load(f)["queries"]


def _load_baseline() -> Optional[dict]:
    if not _BASELINE_FILE.exists():
        return None
    with open(_BASELINE_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save_baseline(metrics: dict, git_sha: str = "") -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        **metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "n_queries": 20,
        "model": "gpt-4o-mini",
    }
    with open(_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"[CI] Baseline updated -> {_BASELINE_FILE}")


def _get_git_sha() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_gold_eval() -> dict:
    """
    Load gold tasks, run them through RAGEvaluator with gpt-4o-mini,
    and return a flat metrics dict.
    """
    from eval.evaluator import RAGEvaluator

    tasks = _load_gold_tasks()

    # Build a synthetic dataset dict keyed by category
    # so we can feed into RAGEvaluator.run_single_query directly
    evaluator = RAGEvaluator(
        top_k=20,
        rerank_top_k=10,
        enable_pii_filter=False,
        judge_model="gpt-4o-mini",
    )
    evaluator._load_pipeline(enable_reranking=True)

    from src.generation.generator import RAGGenerator
    generator = RAGGenerator(model="gpt-4o-mini")

    all_results = []
    total = len(tasks)
    for i, task in enumerate(tasks, 1):
        logger.info(f"[CI] [{i:02d}/{total}] {task['id']} | {task['query'][:60]!r}")
        result = evaluator.run_single_query(
            query_item=task,
            model="gpt-4o-mini",
            category=task.get("category", "unknown"),
            generator=generator,
        )
        all_results.append(result)

    n = len(all_results)
    if n == 0:
        raise RuntimeError("No gold tasks evaluated.")

    recall       = sum(1 for r in all_results if r.recall_hit) / n
    source_hit   = sum(1 for r in all_results if r.source_type_hit) / n
    faithfulness = sum(r.faithfulness for r in all_results) / n
    correctness  = sum(r.correctness  for r in all_results) / n
    composite    = (recall + source_hit + faithfulness + correctness) / 4
    total_cost   = sum(r.total_cost_usd for r in all_results)
    n_blocked    = sum(1 for r in all_results if r.blocked)

    return {
        "recall_at_10":    round(recall,       4),
        "source_type_hit": round(source_hit,   4),
        "faithfulness":    round(faithfulness, 4),
        "correctness":     round(correctness,  4),
        "composite":       round(composite,    4),
        "total_cost_usd":  round(total_cost,   4),
        "n_blocked":       n_blocked,
    }


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------

def check_regression(current: dict, baseline: dict, tolerance: float) -> list[str]:
    """
    Return a list of failure messages. Empty list = pass.
    """
    failures: list[str] = []
    metrics = ["recall_at_10", "source_type_hit", "faithfulness", "correctness", "composite"]

    for metric in metrics:
        cur_val   = current.get(metric, 0.0)
        base_val  = baseline.get(metric, 0.0)
        threshold = _THRESHOLDS.get(metric, 0.0)

        # Absolute threshold check
        if cur_val < threshold:
            failures.append(
                f"FAIL  {metric}: {cur_val:.1%} < absolute threshold {threshold:.1%}"
            )

        # Regression vs baseline
        elif base_val - cur_val > tolerance:
            failures.append(
                f"REGR  {metric}: {cur_val:.1%} (baseline {base_val:.1%}, "
                f"dropped {base_val - cur_val:.1%} > tolerance {tolerance:.1%})"
            )

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    update_baseline: bool = typer.Option(
        False, "--update-baseline",
        help="Run eval and overwrite ci_baseline.json with current scores."
    ),
    tolerance: float = typer.Option(
        _DEFAULT_TOLERANCE, "--tolerance",
        help="Regression tolerance (default 0.03 = 3 points)."
    ),
    quiet: bool = typer.Option(
        False, "--quiet",
        help="Suppress per-query progress output."
    ),
) -> None:
    """
    Run the 20-query gold-task CI gate.

    Exits 0 if all metrics are within tolerance of the baseline.
    Exits 1 if a regression is detected or any absolute threshold is breached.
    """
    if quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    logger.info("[CI] Starting gold-task CI runner (20 queries, gpt-4o-mini)")
    logger.info(f"[CI] Tolerance: {tolerance:.1%}  |  Update baseline: {update_baseline}")

    # Run evaluation
    try:
        metrics = run_gold_eval()
    except Exception as exc:
        logger.error(f"[CI] Evaluation failed: {exc}")
        raise typer.Exit(code=1)

    # Print results
    print("\n" + "="*60)
    print("  GOLD-TASK CI RESULTS  (gpt-4o-mini, 20 gold tasks)")
    print("="*60)
    for key in ["recall_at_10", "source_type_hit", "faithfulness", "correctness", "composite"]:
        val = metrics[key]
        thresh = _THRESHOLDS[key]
        status = "PASS" if val >= thresh else "FAIL"
        print(f"  {key:<20} {val:.1%}  (threshold >= {thresh:.1%})  [{status}]")
    print(f"\n  Cost: ${metrics['total_cost_usd']:.4f}  |  Blocked: {metrics['n_blocked']}")
    print("="*60 + "\n")

    if update_baseline:
        _save_baseline(metrics, git_sha=_get_git_sha())
        print("[CI] Baseline updated successfully.")
        raise typer.Exit(code=0)

    # Load and compare baseline
    baseline = _load_baseline()
    if baseline is None:
        print("[CI] No baseline found. Run with --update-baseline to create one.")
        print("[CI] Checking absolute thresholds only...")
        failures = []
        for metric, threshold in _THRESHOLDS.items():
            val = metrics.get(metric, 0.0)
            if val < threshold:
                failures.append(
                    f"FAIL  {metric}: {val:.1%} < absolute threshold {threshold:.1%}"
                )
    else:
        print(f"[CI] Comparing against baseline (git:{baseline.get('git_sha', 'unknown')} "
              f"created:{baseline.get('created_at', 'unknown')[:10]})")
        failures = check_regression(metrics, baseline, tolerance)

    if failures:
        print("\n[CI] REGRESSION DETECTED:")
        for msg in failures:
            print(f"  {msg}")
        print("\n[CI] EXIT 1 - Pipeline did not meet quality gate.\n")
        raise typer.Exit(code=1)
    else:
        print("[CI] All metrics within tolerance of baseline.")
        print("[CI] EXIT 0 - Quality gate PASSED.\n")
        raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
