"""
DSPy Metric Functions for RAG Optimization
---------------------------------------------
Four metric functions matching eval/evaluator.py logic:
  - keyword_recall_metric: cheap, no API calls (for BootstrapFewShot demo selection)
  - source_type_metric: checks citation source_type matches
  - faithfulness_metric: LLM judge, measures grounding in context
  - correctness_metric: LLM judge, measures answer accuracy vs ground truth
  - rag_composite_metric: weighted blend (primary for optimization)
  - cheap_metric: alias for keyword_recall (used with --cheap flag)

Scoring: All return float in [0.0, 1.0]. Composites average component metrics.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import dspy
from loguru import logger

from eval.judge import LLMJudge, JudgeResult


# ─────────────────────────────────────────────────────────────────────────────
# Cheap Metrics (no API calls)
# ─────────────────────────────────────────────────────────────────────────────


def keyword_recall_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Check if any expected keyword appears in answer or source_ids.

    No API calls — safe to use with BootstrapFewShot demo selection.

    Args:
        example: dspy.Example with expected_keywords
        pred: dspy.Prediction with answer + citations
        trace: Unused (DSPy hook)

    Returns:
        1.0 if any keyword found, 0.0 otherwise
    """
    keywords = example.expected_keywords or []
    if not keywords:
        return 1.0  # Skip if no keywords defined

    answer = pred.answer.lower() if hasattr(pred, "answer") else ""
    citations = pred.citations if hasattr(pred, "citations") else []

    # Check if any keyword appears in answer
    for keyword in keywords:
        if keyword.lower() in answer:
            return 1.0

    # Check if any keyword appears in citation source_ids
    citation_sources = " ".join(
        [str(c.get("source_id", "")).lower() for c in citations]
    )
    for keyword in keywords:
        if keyword.lower() in citation_sources:
            return 1.0

    return 0.0


def source_type_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Check if any citation source_type matches expected_source_types.

    No API calls.

    Args:
        example: dspy.Example with expected_source_types
        pred: dspy.Prediction with citations
        trace: Unused

    Returns:
        1.0 if match found, 0.0 otherwise
    """
    expected = set(example.expected_source_types or [])
    if not expected:
        return 1.0

    citations = pred.citations if hasattr(pred, "citations") else []
    citation_types = {c.get("source_type", "").lower() for c in citations}

    expected_lower = {s.lower() for s in expected}

    if citation_types & expected_lower:  # Intersection
        return 1.0

    return 0.0


def cheap_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Alias for keyword_recall_metric.

    Used when --cheap flag is passed to skip LLM judge calls.

    Args:
        example: dspy.Example
        pred: dspy.Prediction
        trace: Unused

    Returns:
        Same as keyword_recall_metric
    """
    return keyword_recall_metric(example, pred, trace)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Judge Metrics (API calls)
# ─────────────────────────────────────────────────────────────────────────────


_judge_cache = {}


def _get_judge(model: str = "gpt-4o-mini") -> LLMJudge:
    """Get or create LLMJudge instance (cached)."""
    if model not in _judge_cache:
        _judge_cache[model] = LLMJudge(model=model)
    return _judge_cache[model]


def faithfulness_metric(
    example: dspy.Example, pred: dspy.Prediction, trace=None, model: str = "gpt-4o-mini"
) -> float:
    """
    LLM judge: are all answer claims grounded in retrieved context?

    Requires API call (gpt-4o-mini).

    Args:
        example: dspy.Example with answer, expected_source_types
        pred: dspy.Prediction with answer, context
        trace: Unused
        model: Judge model (default gpt-4o-mini)

    Returns:
        Faithfulness score [0.0, 1.0] from LLM judge
    """
    answer = pred.answer if hasattr(pred, "answer") else ""
    context = pred.context if hasattr(pred, "context") else ""
    ground_truth = example.answer if hasattr(example, "answer") else ""
    query = example.query if hasattr(example, "query") else ""

    if not answer or not context:
        return 0.0

    try:
        judge = _get_judge(model)
        result: JudgeResult = judge.score(
            query=query,
            answer=answer,
            ground_truth=ground_truth,
            context_str=context,
        )
        return result.faithfulness
    except Exception as e:
        logger.error(f"Faithfulness judge error: {e}")
        return 0.5  # Neutral on error


def correctness_metric(
    example: dspy.Example, pred: dspy.Prediction, trace=None, model: str = "gpt-4o-mini"
) -> float:
    """
    LLM judge: does answer correctly address the question?

    Requires API call (gpt-4o-mini).

    Args:
        example: dspy.Example with ground_truth answer
        pred: dspy.Prediction with answer
        trace: Unused
        model: Judge model (default gpt-4o-mini)

    Returns:
        Correctness score [0.0, 1.0] from LLM judge
    """
    answer = pred.answer if hasattr(pred, "answer") else ""
    context = pred.context if hasattr(pred, "context") else ""
    ground_truth = example.answer if hasattr(example, "answer") else ""
    query = example.query if hasattr(example, "query") else ""

    if not answer:
        return 0.0

    try:
        judge = _get_judge(model)
        result: JudgeResult = judge.score(
            query=query,
            answer=answer,
            ground_truth=ground_truth,
            context_str=context,
        )
        return result.correctness
    except Exception as e:
        logger.error(f"Correctness judge error: {e}")
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Composite Metric (Primary optimization target)
# ─────────────────────────────────────────────────────────────────────────────


def rag_composite_metric(
    example: dspy.Example, pred: dspy.Prediction, trace=None, model: str = "gpt-4o-mini"
) -> float:
    """
    Primary optimization metric: weighted blend of 4 component metrics.

    Weights: 40% recall + 30% faithfulness + 30% correctness
    (Source type hit integrated into recall)

    Requires API calls (LLM judge for faithfulness + correctness).

    Args:
        example: dspy.Example
        pred: dspy.Prediction
        trace: Unused
        model: Judge model

    Returns:
        Composite score [0.0, 1.0]
    """
    # Cheap metrics (no API)
    recall = keyword_recall_metric(example, pred, trace)
    source = source_type_metric(example, pred, trace)
    recall_combined = (recall + source) / 2  # Blend recall + source

    # Expensive metrics (API)
    faith = faithfulness_metric(example, pred, trace, model=model)
    correct = correctness_metric(example, pred, trace, model=model)

    # Composite: 40% recall, 30% faith, 30% correct
    return 0.4 * recall_combined + 0.3 * faith + 0.3 * correct
