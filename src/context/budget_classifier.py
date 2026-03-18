"""
Dynamic Token Budget Classifier
---------------------------------
Classifies query complexity and returns an appropriate token budget.
Pure heuristic function -- no LLM call, <1ms latency.

Budget tiers:
  SIMPLE:   1024 tokens (single-entity lookups, factual questions)
  MODERATE: 2048 tokens (comparisons, multi-entity, filtered queries)
  COMPLEX:  4096 tokens (reasoning, cross-source, analysis, recommendations)
"""
from __future__ import annotations


# Keywords that indicate higher complexity
_COMPLEX_KEYWORDS = [
    "why", "should", "recommend", "compare", "analyze", "analyse",
    "risk", "escalate", "trend", "forecast", "strategy", "churn",
    "cross-reference", "correlate", "impact", "prioritize",
]

_MODERATE_KEYWORDS = [
    "all", "every", "each", "between", "versus", "top", "bottom",
    "filter", "group", "breakdown", "summary", "history", "timeline",
]

_SIMPLE_KEYWORDS = [
    "what", "who", "when", "where", "list", "how much", "how many",
    "show", "get", "find", "which",
]


def classify_budget(query: str) -> int:
    """
    Classify query complexity and return token budget.

    Args:
        query: The user's raw query text.

    Returns:
        Token budget: 1024 (simple), 2048 (moderate), or 4096 (complex).
    """
    q = query.lower()

    # Count keyword matches
    complex_hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in q)
    moderate_hits = sum(1 for kw in _MODERATE_KEYWORDS if kw in q)

    # Multi-source indicators
    source_mentions = sum(1 for s in ["billing", "invoice", "contract", "crm",
                                       "ticket", "psa", "email", "communication"]
                          if s in q)

    # Question length as a complexity signal
    word_count = len(q.split())

    # Classification logic
    if complex_hits >= 2 or (complex_hits >= 1 and source_mentions >= 2):
        return 4096
    if complex_hits >= 1 or moderate_hits >= 2 or source_mentions >= 3 or word_count > 25:
        return 2048
    return 1024
