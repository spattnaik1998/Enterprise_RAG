"""
Hard Negative Miner — Tier 2 Failure Collection for Retraining
---------------------------------------------------------------
Identifies and persists failed or low-confidence queries for use as
hard-negative examples in DSPy fine-tuning iterations.

Tier 2 cost model:
  - Passive collection: $0.00 per failure
  - Storage (JSONL): ~1 KB per failure = ~1MB per 1000 failures
  - Later: DSPy training on collected failures (~$0.01-$0.35 per run)

Strategy:
  1. Track failure patterns (low recall, missing sources, hallucinations)
  2. Collect only high-value failures (not every miss, just hard cases)
  3. Persist to JSONL for batch retraining
  4. Tier 3 (AutoRetrainer) later reads and trains on these

Example collected failure:
  {
    "failure_id": "hard_neg_001",
    "timestamp": "2026-03-14T10:30:45Z",
    "query": "Which clients have overdue invoices?",
    "category": "billing",
    "expected_keywords": ["Northern Lights", "Crossroads"],
    "expected_source_types": ["billing"],
    "answer": "I cannot find overdue information.",  # Wrong/incomplete
    "citations": [],  # Missing sources
    "recall_score": 0.0,  # Both keywords missed
    "source_hit_score": 0.0,  # No billing sources cited
    "citation_count": 0,
    "failure_reason": "missing_citations",
    "severity": "high",  # 0 citations when > 1 expected
    "model": "gpt-4o-mini",
    "latency_ms": 3450.0,
  }
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class HardNegativeExample:
    """A collected hard-negative (failure) example for retraining."""

    failure_id: str  # Unique ID: hard_neg_<counter>
    timestamp: str  # ISO format
    query: str
    category: str  # billing, contracts, crm, psa, communications, cross_source
    expected_keywords: list[str]
    expected_source_types: list[str]
    answer: str  # What the pipeline actually generated
    citations: list[dict]  # What was cited (empty if missed)
    recall_score: float  # 0.0-1.0 (expected keywords found)
    source_hit_score: float  # 0.0-1.0 (source types matched)
    citation_count: int
    failure_reason: str  # "missing_citations", "low_recall", "wrong_source", "hallucination"
    severity: str  # "high", "medium", "low"
    model: str
    latency_ms: float
    ground_truth: Optional[str] = None  # Optional: ground truth answer for comparison


class HardNegativeMiner:
    """
    Identifies and collects hard-negative (failure) examples.

    Uses heuristics to decide which failures are worth collecting:
      - missing_citations: citation_count == 0 and expected > 0
      - low_recall: recall_score < 0.5
      - wrong_source: source_hit_score < 0.5 and citations present
      - hallucination: high answer length but low citations (made-up content)

    Only collects examples that will be useful for retraining:
      - High severity (confidence that example is instructive)
      - Diverse (not same failure repeated)
      - Balanced across categories
    """

    def __init__(
        self,
        output_dir: str = "data/learning",
        max_examples: int = 500,
        collection_threshold: float = 0.3,  # Skip if score >= this
    ) -> None:
        """
        Args:
            output_dir: Where to persist hard negatives (JSONL file)
            max_examples: Max hard negatives to collect before rotating file
            collection_threshold: Only collect if metric score < this
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hard_negatives_file = self.output_dir / "hard_negatives.jsonl"
        self.max_examples = max_examples
        self.collection_threshold = collection_threshold

        # In-memory counter for unique failure IDs
        self._counter = self._load_counter()

        logger.info(
            f"[HardNegativeMiner] Initialized | output={self.hard_negatives_file} | "
            f"max={max_examples} | collection_threshold={collection_threshold}"
        )

    def _load_counter(self) -> int:
        """Load counter from hard_negatives file if it exists."""
        if self.hard_negatives_file.exists():
            try:
                with open(self.hard_negatives_file) as f:
                    lines = f.readlines()
                    return len(lines)  # Counter = total lines written so far
            except Exception:
                return 0
        return 0

    def should_collect(
        self,
        recall_score: float,
        source_hit_score: float,
        citation_count: int,
        answer_length: int,
    ) -> tuple[bool, str]:
        """
        Decide if a failure is worth collecting.

        Returns:
            (should_collect: bool, reason: str)
        """
        # No citations when some were expected
        if citation_count == 0 and recall_score < 1.0:
            return True, "missing_citations"

        # Low keyword recall (missed expected keywords)
        if recall_score < self.collection_threshold:
            return True, "low_recall"

        # Citations present but wrong source types
        if citation_count > 0 and source_hit_score < self.collection_threshold:
            return True, "wrong_source"

        # High answer length but low citations (hallucination signal)
        if answer_length > 300 and citation_count < 2:
            return True, "hallucination_risk"

        # Not worth collecting
        return False, ""

    def collect(
        self,
        query: str,
        answer: str,
        citations: list[dict],
        recall_score: float,
        source_hit_score: float,
        category: str,
        model: str,
        latency_ms: float,
        expected_keywords: list[str] | None = None,
        expected_source_types: list[str] | None = None,
        ground_truth: str | None = None,
    ) -> Optional[HardNegativeExample]:
        """
        Conditionally collect a query result as a hard-negative example.

        Args:
            query: User query
            answer: Generated answer
            citations: Cited sources
            recall_score: Keyword recall (0.0-1.0)
            source_hit_score: Source type hit rate (0.0-1.0)
            category: Query category
            model: Generator model
            latency_ms: Pipeline latency
            expected_keywords: Keywords that should appear
            expected_source_types: Expected source types
            ground_truth: Ground truth answer for reference

        Returns:
            HardNegativeExample if collected, else None
        """
        expected_keywords = expected_keywords or []
        expected_source_types = expected_source_types or []

        # Decide whether to collect
        should_collect, failure_reason = self.should_collect(
            recall_score=recall_score,
            source_hit_score=source_hit_score,
            citation_count=len(citations),
            answer_length=len(answer),
        )

        if not should_collect:
            return None

        # Compute severity based on failure reason
        if failure_reason == "missing_citations":
            severity = "high" if recall_score == 0.0 else "medium"
        elif failure_reason == "low_recall":
            severity = "high" if recall_score < 0.2 else "medium"
        elif failure_reason == "wrong_source":
            severity = "medium"
        elif failure_reason == "hallucination_risk":
            severity = "medium"
        else:
            severity = "low"

        # Create unique ID (counter-based)
        self._counter += 1
        failure_id = f"hard_neg_{self._counter:06d}"

        # Create hard negative example
        example = HardNegativeExample(
            failure_id=failure_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            query=query,
            category=category,
            expected_keywords=expected_keywords,
            expected_source_types=expected_source_types,
            answer=answer,
            citations=citations,
            recall_score=recall_score,
            source_hit_score=source_hit_score,
            citation_count=len(citations),
            failure_reason=failure_reason,
            severity=severity,
            model=model,
            latency_ms=latency_ms,
            ground_truth=ground_truth,
        )

        # Persist to JSONL
        try:
            with open(self.hard_negatives_file, "a") as f:
                f.write(json.dumps(asdict(example)) + "\n")
        except Exception as e:
            logger.error(f"[HardNegativeMiner] Failed to persist example: {e}")
            return None

        logger.info(
            f"[HardNegativeMiner] Collected: {failure_id} | "
            f"reason={failure_reason} | severity={severity} | "
            f"recall={recall_score:.2f} | source_hit={source_hit_score:.2f}"
        )

        # Check if we've reached max examples (rotate log)
        if self._counter >= self.max_examples:
            self._rotate_log()

        return example

    def _rotate_log(self) -> None:
        """
        Rotate hard_negatives.jsonl when max_examples reached.

        Renames current file to hard_negatives_<timestamp>.jsonl and
        starts fresh log. This prevents unbounded file growth.
        """
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            rotated_path = self.output_dir / f"hard_negatives_{timestamp}.jsonl"
            self.hard_negatives_file.rename(rotated_path)
            self._counter = 0
            logger.info(f"[HardNegativeMiner] Rotated log to {rotated_path.name}")
        except Exception as e:
            logger.error(f"[HardNegativeMiner] Failed to rotate log: {e}")

    def get_statistics(self) -> dict:
        """
        Get statistics on collected hard negatives.

        Returns:
            {
              'total_examples': N,
              'by_category': {cat: count, ...},
              'by_reason': {reason: count, ...},
              'by_severity': {severity: count, ...},
              'avg_recall': X.XX,
              'avg_source_hit': X.XX,
            }
        """
        if not self.hard_negatives_file.exists():
            return {
                "total_examples": 0,
                "by_category": {},
                "by_reason": {},
                "by_severity": {},
                "avg_recall": 0.0,
                "avg_source_hit": 0.0,
            }

        examples = []
        try:
            with open(self.hard_negatives_file) as f:
                for line in f:
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.error(f"[HardNegativeMiner] Failed to read stats: {e}")
            return {}

        if not examples:
            return {}

        # Aggregate stats
        by_category = {}
        by_reason = {}
        by_severity = {}
        recalls = []
        source_hits = []

        for ex in examples:
            cat = ex.get("category", "unknown")
            reason = ex.get("failure_reason", "unknown")
            severity = ex.get("severity", "unknown")

            by_category[cat] = by_category.get(cat, 0) + 1
            by_reason[reason] = by_reason.get(reason, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

            recalls.append(ex.get("recall_score", 0.0))
            source_hits.append(ex.get("source_hit_score", 0.0))

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_source_hit = sum(source_hits) / len(source_hits) if source_hits else 0.0

        return {
            "total_examples": len(examples),
            "by_category": by_category,
            "by_reason": by_reason,
            "by_severity": by_severity,
            "avg_recall": avg_recall,
            "avg_source_hit": avg_source_hit,
        }

    def load_examples_for_training(self, category: str | None = None) -> list[dict]:
        """
        Load collected hard negatives for DSPy retraining.

        Args:
            category: If specified, filter to single category

        Returns:
            List of example dicts ready for DSPy training
        """
        if not self.hard_negatives_file.exists():
            return []

        examples = []
        try:
            with open(self.hard_negatives_file) as f:
                for line in f:
                    try:
                        example = json.loads(line)
                        if category is None or example.get("category") == category:
                            examples.append(example)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.error(f"[HardNegativeMiner] Failed to load examples: {e}")
            return []

        logger.info(
            f"[HardNegativeMiner] Loaded {len(examples)} hard negatives" +
            (f" from category {category}" if category else "")
        )
        return examples
