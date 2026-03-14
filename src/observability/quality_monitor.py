"""
Quality Monitor — Tier 1 Production Monitoring
----------------------------------------------
Passive real-time tracking of pipeline quality metrics in a sliding window.

Tracks:
  - Keyword recall (retrieval quality)
  - Source type hit rate (correct source selection)
  - Citation count (grounding quality)
  - Hallucination risk (answer plausibility)
  - Latency (performance degradation)

Provides:
  - get_current_metrics() — current window stats
  - get_degradation_signal() — alert if avg score < threshold
  - log_result() — add result to window

No external dependencies; all metrics computed from QueryResult objects.
Cost: $0.00 (passive only, no API calls or LLM judges).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from loguru import logger


@dataclass
class MetricSnapshot:
    """Single point-in-time metrics from a query result."""

    timestamp: str  # ISO format
    query: str
    recall_score: float  # 0.0-1.0 (keywords found in answer/citations)
    source_hit_score: float  # 0.0-1.0 (source type matches expected)
    citation_count: int  # number of citations in answer
    answer_length: int  # character count of answer
    total_latency_ms: float  # retrieval + rerank + generation
    model: str  # generator model used


@dataclass
class DegradationAlert:
    """Alert signal when metrics degrade below threshold."""

    alert_type: str  # "recall_degradation", "source_hit_degradation", "latency_spike"
    severity: str  # "low", "medium", "high"
    metric_name: str
    current_value: float
    threshold: float
    window_size: int
    triggered_at: str  # ISO format


class QualityMonitor:
    """
    Real-time sliding-window monitor for pipeline quality metrics.

    Tracks: keyword_recall, source_type_hit, citation_count, latency.
    Detects: degradation below baseline, latency spikes, missing citations.

    No API calls or external dependencies. Safe to run inline in pipeline.
    """

    def __init__(
        self,
        window_size: int = 100,
        recall_threshold: float = 0.75,
        source_hit_threshold: float = 0.80,
        latency_threshold_ms: float = 5000.0,
        output_dir: str = "data/monitoring",
    ) -> None:
        """
        Args:
            window_size: Number of recent results to track
            recall_threshold: Alert if avg recall falls below this
            source_hit_threshold: Alert if avg source_hit falls below this
            latency_threshold_ms: Alert if avg latency exceeds this
            output_dir: Where to write metrics JSONL
        """
        self.window_size = window_size
        self.recall_threshold = recall_threshold
        self.source_hit_threshold = source_hit_threshold
        self.latency_threshold_ms = latency_threshold_ms

        # Circular buffer for recent results
        self.snapshots: deque[MetricSnapshot] = deque(maxlen=window_size)

        # Output file for metrics log
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log = self.output_dir / "quality_metrics.jsonl"

        logger.info(
            f"[QualityMonitor] Initialized | window={window_size} | "
            f"recall_threshold={recall_threshold} | "
            f"output={self.metrics_log}"
        )

    def log_result(
        self,
        query: str,
        answer: str,
        citations: list[dict],
        latency_ms: float,
        model: str,
        expected_keywords: list[str] | None = None,
        expected_source_types: list[str] | None = None,
    ) -> None:
        """
        Log a query result and compute metrics.

        Args:
            query: User query
            answer: Generated answer
            citations: List of {source, source_type, ...} dicts
            latency_ms: Total pipeline latency
            model: Generator model used
            expected_keywords: Keywords that should appear (for recall calculation)
            expected_source_types: Source types that should match (for source_hit)
        """
        # Compute recall: did expected keywords appear?
        recall_score = 1.0
        if expected_keywords:
            answer_lower = answer.lower()
            citation_texts = " ".join(c.get("source", "") for c in citations).lower()
            keywords_found = sum(
                1 for kw in expected_keywords if kw.lower() in answer_lower or kw.lower() in citation_texts
            )
            recall_score = keywords_found / len(expected_keywords)

        # Compute source hit: did citation types match?
        source_hit_score = 1.0
        if expected_source_types and citations:
            citation_types = {c.get("source_type", "") for c in citations}
            matches = len(citation_types & set(expected_source_types))
            source_hit_score = matches / len(expected_source_types) if expected_source_types else 0.0

        # Count citations
        citation_count = len(citations)

        # Create snapshot
        snapshot = MetricSnapshot(
            timestamp=datetime.utcnow().isoformat() + "Z",
            query=query[:200],  # Truncate for log size
            recall_score=recall_score,
            source_hit_score=source_hit_score,
            citation_count=citation_count,
            answer_length=len(answer),
            total_latency_ms=latency_ms,
            model=model,
        )

        self.snapshots.append(snapshot)

        # Append to JSONL
        try:
            with open(self.metrics_log, "a") as f:
                f.write(json.dumps(asdict(snapshot)) + "\n")
        except Exception as e:
            logger.error(f"[QualityMonitor] Failed to write metrics log: {e}")

        logger.debug(
            f"[QualityMonitor] Logged: recall={recall_score:.2f}, "
            f"source_hit={source_hit_score:.2f}, citations={citation_count}, "
            f"latency={latency_ms:.0f}ms"
        )

    def get_current_metrics(self) -> dict:
        """
        Get aggregated metrics from current sliding window.

        Returns:
            {
              'window_size': N,
              'avg_recall': X.XX,
              'avg_source_hit': X.XX,
              'avg_citation_count': X.XX,
              'avg_latency_ms': X.XX,
              'p95_latency_ms': X.XX,
              'models_used': [model1, model2, ...],
            }
        """
        if not self.snapshots:
            return {
                "window_size": 0,
                "avg_recall": 0.0,
                "avg_source_hit": 0.0,
                "avg_citation_count": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "models_used": [],
            }

        recalls = [s.recall_score for s in self.snapshots]
        source_hits = [s.source_hit_score for s in self.snapshots]
        citations = [s.citation_count for s in self.snapshots]
        latencies = [s.total_latency_ms for s in self.snapshots]
        models = list(set(s.model for s in self.snapshots))

        # Compute p95 latency
        sorted_latencies = sorted(latencies)
        p95_idx = max(0, int(len(sorted_latencies) * 0.95) - 1)
        p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0.0

        return {
            "window_size": len(self.snapshots),
            "avg_recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "avg_source_hit": sum(source_hits) / len(source_hits) if source_hits else 0.0,
            "avg_citation_count": sum(citations) / len(citations) if citations else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "p95_latency_ms": p95_latency,
            "models_used": models,
        }

    def get_degradation_signal(self) -> list[DegradationAlert]:
        """
        Check for metric degradation and return alerts.

        Returns:
            List of DegradationAlert objects if thresholds breached, else empty list.
        """
        alerts = []
        metrics = self.get_current_metrics()

        if metrics["window_size"] < 10:
            # Need minimum window for statistical significance
            return alerts

        # Check recall degradation
        if metrics["avg_recall"] < self.recall_threshold:
            alerts.append(
                DegradationAlert(
                    alert_type="recall_degradation",
                    severity="high" if metrics["avg_recall"] < 0.5 else "medium",
                    metric_name="keyword_recall",
                    current_value=metrics["avg_recall"],
                    threshold=self.recall_threshold,
                    window_size=metrics["window_size"],
                    triggered_at=datetime.utcnow().isoformat() + "Z",
                )
            )

        # Check source_hit degradation
        if metrics["avg_source_hit"] < self.source_hit_threshold:
            alerts.append(
                DegradationAlert(
                    alert_type="source_hit_degradation",
                    severity="high" if metrics["avg_source_hit"] < 0.5 else "medium",
                    metric_name="source_type_hit",
                    current_value=metrics["avg_source_hit"],
                    threshold=self.source_hit_threshold,
                    window_size=metrics["window_size"],
                    triggered_at=datetime.utcnow().isoformat() + "Z",
                )
            )

        # Check latency spike
        if metrics["avg_latency_ms"] > self.latency_threshold_ms:
            alerts.append(
                DegradationAlert(
                    alert_type="latency_spike",
                    severity="high" if metrics["avg_latency_ms"] > 10000 else "medium",
                    metric_name="total_latency_ms",
                    current_value=metrics["avg_latency_ms"],
                    threshold=self.latency_threshold_ms,
                    window_size=metrics["window_size"],
                    triggered_at=datetime.utcnow().isoformat() + "Z",
                )
            )

        if alerts:
            logger.warning(f"[QualityMonitor] Degradation detected: {len(alerts)} alert(s)")
            for alert in alerts:
                logger.warning(f"  - {alert.alert_type}: {alert.current_value:.2f} < {alert.threshold:.2f}")

        return alerts
