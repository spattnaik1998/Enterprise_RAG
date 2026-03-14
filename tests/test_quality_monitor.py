"""
Unit tests for QualityMonitor (Tier 1)
"""

import tempfile
from pathlib import Path

import pytest

from src.observability.quality_monitor import QualityMonitor, MetricSnapshot


def test_quality_monitor_init():
    """Test QualityMonitor initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = QualityMonitor(
            window_size=100,
            recall_threshold=0.75,
            output_dir=tmpdir,
        )

        assert monitor.window_size == 100
        assert monitor.recall_threshold == 0.75
        assert monitor.output_dir == Path(tmpdir)
        assert len(monitor.snapshots) == 0


def test_quality_monitor_log_result():
    """Test logging a query result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = QualityMonitor(window_size=10, output_dir=tmpdir)

        # Log a result
        monitor.log_result(
            query="Which clients are overdue?",
            answer="Northern Lights and Crossroads have overdue invoices.",
            citations=[
                {"source": "invoice_001", "source_type": "billing"},
                {"source": "invoice_002", "source_type": "billing"},
            ],
            latency_ms=3500.0,
            model="gpt-4o-mini",
            expected_keywords=["Northern Lights", "Crossroads"],
            expected_source_types=["billing"],
        )

        # Check window
        assert len(monitor.snapshots) == 1
        snapshot = list(monitor.snapshots)[0]
        assert snapshot.recall_score == 1.0  # Both keywords found
        assert snapshot.source_hit_score == 1.0  # Billing source matched
        assert snapshot.citation_count == 2
        assert snapshot.total_latency_ms == 3500.0

        # Check JSONL was written
        metrics_log = Path(tmpdir) / "quality_metrics.jsonl"
        assert metrics_log.exists()
        with open(metrics_log) as f:
            lines = f.readlines()
            assert len(lines) == 1


def test_quality_monitor_recall_calculation():
    """Test keyword recall calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = QualityMonitor(window_size=10, output_dir=tmpdir)

        # Case 1: Both keywords found in answer
        monitor.log_result(
            query="Test",
            answer="Northern Lights and Crossroads",
            citations=[],
            latency_ms=1000.0,
            model="gpt-4o-mini",
            expected_keywords=["Northern Lights", "Crossroads"],
            expected_source_types=[],
        )
        assert list(monitor.snapshots)[-1].recall_score == 1.0

        # Case 2: One keyword found
        monitor.log_result(
            query="Test",
            answer="Northern Lights",
            citations=[],
            latency_ms=1000.0,
            model="gpt-4o-mini",
            expected_keywords=["Northern Lights", "Crossroads"],
            expected_source_types=[],
        )
        assert list(monitor.snapshots)[-1].recall_score == 0.5

        # Case 3: No keywords found
        monitor.log_result(
            query="Test",
            answer="Some other company",
            citations=[],
            latency_ms=1000.0,
            model="gpt-4o-mini",
            expected_keywords=["Northern Lights", "Crossroads"],
            expected_source_types=[],
        )
        assert list(monitor.snapshots)[-1].recall_score == 0.0


def test_quality_monitor_get_current_metrics():
    """Test metrics aggregation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = QualityMonitor(window_size=10, output_dir=tmpdir)

        # Log 5 results with varying scores
        for i in range(5):
            monitor.log_result(
                query=f"Query {i}",
                answer=f"Answer {i}",
                citations=[{"source": "src", "source_type": "billing"}],
                latency_ms=1000.0 + i * 100,
                model="gpt-4o-mini",
                expected_keywords=["kw1", "kw2"],
                expected_source_types=["billing"],
            )

        metrics = monitor.get_current_metrics()

        assert metrics["window_size"] == 5
        assert "avg_recall" in metrics
        assert "avg_source_hit" in metrics
        assert "avg_latency_ms" in metrics
        assert metrics["avg_latency_ms"] >= 1000.0


def test_quality_monitor_degradation_detection():
    """Test degradation alert detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = QualityMonitor(
            window_size=10,
            recall_threshold=0.8,
            source_hit_threshold=0.9,
            output_dir=tmpdir,
        )

        # Log results with low recall
        for _ in range(15):
            monitor.log_result(
                query="Test",
                answer="Wrong answer",
                citations=[],
                latency_ms=1000.0,
                model="gpt-4o-mini",
                expected_keywords=["expected1", "expected2"],
                expected_source_types=["billing"],
            )

        # Get degradation signals
        alerts = monitor.get_degradation_signal()

        assert len(alerts) > 0
        alert_types = {a.alert_type for a in alerts}
        assert "recall_degradation" in alert_types


def test_quality_monitor_window_rotation():
    """Test that window maintains max size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = QualityMonitor(window_size=5, output_dir=tmpdir)

        # Log 10 results
        for i in range(10):
            monitor.log_result(
                query=f"Query {i}",
                answer=f"Answer {i}",
                citations=[],
                latency_ms=1000.0,
                model="gpt-4o-mini",
            )

        # Window should max out at 5
        assert len(monitor.snapshots) == 5
        # Oldest results should be dropped
        snapshot_queries = [s.query for s in monitor.snapshots]
        assert "Query 0" not in snapshot_queries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
