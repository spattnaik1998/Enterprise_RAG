"""
Unit tests for HardNegativeMiner (Tier 2)
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.learning.hard_negative_miner import HardNegativeMiner, HardNegativeExample


def test_hard_negative_miner_init():
    """Test HardNegativeMiner initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir, max_examples=500)

        assert miner.max_examples == 500
        assert miner.output_dir == Path(tmpdir)
        assert miner.hard_negatives_file == Path(tmpdir) / "hard_negatives.jsonl"


def test_should_collect_missing_citations():
    """Test should_collect for missing citations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir)

        # Case: No citations when expected
        should_collect, reason = miner.should_collect(
            recall_score=0.0,
            source_hit_score=0.0,
            citation_count=0,
            answer_length=100,
        )
        assert should_collect is True
        assert reason == "missing_citations"


def test_should_collect_low_recall():
    """Test should_collect for low recall."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir, collection_threshold=0.3)

        # Case: Low recall
        should_collect, reason = miner.should_collect(
            recall_score=0.2,
            source_hit_score=1.0,
            citation_count=2,
            answer_length=100,
        )
        assert should_collect is True
        assert reason == "low_recall"


def test_should_collect_wrong_source():
    """Test should_collect for wrong source types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir, collection_threshold=0.3)

        # Case: Wrong sources
        should_collect, reason = miner.should_collect(
            recall_score=0.8,
            source_hit_score=0.1,
            citation_count=2,
            answer_length=100,
        )
        assert should_collect is True
        assert reason == "wrong_source"


def test_should_collect_hallucination_risk():
    """Test should_collect for hallucination risk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir)

        # Case: High answer length but only 1 citation (high answer-to-citation ratio)
        should_collect, reason = miner.should_collect(
            recall_score=1.0,  # Good recall but...
            source_hit_score=0.8,
            citation_count=1,  # Only 1 citation
            answer_length=500,  # Long answer (signal of hallucination)
        )
        assert should_collect is True
        assert reason == "hallucination_risk"


def test_should_not_collect_good_result():
    """Test that good results are not collected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir)

        # Case: Good result
        should_collect, reason = miner.should_collect(
            recall_score=1.0,
            source_hit_score=1.0,
            citation_count=3,
            answer_length=150,
        )
        assert should_collect is False


def test_collect_hard_negative():
    """Test collecting a hard negative example."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir)

        example = miner.collect(
            query="Which clients are overdue?",
            answer="I cannot find overdue information.",
            citations=[],
            recall_score=0.0,
            source_hit_score=0.0,
            category="billing",
            model="gpt-4o-mini",
            latency_ms=3000.0,
            expected_keywords=["Northern Lights", "Crossroads"],
            expected_source_types=["billing"],
            ground_truth="Northern Lights and Crossroads have overdue invoices.",
        )

        # Check example was created
        assert example is not None
        assert example.failure_id == "hard_neg_000001"
        assert example.failure_reason == "missing_citations"
        assert example.severity == "high"

        # Check JSONL was written
        jsonl_file = Path(tmpdir) / "hard_negatives.jsonl"
        assert jsonl_file.exists()
        with open(jsonl_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["failure_id"] == "hard_neg_000001"
            assert data["category"] == "billing"


def test_collect_multiple_examples():
    """Test collecting multiple examples with incrementing IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir)

        for i in range(3):
            example = miner.collect(
                query=f"Query {i}",
                answer="Wrong answer",
                citations=[],
                recall_score=0.0,
                source_hit_score=0.0,
                category="billing",
                model="gpt-4o-mini",
                latency_ms=1000.0,
            )
            assert example.failure_id == f"hard_neg_{i+1:06d}"

        # Check all examples in JSONL
        jsonl_file = Path(tmpdir) / "hard_negatives.jsonl"
        with open(jsonl_file) as f:
            lines = f.readlines()
            assert len(lines) == 3


def test_no_collect_on_good_result():
    """Test that good results are not collected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir)

        example = miner.collect(
            query="Test query",
            answer="Good answer with citations",
            citations=[{"source": "src1", "source_type": "billing"}],
            recall_score=1.0,
            source_hit_score=1.0,
            category="billing",
            model="gpt-4o-mini",
            latency_ms=1000.0,
        )

        # Should not be collected
        assert example is None

        # JSONL should be empty
        jsonl_file = Path(tmpdir) / "hard_negatives.jsonl"
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                assert len(f.readlines()) == 0


def test_get_statistics():
    """Test statistics aggregation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir)

        # Collect examples from multiple categories
        for cat in ["billing", "contracts", "billing"]:
            miner.collect(
                query=f"Query in {cat}",
                answer="Wrong",
                citations=[],
                recall_score=0.0,
                source_hit_score=0.0,
                category=cat,
                model="gpt-4o-mini",
                latency_ms=1000.0,
            )

        stats = miner.get_statistics()

        assert stats["total_examples"] == 3
        assert stats["by_category"]["billing"] == 2
        assert stats["by_category"]["contracts"] == 1
        assert "by_reason" in stats
        assert "by_severity" in stats


def test_load_examples_for_training():
    """Test loading examples for DSPy retraining."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir)

        # Collect examples
        for i in range(3):
            miner.collect(
                query=f"Query {i}",
                answer="Wrong",
                citations=[],
                recall_score=0.0,
                source_hit_score=0.0,
                category="billing",
                model="gpt-4o-mini",
                latency_ms=1000.0,
            )

        # Load all examples
        all_examples = miner.load_examples_for_training()
        assert len(all_examples) == 3

        # Load examples for specific category
        billing_examples = miner.load_examples_for_training(category="billing")
        assert len(billing_examples) == 3

        # Load examples for non-existent category
        crm_examples = miner.load_examples_for_training(category="crm")
        assert len(crm_examples) == 0


def test_severity_calculation():
    """Test severity is calculated correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = HardNegativeMiner(output_dir=tmpdir)

        # Case 1: Zero recall = high severity
        ex1 = miner.collect(
            query="Query",
            answer="Wrong",
            citations=[],
            recall_score=0.0,
            source_hit_score=0.0,
            category="billing",
            model="gpt-4o-mini",
            latency_ms=1000.0,
        )
        assert ex1.severity == "high"

        # Case 2: Low recall (0.2) = medium severity
        ex2 = miner.collect(
            query="Query",
            answer="Partial",
            citations=[],
            recall_score=0.2,
            source_hit_score=0.0,
            category="billing",
            model="gpt-4o-mini",
            latency_ms=1000.0,
        )
        assert ex2.severity == "medium"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
