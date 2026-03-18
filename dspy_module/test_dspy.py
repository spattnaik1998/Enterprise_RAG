#!/usr/bin/env python
"""
Smoke Test for DSPy Framework Implementation
---------------------------------------------
Verifies that all DSPy components are correctly structured and can be imported
without making API calls. Safe to run offline.

Usage:
    python dspy_module/test_dspy.py
"""

import sys
from pathlib import Path

# Windows UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def test_imports() -> bool:
    """Test that all modules can be imported (after dspy-ai is installed)."""
    print("\n[1] Testing imports...")
    try:
        # This will fail if dspy-ai not installed, but shows error clearly
        import dspy
        print("  ✓ dspy module found")

        from dspy_module.signatures import (
            RAGSignature,
            CreativeProposalSignature,
            ConservativeProposalSignature,
            PolicyVerdictSignature,
            RerankerSignature,
        )
        print("  ✓ Signatures imported")

        from dspy_module.modules import (
            DSPyRetrieverAdapter,
            DSPyRAGModule,
            DSPyRerankerModule,
            DSPyCouncilModule,
        )
        print("  ✓ Modules imported")

        from dspy_module.dataset import MSPDataset, DatasetSummary
        print("  ✓ Dataset loader imported")

        from dspy_module.metrics import (
            keyword_recall_metric,
            source_type_metric,
            faithfulness_metric,
            correctness_metric,
            rag_composite_metric,
            cheap_metric,
        )
        print("  ✓ Metrics imported")

        return True
    except ImportError as e:
        print(f"\n  ✗ Import failed: {e}")
        print("\n  Install dspy-ai:")
        print("    pip install dspy-ai>=2.5")
        return False


def test_dataset() -> bool:
    """Test dataset loading and split."""
    print("\n[2] Testing dataset loading...")
    try:
        from dspy_module.dataset import MSPDataset

        dataset = MSPDataset(seed=42)
        summary = dataset.summary()

        print(f"  ✓ Loaded {summary.total_queries} total queries")
        print(f"    - Train: {summary.total_train}")
        print(f"    - Dev:   {summary.total_dev}")
        print(f"    - Gold:  {summary.total_gold}")

        # Check split correctness
        assert summary.total_train == len(dataset.train), "Train size mismatch"
        assert summary.total_dev == len(dataset.dev), "Dev size mismatch"
        assert summary.total_gold == len(dataset.gold), "Gold size mismatch"

        # Check stratification
        for cat, stats in summary.split_by_category.items():
            total_cat = stats["train"] + stats["dev"] + stats["gold"]
            print(f"    - {cat:18} {total_cat:2} queries (train: {stats['train']}, dev: {stats['dev']}, gold: {stats['gold']})")

        return True
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        return False


def test_signatures() -> bool:
    """Test that signatures are properly defined."""
    print("\n[3] Testing signature definitions...")
    try:
        import dspy
        from dspy_module.signatures import RAGSignature, CreativeProposalSignature

        # Verify RAGSignature is a dspy.Signature
        assert issubclass(RAGSignature, dspy.Signature), "RAGSignature not a dspy.Signature"
        print("  ✓ RAGSignature is a valid dspy.Signature")

        # Check via annotations (DSPy fields are type annotations)
        annotations = RAGSignature.__annotations__
        assert "query" in annotations, "Missing 'query' field"
        assert "context" in annotations, "Missing 'context' field"
        assert "answer" in annotations, "Missing 'answer' field"
        print("  ✓ RAGSignature has correct fields (query, context, answer)")

        # Verify CreativeProposalSignature exists
        assert issubclass(CreativeProposalSignature, dspy.Signature)
        print("  ✓ CreativeProposalSignature is valid")

        return True
    except Exception as e:
        print(f"  ✗ Signature test failed: {e}")
        return False


def test_metrics() -> bool:
    """Test metric functions (no API calls)."""
    print("\n[4] Testing metric functions...")
    try:
        import dspy
        from dspy_module.metrics import keyword_recall_metric, source_type_metric

        # Create fake example and prediction
        example = dspy.Example(
            query="Which clients have overdue invoices?",
            expected_keywords=["Northern Lights", "Crossroads"],
            expected_source_types=["billing"],
        )

        pred = dspy.Prediction(
            answer="Northern Lights Healthcare has overdue invoices.",
            citations=[
                {"source_id": "invoice_001", "source_type": "billing"},
            ],
        )

        # Test keyword recall (should find "Northern Lights")
        recall_score = keyword_recall_metric(example, pred)
        assert 0.0 <= recall_score <= 1.0, "Invalid recall score"
        print(f"  ✓ keyword_recall_metric: {recall_score} (expected ~1.0)")

        # Test source type (should match "billing")
        source_score = source_type_metric(example, pred)
        assert 0.0 <= source_score <= 1.0, "Invalid source score"
        print(f"  ✓ source_type_metric: {source_score} (expected 1.0)")

        return True
    except Exception as e:
        print(f"  ✗ Metric test failed: {e}")
        return False


def test_file_structure() -> bool:
    """Test that all required files exist."""
    print("\n[5] Testing file structure...")
    try:
        base_dir = Path(__file__).parent
        required_files = [
            "__init__.py",
            "signatures.py",
            "modules.py",
            "dataset.py",
            "metrics.py",
            "trainer.py",
        ]

        for fname in required_files:
            fpath = base_dir / fname
            assert fpath.exists(), f"Missing {fname}"
            print(f"  ✓ {fname}")

        # Check compiled dir
        compiled_dir = base_dir / "compiled"
        assert compiled_dir.exists(), "Missing compiled/ directory"
        print(f"  ✓ compiled/ directory")

        return True
    except Exception as e:
        print(f"  ✗ File structure test failed: {e}")
        return False


def test_trainer_cli() -> bool:
    """Test that trainer CLI can be imported."""
    print("\n[6] Testing trainer CLI...")
    try:
        from dspy_module import trainer
        assert hasattr(trainer, "app"), "trainer.app not found"
        print("  ✓ Trainer CLI app created")

        # Check main commands exist (train_rag is required, others optional)
        required_commands = ["train_rag"]
        optional_commands = ["train_reranker", "train_council", "evaluate", "compare"]

        for cmd in required_commands:
            assert hasattr(trainer, cmd), f"Missing required command: {cmd}"
            print(f"    - {cmd} (required)")

        for cmd in optional_commands:
            if hasattr(trainer, cmd):
                print(f"    - {cmd}")

        return True
    except Exception as e:
        print(f"  ✗ Trainer CLI test failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("=" * 70)
    print("DSPy Framework Implementation — Smoke Test")
    print("=" * 70)

    results = []

    # Test 1: Imports
    try:
        results.append(("Imports", test_imports()))
    except Exception as e:
        print(f"Imports test crashed: {e}")
        results.append(("Imports", False))
        return 1

    if not results[-1][1]:
        print("\n[!] Cannot proceed without dspy-ai. Install with:")
        print("    pip install dspy-ai>=2.5")
        return 1

    # Test 2-6
    results.append(("Dataset", test_dataset()))
    results.append(("Signatures", test_signatures()))
    results.append(("Metrics", test_metrics()))
    results.append(("File Structure", test_file_structure()))
    results.append(("Trainer CLI", test_trainer_cli()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n[✓] All tests passed! DSPy framework is ready.")
        print("\nNext steps:")
        print("  1. Run cheapest training:")
        print("     python -m dspy_module.trainer train-rag --optimizer bootstrap --cheap --categories billing")
        print("\n  2. Compare with baseline:")
        print("     python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5 --dspy dspy_module/compiled/rag_latest.json")
        return 0
    else:
        print(f"\n[✗] {total - passed} test(s) failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
