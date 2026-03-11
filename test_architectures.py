#!/usr/bin/env python3
"""
Quick verification script for multi-agent architecture implementations.

Tests basic imports and syntax without running expensive evaluations.

Usage:
    python test_architectures.py
"""
from __future__ import annotations

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

def test_imports():
    """Test that all new modules can be imported."""
    print("[TEST] Importing Architecture A (Parallel Orchestrator)...")
    try:
        from eval.orchestrator import (
            ParallelEvalOrchestrator,
            ModelShardAgent,
            JudgePoolWorker,
            ShardTask,
        )
        print("  ✓ eval/orchestrator.py imports successfully")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False

    print("[TEST] Importing Architecture B (Query Router)...")
    try:
        from src.agents.router import (
            QueryRouterAgent,
            QueryClassifier,
            DirectRAGAgent,
            ToolComposerAgent,
        )
        print("  ✓ src/agents/router.py imports successfully")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False

    print("[TEST] Importing Architecture C (Judge Panel)...")
    try:
        from eval.judge_panel import (
            JudgePanelOrchestrator,
            SpecialistJudge,
            DomainClassifier,
            CalibrationAgent,
        )
        print("  ✓ eval/judge_panel.py imports successfully")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False

    return True


def test_classifier():
    """Test query classifier."""
    print("\n[TEST] Testing QueryClassifier...")
    try:
        from src.agents.router import QueryClassifier

        classifier = QueryClassifier(use_llm_fallback=False)

        # Test simple pattern
        result = classifier.classify("What is Alpine Financial's contract value?")
        assert result.query_class == "SIMPLE", f"Expected SIMPLE, got {result.query_class}"
        print(f"  ✓ Classified simple query: {result.query_class} (confidence={result.confidence:.2f})")

        # Test complex pattern
        result = classifier.classify("Should we escalate Alpine Financial?")
        assert result.query_class == "COMPLEX", f"Expected COMPLEX, got {result.query_class}"
        print(f"  ✓ Classified complex query: {result.query_class} (confidence={result.confidence:.2f})")

        # Test aggregate pattern
        result = classifier.classify("Get client 360 for Alpine")
        assert result.query_class == "AGGREGATE", f"Expected AGGREGATE, got {result.query_class}"
        print(f"  ✓ Classified aggregate query: {result.query_class} (confidence={result.confidence:.2f})")

        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_judge_panel():
    """Test domain classifier in judge panel."""
    print("\n[TEST] Testing JudgePanelOrchestrator.DomainClassifier...")
    try:
        from eval.judge_panel import DomainClassifier

        classifier = DomainClassifier()

        # Test domain classification
        result = classifier.classify("Which clients have overdue invoices?")
        assert result in ["billing", "cross_source"], f"Expected billing/cross_source, got {result}"
        print(f"  ✓ Classified billing query to: {result}")

        result = classifier.classify("What are the SLA terms for the contract?")
        assert result in ["contracts", "cross_source"], f"Expected contracts/cross_source, got {result}"
        print(f"  ✓ Classified contracts query to: {result}")

        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_specialist_prompts():
    """Test that specialist judge prompts exist."""
    print("\n[TEST] Testing SpecialistJudge domain prompts...")
    try:
        from eval.judge_panel import SpecialistJudge

        domains = ["billing", "contracts", "crm", "psa", "communications", "cross_source"]
        for domain in domains:
            assert domain in SpecialistJudge.DOMAIN_PROMPTS, f"Missing prompt for {domain}"
            prompt = SpecialistJudge.DOMAIN_PROMPTS[domain]
            assert len(prompt) > 100, f"Prompt for {domain} too short"
            print(f"  ✓ {domain:15s} -- {len(prompt):4d} chars")

        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_run_eval_flags():
    """Test that run_eval.py has new flags."""
    print("\n[TEST] Checking eval/run_eval.py for new flags...")
    try:
        run_eval_path = Path("eval/run_eval.py")
        content = run_eval_path.read_text()

        assert "--parallel" in content, "Missing --parallel flag"
        print("  ✓ --parallel flag added")

        assert "--specialist-judges" in content, "Missing --specialist-judges flag"
        print("  ✓ --specialist-judges flag added")

        assert "ParallelEvalOrchestrator" in content, "Missing ParallelEvalOrchestrator import"
        print("  ✓ ParallelEvalOrchestrator imported")

        assert "JudgePanelOrchestrator" in content, "Missing JudgePanelOrchestrator import"
        print("  ✓ JudgePanelOrchestrator imported")

        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Multi-Agent Architecture Implementation Tests")
    print("=" * 70 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Query Classifier", test_classifier),
        ("Judge Panel Domain Classifier", test_judge_panel),
        ("Specialist Judge Prompts", test_specialist_prompts),
        ("eval/run_eval.py Flags", test_run_eval_flags),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")

    all_pass = all(result for _, result in results)
    print("=" * 70)
    if all_pass:
        print("All tests PASSED ✓")
        return 0
    else:
        print("Some tests FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
