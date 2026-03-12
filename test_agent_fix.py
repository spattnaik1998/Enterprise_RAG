#!/usr/bin/env python
"""
Quick test to verify the CouncilOrchestrator can return answers.
This bypasses the HTTP layer and tests the agent directly.
"""
import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

async def test_agent():
    """Test the agent with a simple query."""
    from src.serving.pipeline import RAGPipeline
    from src.embedding.supabase_index import SupabaseIndex
    from src.agents.council import CouncilOrchestrator

    print("[Test] Initializing Supabase index...")
    try:
        index = SupabaseIndex()
        print(f"[Test] Index loaded: {index.ntotal} vectors")
    except Exception as e:
        print(f"[Test] ERROR loading index: {e}")
        return False

    print("[Test] Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline(index=index)
        print(f"[Test] Pipeline ready")
    except Exception as e:
        print(f"[Test] ERROR initializing pipeline: {e}")
        return False

    print("[Test] Creating CouncilOrchestrator...")
    try:
        council = CouncilOrchestrator(pipeline)
        print("[Test] Council created")
    except Exception as e:
        print(f"[Test] ERROR creating council: {e}")
        return False

    # Test query
    test_queries = [
        "Which clients have overdue invoices?",
        "What is the renewal status for Alpine Financial?",
        "Show me high-risk accounts in the CRM",
    ]

    for query in test_queries:
        print(f"\n[Test] Testing query: {query!r}")
        try:
            verdict = await council.run(
                query=query,
                budget_tokens=3000,
                session_id="test_session_1",
            )
            print(f"[Test]   ✓ Got verdict from agent: {verdict.winning_agent}")
            print(f"[Test]   Answer: {verdict.accepted_answer[:100]}...")
            print(f"[Test]   Cost: ${verdict.total_cost_usd:.6f}")
            print(f"[Test]   Latency: {verdict.latency_ms:.0f}ms")
            print(f"[Test]   Escalated: {verdict.escalated}")
            if not verdict.accepted_answer or verdict.accepted_answer == "":
                print("[Test]   WARNING: Empty answer returned!")
                return False
        except Exception as e:
            print(f"[Test]   ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n[Test] ✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_agent())
    sys.exit(0 if success else 1)
