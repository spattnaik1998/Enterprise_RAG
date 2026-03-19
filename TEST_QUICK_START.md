# Quick Start Testing Guide for Review Brief v2

## Status: ✓ Phase 1-2 Complete (Smoke + Unit Tests)

**Already Verified:**
- [x] All imports working (4 modified modules)
- [x] RouteType enum (skill, direct_rag, council, tool_composer)
- [x] NO_CONTEXT_RESPONSE contains only MSP topics (no RAG/AI/ML content)
- [x] TierClassifier correctly maps all 5 sources to tiers
- [x] RouteDecision dataclass works for all route types
- [x] SkillRegistry loads 5 skills successfully
- [x] AUDIT_HMAC_KEY enforcement (dev allows, production requires)

---

## Phase 3: Integration Tests (20 min, requires FAISS index)

Run locally without external APIs:

```bash
# Test DirectRAGAgent doesn't crash with new signature
python << 'EOF'
import asyncio
from src.serving.pipeline import RAGPipeline
from src.agents.router import DirectRAGAgent

try:
    pipeline = RAGPipeline(index_dir="data/index")
    agent = DirectRAGAgent(pipeline)

    async def test():
        verdict = await agent.run("What clients have overdue invoices?")
        print(f"PASS: DirectRAGAgent executed")
        print(f"  Answer length: {len(verdict.accepted_answer)} chars")
        print(f"  Latency: {verdict.latency_ms:.0f}ms")
        print(f"  Cost: ${verdict.total_cost_usd:.6f}")

    asyncio.run(test())
except FileNotFoundError:
    print("SKIP: FAISS index not found (run: python -m src.main phase2)")
except Exception as e:
    print(f"FAIL: {e}")
EOF
```

**Expected**: Runs without TypeError; completes in < 5 seconds

---

## Phase 4: End-to-End Tests (30 min, requires OpenAI/Anthropic keys)

### Test 4.1: NO_CONTEXT_RESPONSE Fallback
```bash
python -m src.main phase3 --query "What is the difference between machine learning and AI?" --json
```

**Expected Output**: Fallback message with MSP topics, no "AI/ML research" content

### Test 4.2: Router Logging Visibility
```bash
python << 'EOF'
import asyncio
import logging
from src.agents.router import QueryRouterAgent
from src.serving.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)

try:
    pipeline = RAGPipeline(index_dir="data/index")
    router = QueryRouterAgent(pipeline=pipeline, council=None, use_llm_classifier=False)

    async def test():
        queries = [
            "What is ClientA's invoice total?",      # Should route to DIRECT_RAG
            "Should we escalate Alpine?",              # Should route to COUNCIL
        ]
        for query in queries:
            print(f"\n>>> {query}")
            verdict = await router.route(query)
            print(f"    Agent: {verdict.winning_agent} | Cost: ${verdict.total_cost_usd:.4f}")

    asyncio.run(test())
except FileNotFoundError:
    print("SKIP: FAISS index not found")
except Exception as e:
    print(f"FAIL: {e}")
EOF
```

**Expected Output** (in logs):
```
[Router] Decision: route=direct_rag | confidence=0.92 | cost=$0.0012 | latency=1234ms
[Router] Decision: route=council   | confidence=0.65 | cost=$0.0082 | latency=9876ms
```

### Test 4.3: Skill Routing (if available)
```bash
python << 'EOF'
import asyncio
from src.skills.registry import SkillRegistry
from src.skills.base import SkillContext

async def test():
    registry = SkillRegistry()

    # Test AR Risk Report Skill
    skill = registry.get('ar_risk_report')
    if skill:
        ctx = SkillContext(query="Generate AR risk report for overdue invoices")
        result = await skill.execute(ctx)
        print(f"AR Risk Report Skill: success={result.success}, latency={result.latency_ms:.0f}ms")

asyncio.run(test())
EOF
```

**Expected**: Skill executes successfully with latency info

---

## Phase 5: Regression Tests (10 min)

Ensure no breakage to existing functionality:

```bash
# Test 5.1: Standard RAG pipeline
python -m src.main phase3 --query "What is ClientA's account health?" --json

# Test 5.2: Council orchestrator (if available)
python -m src.agents.council_cli --query "Should we escalate this client?"

# Test 5.3: Evaluation framework (requires all dependencies)
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 1 --no-rerank
```

**Expected**: All existing pipelines still work; no regressions

---

## Phase 6: Cost/Latency Benchmark (10 min, optional)

Compare DirectRAGAgent vs Council:

```bash
python << 'EOF'
import asyncio
import time
from src.serving.pipeline import RAGPipeline
from src.agents.router import DirectRAGAgent

try:
    pipeline = RAGPipeline(index_dir="data/index")
    agent = DirectRAGAgent(pipeline)

    async def benchmark():
        query = "What is ClientA's contract value?"
        latencies = []

        for i in range(3):
            start = time.perf_counter()
            verdict = await agent.run(query)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

        avg = sum(latencies) / len(latencies)
        print(f"DirectRAGAgent average latency: {avg:.0f}ms")
        print(f"Cost per query: ${verdict.total_cost_usd:.4f}")

        if avg < 3000:
            print("PASS: Meets < 3s latency target (fast_path optimization works)")
        else:
            print("WARN: Exceeds latency target")

    asyncio.run(benchmark())
except FileNotFoundError:
    print("SKIP: FAISS index not found")
except Exception as e:
    print(f"FAIL: {e}")
EOF
```

**Expected Targets**:
- DirectRAGAgent: < 3000ms, ~$0.0010 per query
- Council: ~8-12s, ~$0.0080 per query (3x more expensive + slower)

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `TypeError` in DirectRAGAgent | Old kwargs (top_k/rerank_top_k) | Confirm `git show HEAD src/agents/router.py` shows kwargs removed |
| FAISS index not found | Phase II not run | `python -m src.main phase2` (requires Phase I data) |
| LLM timeouts | API issues | Check `.env` keys; verify internet connection |
| Skills not matching | Confidence below 0.7 | Check `skill.matches_query(query)` score |
| AUDIT_HMAC_KEY errors | ENVIRONMENT != "dev" | Set `ENVIRONMENT=dev` or provide key in production |

---

## Test Coverage Summary

| Component | Phase | Status | Command |
|-----------|-------|--------|---------|
| **Context Cleanup** | 1-2 | ✓ PASS | Import + content verification |
| **DirectRAGAgent Bug** | 3-4 | Pending | Run DirectRAGAgent with query |
| **Skills Integration** | 3-4 | Pending | Test skill matching + execution |
| **AUDIT_HMAC_KEY** | 2 | ✓ PASS | Env enforcement verified |
| **Router Logging** | 4 | Pending | Run QueryRouterAgent + check logs |
| **Regression** | 5 | Pending | Run eval + council CLI |

---

## Recommended Test Order

1. **Quick Smoke** (5 min): Just ran Phase 1-2 ✓
2. **Integration** (20 min): Phase 3 (if FAISS available)
3. **E2E** (30 min): Phase 4 (if OpenAI/Anthropic keys available)
4. **Regression** (10 min): Phase 5 (ensure no breakage)
5. **Benchmark** (10 min): Phase 6 (optional, cost/latency comparison)

**Total time**: ~75 minutes (can skip E2E/Benchmark for ~50 min)

---

## Success Criteria

- [x] All imports work (Phase 1)
- [x] All unit tests pass (Phase 2)
- [ ] DirectRAGAgent executes without TypeError (Phase 3)
- [ ] Router logs routing decisions (Phase 4)
- [ ] NO_CONTEXT_RESPONSE shows only MSP topics (Phase 4)
- [ ] Existing pipelines still work (Phase 5)
- [ ] DirectRAG cost is 8-10x cheaper than Council (Phase 6, optional)

---

## Full Testing Details

For complete test cases, code snippets, and detailed troubleshooting:
→ See **TEST_STRATEGY_V2.md**

