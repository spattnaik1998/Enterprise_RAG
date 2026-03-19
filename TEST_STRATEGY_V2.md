# Testing Strategy: Review Brief v2 Implementation

## Overview
This document outlines a multi-level testing strategy for validating the four main changes:
1. Context rot cleanup (prompts.py + tiers.py)
2. DirectRAGAgent kwargs bug fix (router.py)
3. Skills → Router integration (router.py)
4. AUDIT_HMAC_KEY enforcement (gateway.py)

---

## Phase 1: Smoke Tests (5 min, no external APIs)

### 1.1 Import Validation
```bash
python -c "
from src.agents.router import QueryRouterAgent, RouteType, RouteDecision
from src.security.gateway import AgentSecurityGateway
from src.context.tiers import TierClassifier
from src.generation import prompts
print('PASS: All imports successful')
print('RouteType:', [e.value for e in RouteType])
"
```

**Expected**: All imports succeed; RouteType has 4 values (skill, direct_rag, council, tool_composer)

### 1.2 Content Verification
```bash
python -c "
from src.generation import prompts
from src.context.tiers import TierClassifier

# Check NO_CONTEXT_RESPONSE
assert 'communications' in prompts.NO_CONTEXT_RESPONSE, 'Missing communications'
assert 'arxiv' not in prompts.NO_CONTEXT_RESPONSE, 'Should not mention arxiv'
print('PASS: NO_CONTEXT_RESPONSE is clean')

# Check tiers docstring
import inspect
tiers_doc = inspect.getsource(TierClassifier.__module__)
assert 'arxiv' not in tiers_doc.lower(), 'Stale arxiv reference'
print('PASS: tiers.py docstring cleaned')
"
```

**Expected**: NO_CONTEXT_RESPONSE contains only MSP topics; no arxiv/wikipedia/rss references

---

## Phase 2: Unit Tests (15 min, no external APIs)

### 2.1 TierClassifier Tests
```bash
python << 'EOF'
from src.context.tiers import TierClassifier

tc = TierClassifier()
tests = [
    ('billing', 'working'),
    ('psa', 'working'),
    ('communications', 'working'),
    ('crm', 'ephemeral'),
    ('contracts', 'ephemeral'),
    ('unknown_source', 'ephemeral'),  # default
]

for source, expected_tier in tests:
    actual = tc.classify(source)
    assert actual == expected_tier, f"Expected {expected_tier}, got {actual}"
    print(f"PASS: {source} -> {actual}")

# Test priority ordering
assert tc.priority('working') > tc.priority('ephemeral'), "Priority ordering broken"
assert tc.priority('ephemeral') > tc.priority('longterm'), "Priority ordering broken"
print("PASS: Tier priorities are correct")
EOF
```

**Expected**: All tier classifications correct; priority ordering working

### 2.2 RouteDecision Dataclass Tests
```bash
python << 'EOF'
from src.agents.router import RouteDecision, RouteType, ClassificationResult

# Test SKILL route
decision1 = RouteDecision(
    route_type=RouteType.SKILL,
    confidence=0.85,
    skill_name="ARRiskReportSkill",
    reasoning="Skill match for AR risk"
)
assert decision1.route_type == RouteType.SKILL
assert decision1.skill_name == "ARRiskReportSkill"
print("PASS: RouteDecision SKILL route works")

# Test DIRECT_RAG route
decision2 = RouteDecision(
    route_type=RouteType.DIRECT_RAG,
    confidence=0.92,
    classification=ClassificationResult(
        query_class="SIMPLE",
        confidence=0.92,
        reasoning="Simple factual query"
    ),
    estimated_cost_usd=0.0012
)
assert decision2.route_type == RouteType.DIRECT_RAG
assert decision2.estimated_cost_usd == 0.0012
print("PASS: RouteDecision DIRECT_RAG route works")

# Test COUNCIL route
decision3 = RouteDecision(
    route_type=RouteType.COUNCIL,
    confidence=0.65,
    estimated_cost_usd=0.0082
)
assert decision3.route_type == RouteType.COUNCIL
print("PASS: RouteDecision COUNCIL route works")

# Test TOOL_COMPOSER route
decision4 = RouteDecision(
    route_type=RouteType.TOOL_COMPOSER,
    confidence=0.78,
    estimated_cost_usd=0.0000
)
assert decision4.route_type == RouteType.TOOL_COMPOSER
print("PASS: RouteDecision TOOL_COMPOSER route works")
EOF
```

**Expected**: All RouteDecision variants instantiate correctly

### 2.3 SkillRegistry Integration
```bash
python << 'EOF'
from src.skills.registry import SkillRegistry

registry = SkillRegistry()
skills = registry.list_skills()
print(f"PASS: SkillRegistry loaded {len(skills)} skills")

# Verify all expected skills are present
expected_skills = ['ARRiskReportSkill', 'TicketTriageSkill', 'ContractRenewalSkill',
                   'ClientHealthSkill', 'EscalationBriefSkill']
loaded_names = [s['name'] for s in skills]
for skill_name in expected_skills:
    # Check if skill name is in the loaded names (might be formatted differently)
    matches = [n for n in loaded_names if skill_name.lower() in n.lower()]
    if matches:
        print(f"PASS: {skill_name} found in registry")
    else:
        print(f"WARNING: {skill_name} not found in registry")
EOF
```

**Expected**: All 5 skills loaded successfully

### 2.4 AUDIT_HMAC_KEY Enforcement Tests
```bash
python << 'EOF'
import os
import sys

# Test 1: Dev environment (should pass)
os.environ['ENVIRONMENT'] = 'dev'
os.environ.pop('AUDIT_HMAC_KEY', None)

try:
    # Fresh import to test __init__
    if 'src.security.gateway' in sys.modules:
        del sys.modules['src.security.gateway']
    from src.security.gateway import AgentSecurityGateway
    gw = AgentSecurityGateway()
    print("PASS: Dev environment allows missing AUDIT_HMAC_KEY")
except RuntimeError as e:
    print(f"FAIL: Dev should not require key: {e}")

# Test 2: Production without key (should fail)
os.environ['ENVIRONMENT'] = 'production'
os.environ.pop('AUDIT_HMAC_KEY', None)

try:
    if 'src.security.gateway' in sys.modules:
        del sys.modules['src.security.gateway']
    from src.security.gateway import AgentSecurityGateway
    gw = AgentSecurityGateway()
    print("FAIL: Production should require AUDIT_HMAC_KEY")
except RuntimeError as e:
    if "AUDIT_HMAC_KEY" in str(e):
        print("PASS: Production enforces AUDIT_HMAC_KEY requirement")
    else:
        print(f"FAIL: Wrong error: {e}")

# Test 3: Production with key (should pass)
os.environ['ENVIRONMENT'] = 'production'
os.environ['AUDIT_HMAC_KEY'] = 'test-key-secure-12345'

try:
    if 'src.security.gateway' in sys.modules:
        del sys.modules['src.security.gateway']
    from src.security.gateway import AgentSecurityGateway
    gw = AgentSecurityGateway()
    print("PASS: Production allows properly configured AUDIT_HMAC_KEY")
except RuntimeError as e:
    print(f"FAIL: Production with key should pass: {e}")

# Reset to dev for other tests
os.environ['ENVIRONMENT'] = 'dev'
os.environ.pop('AUDIT_HMAC_KEY', None)
EOF
```

**Expected**: Dev allows missing key; production requires it; production allows it when set

---

## Phase 3: Integration Tests (20 min, local FAISS index required)

### 3.1 DirectRAGAgent Pipeline Integration
```bash
python << 'EOF'
import asyncio
from src.serving.pipeline import RAGPipeline
from src.agents.router import DirectRAGAgent

# Load local FAISS index
try:
    pipeline = RAGPipeline(index_dir="data/index")
    print(f"PASS: Loaded FAISS index with {pipeline.index.ntotal} vectors")
except Exception as e:
    print(f"SKIP: Index not available: {e}")
    exit(0)

# Test DirectRAGAgent query signature
agent = DirectRAGAgent(pipeline)

async def test_direct_rag():
    # Use a simple test query
    verdict = await agent.run(
        query="What clients have overdue invoices?",
        abac_ctx=None
    )
    assert verdict is not None, "Verdict should not be None"
    assert verdict.accepted_answer, "Should have an answer"
    print(f"PASS: DirectRAGAgent executed successfully")
    print(f"      Answer length: {len(verdict.accepted_answer)} chars")
    print(f"      Cost: ${verdict.total_cost_usd:.6f}")
    print(f"      Latency: {verdict.latency_ms:.0f}ms")
    return True

try:
    result = asyncio.run(test_direct_rag())
    if result:
        print("PASS: DirectRAGAgent integration test passed")
except Exception as e:
    print(f"FAIL: DirectRAGAgent test failed: {e}")
    import traceback
    traceback.print_exc()
EOF
```

**Expected**: DirectRAGAgent runs without TypeError; generates answer; completes in < 5 seconds

### 3.2 QueryRouterAgent Skill Matching
```bash
python << 'EOF'
import asyncio
from src.agents.router import QueryRouterAgent, QueryClassifier
from src.skills.registry import SkillRegistry

# Test skill matching without full pipeline
registry = SkillRegistry()

test_queries = [
    ("Generate AR risk report for overdue clients", "ARRiskReportSkill"),
    ("What is Alpine's invoice total?", None),  # Should not match skill
    ("Triage open support tickets", "TicketTriageSkill"),
]

for query, expected_skill in test_queries:
    matched_skill = registry.match(query)
    if expected_skill is None:
        if matched_skill is None or matched_skill.matches_query(query) <= 0.7:
            print(f"PASS: Query '{query[:40]}...' correctly did not match skill")
        else:
            print(f"FAIL: Query '{query[:40]}...' matched skill when it shouldn't")
    else:
        if matched_skill and expected_skill.lower() in matched_skill.name.lower():
            score = matched_skill.matches_query(query)
            print(f"PASS: Query matched {matched_skill.name} (score={score:.2f})")
        else:
            print(f"FAIL: Query did not match expected skill {expected_skill}")
EOF
```

**Expected**: Skill-triggering queries match correctly; generic queries do not

### 3.3 QueryClassifier Heuristics
```bash
python << 'EOF'
from src.agents.router import QueryClassifier

classifier = QueryClassifier(use_llm_fallback=False)

test_cases = [
    ("What is Alpine's contract value?", "SIMPLE"),
    ("Should we escalate this client?", "COMPLEX"),
    ("Get me a client 360 view", "AGGREGATE"),
    ("How much does ClientA owe?", "SIMPLE"),
    ("Compare our top 5 at-risk accounts", "COMPLEX"),
]

for query, expected_class in test_cases:
    result = classifier.classify(query)
    if result.query_class == expected_class:
        print(f"PASS: '{query[:40]}...' classified as {result.query_class}")
    else:
        print(f"FAIL: '{query[:40]}...' expected {expected_class}, got {result.query_class}")
EOF
```

**Expected**: Heuristic classifier correctly identifies SIMPLE/COMPLEX/AGGREGATE queries

---

## Phase 4: End-to-End Behavioral Tests (30 min, requires FAISS + OpenAI/Anthropic keys)

### 4.1 NO_CONTEXT_RESPONSE Fallback Test
```bash
python -m src.main phase3 --query "What is the difference between machine learning and AI?" --json
```

**Expected**: Answer should fall back to NO_CONTEXT_RESPONSE (no context found); text should list only MSP topics

**Success Criteria**:
- Response includes "could not find relevant information"
- Mentions billing, PSA, CRM, communications, contracts
- No mention of "AI/ML", "research", "papers"

### 4.2 DirectRAGAgent Latency Test
```bash
python << 'EOF'
import asyncio
import time
from src.serving.pipeline import RAGPipeline
from src.agents.router import DirectRAGAgent

try:
    pipeline = RAGPipeline(index_dir="data/index")
except Exception as e:
    print(f"SKIP: Index not available: {e}")
    exit(0)

agent = DirectRAGAgent(pipeline)

async def benchmark():
    latencies = []
    for _ in range(3):
        start = time.perf_counter()
        verdict = await agent.run("What is ClientA's account health?")
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

    avg = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    print(f"DirectRAGAgent latencies: {[f'{l:.0f}ms' for l in latencies]}")
    print(f"Average: {avg:.0f}ms, Max: {max_latency:.0f}ms")

    # Target: < 3 seconds (fast_path optimization)
    if max_latency < 3000:
        print("PASS: DirectRAGAgent meets latency target (< 3s)")
    else:
        print("WARN: DirectRAGAgent exceeds latency target")

asyncio.run(benchmark())
EOF
```

**Success Criteria**:
- DirectRAGAgent queries complete in < 3 seconds
- Faster than full Council pipeline (typical 8-12s)
- Shows cost savings from skipping full orchestration

### 4.3 Router Decision Logging Test
```bash
python << 'EOF'
import asyncio
import logging
from src.agents.router import QueryRouterAgent
from src.serving.pipeline import RAGPipeline

# Enable debug logging to see router decisions
logging.basicConfig(level=logging.DEBUG)

try:
    pipeline = RAGPipeline(index_dir="data/index")
except Exception as e:
    print(f"SKIP: Index not available: {e}")
    exit(0)

router = QueryRouterAgent(pipeline=pipeline, council=None, use_llm_classifier=False)

async def test_routing():
    test_queries = [
        "What is ClientA's invoice total?",      # SIMPLE
        "Should we escalate Alpine?",              # COMPLEX
        "Get full client 360 view",                # AGGREGATE
    ]

    for query in test_queries:
        print(f"\n>>> Query: {query}")
        verdict = await router.route(query)
        print(f"    Agent: {verdict.winning_agent}")
        print(f"    Cost: ${verdict.total_cost_usd:.4f}")
        print(f"    Latency: {verdict.latency_ms:.0f}ms")

asyncio.run(test_routing())
EOF
```

**Expected Output** (in logs):
```
[Router] Decision: route=direct_rag        | confidence=0.92 | cost=$0.0012 | latency=1234ms
[Router] Decision: route=council           | confidence=0.65 | cost=$0.0082 | latency=9876ms
[Router] Decision: route=tool_composer     | confidence=0.78 | cost=$0.0000 | latency=2345ms
```

**Success Criteria**:
- All routing decisions logged with RouteDecision data
- Confidence scores sensible (0.0-1.0)
- Cost differential visible (DirectRAG ~$0.001 vs Council ~$0.008)

### 4.4 Skill Execution Test (if available)
```bash
python << 'EOF'
import asyncio
from src.agents.router import QueryRouterAgent
from src.skills.registry import SkillRegistry
from src.skills.base import SkillContext

async def test_skill():
    registry = SkillRegistry()
    skill = registry.get('ARRiskReportSkill')

    if not skill:
        print("SKIP: ARRiskReportSkill not available")
        return

    context = SkillContext(query="Generate AR risk report for clients with overdue invoices")
    result = await skill.execute(context)

    print(f"Skill: {skill.name}")
    print(f"Success: {result.success}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    if result.data:
        print(f"Data length: {len(str(result.data))} chars")
    if result.error:
        print(f"Error: {result.error}")

asyncio.run(test_skill())
EOF
```

**Success Criteria**:
- Skill executes without errors
- Returns SkillResult with success flag
- Latency logged correctly

---

## Phase 5: Regression Tests (10 min)

### 5.1 Existing Pipeline Functionality
```bash
# Ensure basic RAG still works
python -m src.main phase3 --query "What is ClientA's status?" --json
```

**Expected**: Standard RAG pipeline still functions; answer includes citations

### 5.2 Council Orchestrator
```bash
python -m src.agents.council_cli --query "Should we escalate Alpine?"
```

**Expected**: Council pipeline still works; returns 3-agent voting verdict

### 5.3 Eval Framework
```bash
# Run a quick smoke eval
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 1 --no-rerank
```

**Expected**: Eval framework still works; metrics calculated correctly

---

## Phase 6: Performance Baseline (optional, 10 min)

Create a cost/latency comparison:

```bash
python << 'EOF'
import asyncio
import time
from src.serving.pipeline import RAGPipeline
from src.agents.router import DirectRAGAgent, QueryRouterAgent
from src.agents.council import CouncilOrchestrator

try:
    pipeline = RAGPipeline(index_dir="data/index")
except Exception as e:
    print(f"SKIP: Index not available: {e}")
    exit(0)

queries = [
    "What is ClientA's contract value?",
    "Which clients have overdue invoices?",
    "Get Alpine's full profile",
]

async def benchmark():
    print("\n=== Cost & Latency Comparison ===\n")

    for query in queries:
        print(f"Query: {query}\n")

        # DirectRAG
        agent = DirectRAGAgent(pipeline)
        start = time.perf_counter()
        verdict1 = await agent.run(query)
        latency1 = (time.perf_counter() - start) * 1000

        print(f"  DirectRAG:  ${verdict1.total_cost_usd:.4f}  {latency1:7.0f}ms")
        print()

asyncio.run(benchmark())
EOF
```

**Success Criteria**:
- DirectRAGAgent cost typically $0.0008-0.0015
- DirectRAGAgent latency typically 1000-3000ms
- Both significantly lower than full Council pipeline

---

## Test Execution Checklist

- [ ] Phase 1: Smoke tests (5 min)
- [ ] Phase 2: Unit tests (15 min)
- [ ] Phase 3: Integration tests (20 min)
- [ ] Phase 4: End-to-end tests (30 min)
- [ ] Phase 5: Regression tests (10 min)
- [ ] Phase 6: Performance baseline (optional, 10 min)

**Total time**: ~90 minutes (or ~50 minutes if skipping E2E)

---

## Troubleshooting

### DirectRAGAgent returns TypeError
- **Cause**: Old code still passing `top_k`/`rerank_top_k` kwargs
- **Fix**: Ensure `router.py` has been updated; `git diff HEAD~1 src/agents/router.py` should show kwargs removed

### Skills not matching
- **Cause**: `SkillRegistry.match()` score below 0.7 threshold
- **Fix**: Check `skill.matches_query(query)` returns > 0.7; may need to override in skill subclass

### AUDIT_HMAC_KEY errors in dev
- **Cause**: ENVIRONMENT env var set to something other than "dev"
- **Fix**: Unset or set to "dev": `unset ENVIRONMENT` or `ENVIRONMENT=dev`

### FAISS index not found
- **Cause**: Phase II not run
- **Fix**: Run `python -m src.main phase2` to build index (requires Phase I data)

### LLM/API timeouts
- **Cause**: OpenAI or Anthropic API issues
- **Fix**: Check connectivity; verify API keys in `.env`; increase timeout if needed

