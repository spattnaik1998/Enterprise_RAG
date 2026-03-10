# Feature Design Document
## TechVault MSP Enterprise RAG — Agentic Security, Latency, Context & Eval

Generated from `project_use_cases.md`. Designed to integrate with the existing
6-stage RAG pipeline documented in `CLAUDE.md`.

---

## System Themes Identified

| Theme | Source use case | Cross-cutting concern |
|---|---|---|
| Agent Security | Use case 2 (ASG) | Every feature |
| Latency-aware Architecture | Use case 4 (Edge Agent) | ContextManager, Trace |
| Context Engineering | Use case 3 (Context-Optimized RAG) | All retrieval paths |
| Agentic Evaluation | Use case 1 (Eval Platform) | Every feature needs an eval hook |
| Multi-Agent Orchestration | Use case 5 (Council Demo) | Builds on security + trace |
| Observability & Traceability | Use case 6 (Tail-Capture) | LangSmith extension |

---

## Priority Tiers

| Tier | Rationale |
|---|---|
| **Tier 1** — High impact, low complexity | Extend existing modules; unblock all other features |
| **Tier 2** — Moderate complexity | New subsystems; depend on Tier 1 primitives |
| **Tier 3** — Advanced / research | Novel architectures; depend on Tier 1 + 2 |

---

# TIER 1 — High Impact + Easy to Implement

---

## FEATURE 1 — Policy-Driven Agent Security Gateway (ASG)

### Problem it solves
The existing `PromptGuard` (`src/retrieval/guardrails.py`) uses 13 hardcoded
regex patterns and has no concept of who is calling, what data they may access,
or whether the tool call is authorized. There is no audit trail. As MCP tool
calls multiply (17 tools today), the attack surface expands with no enforcement
layer between callers and sensitive enterprise data (billing, CRM, contracts).

### System components required
- Policy engine (YAML-defined rules, evaluated at call time)
- ABAC attribute provider (user role, data classification, environment)
- Enhanced PromptGuard (loadable rule sets + justification fields)
- Immutable audit logger (append-only, SHA-256 signed entries)
- ASG middleware (intercepts `RAGPipeline.query()` and MCP tool calls)

### Detailed implementation plan

**Step 1 — Policy DSL loader** (`src/security/policy_engine.py`)
Parse `config/policies.yaml`. Each policy has an `id`, an `action` it governs,
an `abac` block with `required_attrs` list and `rules` list (if/effect pairs),
and an optional `justification_required` boolean. Load policies at startup into
a dict keyed by action name.

**Step 2 — ABAC context object** (`src/security/abac.py`)
Define `ABACContext(user_role, data_classification, environment, extra_attrs)`.
Pass this context into every tool call wrapper. For the demo, read `user_role`
from an `X-User-Role` HTTP header (FastAPI dependency injection).

**Step 3 — Policy evaluator** (`src/security/policy_engine.py`)
`PolicyEngine.evaluate(action, abac_ctx) -> PolicyDecision(allowed, reasons)`.
Evaluate each rule using `eval()` against the ABAC context dict. Return `deny`
on any matching deny rule; return `allow` if no deny matches. Cache compiled
rules for latency.

**Step 4 — Enhanced PromptGuard** (`src/retrieval/guardrails.py`)
Load additional injection patterns from `config/injection_patterns.yaml` (100
curated patterns) at startup to reach the 98% TP / 2% FP target. Keep the
existing 13 hardcoded patterns as the always-on baseline; supplemental patterns
are loaded from YAML so they can be updated without code changes.

**Step 5 — Audit logger** (`src/security/audit_logger.py`)
Append-only JSONL file (`data/audit/audit.jsonl`). Each entry is a JSON object
signed with `hmac.new(secret_key, entry_bytes, sha256).hexdigest()`. Fields:
`timestamp`, `action`, `user_role`, `allowed`, `reasons`, `query_hash`,
`policy_ids_evaluated`, `hmac_sig`. Provide `AuditLogger.append(entry)` and
`AuditLogger.verify_chain()` (re-computes all HMACs and returns any breaks).

**Step 6 — ASG middleware** (`src/security/gateway.py`)
`AgentSecurityGateway` wraps `RAGPipeline.query()`:
```
check_prompt_guard -> evaluate_policies -> execute -> audit_log -> return
```
Also wraps each MCP tool call in `src/collection/mcp/server.py` via a decorator
`@asg_tool(action="billing_read", classification="sensitive")`.

**Step 7 — FastAPI integration** (`app/server.py`)
Inject `ABACContext` as a FastAPI dependency from request headers.
Pass it through `POST /api/chat` into `AgentSecurityGateway.handle()`.

### Files to create or modify

| File | Action | Purpose |
|---|---|---|
| `src/security/__init__.py` | Create | Package |
| `src/security/policy_engine.py` | Create | PolicyEngine + PolicyDecision |
| `src/security/abac.py` | Create | ABACContext dataclass |
| `src/security/audit_logger.py` | Create | HMAC-signed append-only log |
| `src/security/gateway.py` | Create | ASG orchestrator |
| `src/retrieval/guardrails.py` | Modify | Load extra patterns from YAML |
| `config/policies.yaml` | Create | Policy DSL definitions |
| `config/injection_patterns.yaml` | Create | 100 curated injection strings |
| `app/server.py` | Modify | Inject ABACContext dependency |
| `src/collection/mcp/server.py` | Modify | Add @asg_tool decorators |
| `data/audit/.gitkeep` | Create | Audit log directory |

### API interfaces

```python
# Policy evaluation
class PolicyEngine:
    def evaluate(self, action: str, ctx: ABACContext) -> PolicyDecision: ...
    def load_policies(self, path: str) -> None: ...

# ABAC context
@dataclass
class ABACContext:
    user_role: str          # e.g. "finance", "technician", "admin"
    data_classification: str  # "public", "internal", "sensitive", "restricted"
    environment: str        # "production", "staging", "dev"
    extra_attrs: dict = field(default_factory=dict)

@dataclass
class PolicyDecision:
    allowed: bool
    reasons: list[str]
    policy_ids_evaluated: list[str]

# Gateway
class AgentSecurityGateway:
    def handle(self, query: str, ctx: ABACContext, generator=None) -> QueryResult: ...
```

### Data schemas

```yaml
# config/policies.yaml
policies:
  - id: allow_billing_read
    action: read_billing
    abac:
      required_attrs: [user.role, data.classification]
      rules:
        - if: "data_classification == 'sensitive' and user_role != 'finance'"
          effect: deny
    justification_required: false

  - id: block_contract_export_non_admin
    action: export_contracts
    abac:
      required_attrs: [user.role]
      rules:
        - if: "user_role not in ['admin', 'legal']"
          effect: deny
    justification_required: true
```

```json
// audit log entry (data/audit/audit.jsonl — one JSON object per line)
{
  "timestamp": "2026-03-09T10:00:00Z",
  "action": "read_billing",
  "user_role": "finance",
  "data_classification": "sensitive",
  "allowed": true,
  "reasons": [],
  "policy_ids_evaluated": ["allow_billing_read"],
  "query_hash": "sha256:...",
  "hmac_sig": "..."
}
```

### Security considerations
- Rule evaluation uses `eval()` on a restricted namespace (only `ABACContext`
  fields). Never expose `__builtins__` in the eval scope.
- HMAC secret key stored in `.env` as `AUDIT_HMAC_KEY`; never committed.
- Audit file is append-only; deny filesystem write access to the audit dir
  except via `AuditLogger`. In production, use a write-once blob store.
- All tool call arguments are hashed (not stored raw) in audit entries to avoid
  storing PII in the audit log.

### Latency considerations
- Policy evaluation is in-memory (dict lookup + rule eval on ABAC context dict)
  — target < 1 ms per call.
- Audit log write is synchronous but unbuffered JSONL append (< 0.5 ms on SSD).
- For high-throughput paths, batch audit writes with a 100 ms flush interval.
- Compiled `re.Pattern` objects for injection patterns are cached at module load.

### Context engineering strategy
- ABAC context travels with every query as a lightweight immutable struct; never
  embedded in the LLM context window.
- Guardrail block reasons are returned to the caller but not injected into
  subsequent LLM prompts (prevents information leakage about rule logic).

### Evaluation strategy
- **Injection test suite**: `eval/datasets/security_injection.json` — 100
  crafted injection strings. Run: `python -m eval.run_security --mode injection`.
  Pass threshold: TP >= 98%, FP <= 2% on 50 benign billing queries.
- **ABAC unit tests**: parameterized tests over (role, classification, action)
  combinations. All deny rules must fire correctly.
- **Audit integrity test**: write N entries, corrupt entry K, run
  `AuditLogger.verify_chain()`, assert it detects the break at position K.

### Implementation steps for Claude Code
1. Create `src/security/` package with `abac.py`, `policy_engine.py`,
   `audit_logger.py`, `gateway.py`.
2. Write `config/policies.yaml` with 5 starter policies covering billing, CRM,
   contracts, PSA, and cross-source queries.
3. Write `config/injection_patterns.yaml` with 100 patterns (start from the 13
   existing, add categories: indirect injection, JSON injection, Markdown
   injection, tool-call smuggling, Unicode confusable attacks).
4. Modify `src/retrieval/guardrails.py` to load `injection_patterns.yaml` at
   startup and merge with the existing hardcoded list.
5. Modify `app/server.py` to extract `X-User-Role` and `X-Data-Classification`
   headers and build `ABACContext`; pass into gateway.
6. Modify `src/collection/mcp/server.py` — add `@asg_tool` decorator to each
   of the 17 tool functions specifying `action` and `classification`.
7. Add `eval/run_security.py` CLI that loads the injection test dataset, runs
   `PromptGuard.check()` on each, and reports TP/FP metrics.

### Estimated complexity: LOW-MEDIUM (3–5 days)
### Dependencies: None (Tier 1, foundational)
### Tests
- `tests/security/test_policy_engine.py` — unit tests for rule evaluation
- `tests/security/test_audit_logger.py` — append, verify, detect corruption
- `tests/security/test_gateway.py` — integration test for full ASG flow
- `eval/run_security.py` — injection benchmark (100 strings)

---

## FEATURE 2 — Gold-Task CI Runner + Regression Gate

### Problem it solves
There is no automated gate preventing regressions from being merged. The
existing `eval/run_eval.py` is an operator tool requiring ~$2.82 to run in full.
Pull requests can silently degrade RAG quality. A cheap, fast gold-task subset
run in CI catches regressions before they land.

### System components required
- Gold-task dataset (20 queries drawn from existing `eval/datasets/*.json`)
- CI runner script (`eval/run_ci.py`) — single model, small sample, strict gate
- GitHub Actions workflow (`.github/workflows/eval_ci.yml`)
- Regression comparator (compare current run vs baseline stored in repo)

### Detailed implementation plan

**Step 1 — Gold task selection**
From the 80 existing queries, select 20 covering all 6 categories, weighted
toward high-difficulty items (where regressions show first). Store as
`eval/datasets/gold_tasks.json` with the same schema as other datasets but
tagged `"gold": true`.

**Step 2 — CI runner** (`eval/run_ci.py`)
Thin wrapper around `RAGEvaluator.run()` with fixed parameters:
- Model: `gpt-4o-mini` only (cheapest, ~$0.01 for 20 queries)
- Categories: all 6 via the 20 gold tasks
- `enable_reranking=True`
- Saves result to `eval/results/ci_latest.json`
- Loads `eval/results/ci_baseline.json` (committed to repo)
- Compares composite score: if `baseline_composite - current_composite > 0.03`
  (3 point regression), exits 1.
- Also fails if any individual metric drops more than 5 points vs baseline.

**Step 3 — Baseline update command**
`python -m eval.run_ci --update-baseline` — runs the eval, saves result as the
new `eval/results/ci_baseline.json`, and prints a diff. Designed to be run
manually by a maintainer before merging a deliberate model change.

**Step 4 — GitHub Actions workflow**
`.github/workflows/eval_ci.yml`:
```yaml
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python -m src.main phase2  # skipped if data/index/ cached
      - run: python -m eval.run_ci
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```
Cache `data/index/` with `actions/cache` keyed on `data/validated/` hash to
avoid re-embedding on every PR.

### Files to create or modify

| File | Action | Purpose |
|---|---|---|
| `eval/run_ci.py` | Create | CI runner + regression comparator |
| `eval/datasets/gold_tasks.json` | Create | 20 curated gold queries |
| `eval/results/ci_baseline.json` | Create | Committed baseline scores |
| `.github/workflows/eval_ci.yml` | Create | GitHub Actions job |

### API interfaces

```bash
# Normal CI run (called by GH Actions — exits 0/1)
python -m eval.run_ci

# Update baseline after intentional model change
python -m eval.run_ci --update-baseline

# Run with custom threshold override
python -m eval.run_ci --regression-tolerance 0.05
```

### Data schemas

```json
// eval/results/ci_baseline.json
{
  "model": "gpt-4o-mini",
  "n_queries": 20,
  "composite": 0.863,
  "recall_at_10": 0.900,
  "source_type_hit": 0.900,
  "faithfulness": 0.857,
  "correctness": 0.795,
  "created_at": "2026-03-09T00:00:00Z",
  "git_sha": "16e2e3a"
}
```

### Security considerations
- `OPENAI_API_KEY` lives in GitHub Actions secrets; never printed in logs.
- CI runner uses `--quiet` mode to suppress verbose output (and any potential
  PII in query answers) from the Actions log stream.

### Latency considerations
- 20 queries x 1 model x (retrieval + rerank + generation + judge) = ~60 s
  at typical API latency. Within GitHub Actions 6-minute job limit.
- Index cache eliminates ~15 s Phase II re-run on most PRs.

### Context engineering strategy
- Gold tasks are selected to be concise (< 20 token queries) and have
  unambiguous `expected_keywords` so recall scoring is deterministic.

### Evaluation strategy
The runner IS the evaluation. Meta-validation:
- Manually verify that the 20 gold tasks represent all 6 categories.
- Run baseline update on a known-good state; confirm all metrics pass.
- Introduce a deliberate retrieval regression (change `dense_weight` to 0.3)
  and verify `run_ci.py` exits 1.

### Implementation steps for Claude Code
1. Pick 3–4 queries from each `eval/datasets/*.json` file (prefer difficulty
   `"hard"`); write `eval/datasets/gold_tasks.json`.
2. Implement `eval/run_ci.py` using `RAGEvaluator.run()` with `sample_n=0`
   and the gold tasks file loaded directly.
3. Run once on the current codebase; save output as `ci_baseline.json`.
4. Write `.github/workflows/eval_ci.yml` with secrets and index cache.
5. Test the regression path: temporarily lower `dense_weight` in
   `config/config.yaml`, run `run_ci.py`, confirm exit 1, revert.

### Estimated complexity: LOW (2–3 days)
### Dependencies: None (standalone extension of existing eval/)
### Tests
- Manual: run `python -m eval.run_ci` and confirm it exits 0 on main.
- Regression injection test (see implementation step 5 above).

---

## FEATURE 3 — ContextManager SDK

### Problem it solves
The current `HybridRetriever` returns up to 20 chunks flat, with no concept of
token budget, freshness, or context tier (working vs. ephemeral vs. long-term).
For latency-sensitive paths (p95 <= 250 ms local, 1,200 ms cloud-assisted),
callers need a budget-aware context API that also drives the "lost in the
middle" mitigation and enables the progressive-disclosure UI.

### System components required
- `ContextPiece` dataclass (id, text, tokens, freshness_score, relevance_score,
  tier, source_type)
- `ContextManager` class with `get_context(query, budget_tokens)` method
- Freshness scorer (uses `chunk.metadata.date` vs. current timestamp)
- Tier classifier (working/ephemeral/longterm based on source_type heuristics)
- `ContextBundle` return type (selected pieces + metadata about truncation)
- Integration hook in `RAGPipeline.query()` to replace raw chunk list with
  `ContextBundle`

### Detailed implementation plan

**Step 1 — ContextPiece + ContextBundle** (`src/context/schemas.py`)
```python
@dataclass
class ContextPiece:
    id: str
    text: str
    tokens: int
    relevance_score: float
    freshness_score: float   # 1.0 = today, decays linearly over 90 days
    tier: Literal["working", "ephemeral", "longterm"]
    source_type: str
    chunk_index: int         # position in original retrieval result

@dataclass
class ContextBundle:
    pieces: list[ContextPiece]
    total_tokens: int
    budget_tokens: int
    truncated: bool          # True if pieces were dropped to fit budget
    dropped_count: int
    strategy_used: str       # "budget_constrained" | "full"
```

**Step 2 — Freshness scorer** (`src/context/freshness.py`)
`FreshnessScorer.score(chunk) -> float`. Read `chunk.metadata.get("date")`;
if present, score = max(0, 1 - days_since / 90). If no date, score = 0.5
(neutral). This penalises stale RSS entries and rewards recent billing records.

**Step 3 — Tier classifier** (`src/context/tiers.py`)
Map `source_type` to tier:
- `working`: billing, psa, communications (operationally immediate)
- `ephemeral`: crm, contracts (query-relevant but not time-critical)
- `longterm`: arxiv, wikipedia, rss (background knowledge)
Tier influences sort order: working pieces are placed at the top and bottom of
the context window (mitigating "lost in the middle"); longterm pieces go in the
middle.

**Step 4 — ContextManager** (`src/context/manager.py`)
`ContextManager.get_context(query, budget_tokens, retrieval_results) -> ContextBundle`:
1. Convert `Chunk` list to `ContextPiece` list (compute freshness + tier).
2. Sort by `(tier_priority, relevance_score * freshness_score)` descending.
3. Greedily pack pieces into budget: accumulate `tokens` until `>= budget`.
4. Apply "lost in the middle" reorder: working pieces at indices 0..K and -K..-1;
   ephemeral + longterm in the middle.
5. Return `ContextBundle`.

**Step 5 — Integration in RAGPipeline** (`src/serving/pipeline.py`)
After `LLMReranker.rerank()`, pass chunks through `ContextManager.get_context()`
with `budget_tokens` from config (default 3000, lowered to 1024 for fast path).
Pass `bundle.pieces` to `RAGGenerator.generate()` instead of raw chunks.

**Step 6 — Fast path** (`src/serving/pipeline.py`)
Add `fast_path: bool` parameter to `pipeline.query()`. When `True`, skip
`LLMReranker` and set `budget_tokens=1024`. Target: p95 < 400 ms total (
embedding ~50 ms + FAISS ~10 ms + BM25 ~5 ms + context pack ~1 ms + generation
~300 ms).

**Step 7 — Progressive-disclosure API endpoint** (`app/server.py`)
Extend `POST /api/chat` response to include `context_pieces` array:
```json
{
  "answer": "...",
  "citations": [...],
  "context_pieces": [
    {"id": "...", "tier": "working", "relevance_score": 0.92,
     "freshness_score": 0.85, "tokens": 142, "included": true}
  ],
  "context_budget_tokens": 3000,
  "context_truncated": false
}
```

### Files to create or modify

| File | Action | Purpose |
|---|---|---|
| `src/context/__init__.py` | Create | Package |
| `src/context/schemas.py` | Create | ContextPiece, ContextBundle |
| `src/context/freshness.py` | Create | FreshnessScorer |
| `src/context/tiers.py` | Create | TierClassifier |
| `src/context/manager.py` | Create | ContextManager.get_context() |
| `src/serving/pipeline.py` | Modify | Plug ContextManager after reranker |
| `app/server.py` | Modify | Return context_pieces in /api/chat response |
| `config/config.yaml` | Modify | Add context.budget_tokens, context.fast_path |

### API interfaces

```python
class ContextManager:
    def get_context(
        self,
        query: str,
        retrieval_results: list[Chunk],
        budget_tokens: int = 3000,
    ) -> ContextBundle: ...

class FreshnessScorer:
    def score(self, chunk: Chunk) -> float: ...

class TierClassifier:
    def classify(self, source_type: str) -> Literal["working", "ephemeral", "longterm"]: ...
```

### Data schemas
See `ContextPiece` and `ContextBundle` in implementation plan above.

### Security considerations
- `ContextPiece.text` may contain PII; apply `PIIFilter` before returning
  `context_pieces` in the API response (only redact API-facing fields, not the
  internally-used generation context).
- Long-term memory writes (future extension) must go through the memory safety
  layer (PII tagging + ACL check) before persistence.

### Latency considerations
- `get_context()` is a synchronous CPU-bound operation: sort + greedy pack over
  ≤ 20 chunks. Target < 2 ms.
- `budget_tokens=1024` fast path eliminates the LLM reranker (~300–500 ms OpenAI
  RTT); total pipeline drops from ~1,200 ms to ~400 ms for working-context-only
  queries.
- Token counting uses `tiktoken` (already installed); cache the encoder at
  module load.

### Context engineering strategy
This feature IS the context engineering strategy:
- Progressive disclosure: start with `budget_tokens=1024` (working tier only);
  client UI offers "load more context" that re-queries with `budget_tokens=3000`.
- "Lost in the middle" mitigation: reorder so the most relevant chunks are at
  position 0 and N-1 of the context window.
- Freshness decay: stale documents are down-ranked automatically; the model
  sees recent MSP data first.

### Evaluation strategy
- **Token reduction test**: compare total context tokens per query with/without
  `ContextManager` on the 80-query eval set. Target: >= 30% reduction at
  `budget_tokens=3000` vs. naive full-context.
- **Cited context precision**: measure what fraction of `context_pieces` appear
  as citations in the final answer. Target: >= 80%.
- **Fast path accuracy**: run the 20 gold tasks with `fast_path=True`; confirm
  composite score >= 0.75 (acceptable degradation from 0.82 full-path baseline).

### Implementation steps for Claude Code
1. Create `src/context/` package with `schemas.py`, `freshness.py`, `tiers.py`.
2. Implement `manager.py`; write unit tests for sort order and greedy packing.
3. Modify `src/serving/pipeline.py`: inject `ContextManager` between reranker
   and generator; add `fast_path` flag.
4. Modify `app/server.py`: extend `ChatResponse` Pydantic model to include
   `context_pieces`; populate from `ContextBundle`.
5. Add `context.budget_tokens` and `context.fast_path_budget_tokens` to
   `config/config.yaml`.
6. Write a quick-bench script `scripts/bench_context.py` that runs 20 queries
   in full-path vs. fast-path mode and reports latency + token stats.

### Estimated complexity: LOW-MEDIUM (3–4 days)
### Dependencies: None (integrates with existing Phase II/III)
### Tests
- `tests/context/test_manager.py` — budget constraint, sort order, truncation
- `tests/context/test_freshness.py` — date-present, date-absent, very-old cases
- `scripts/bench_context.py` — latency comparison script

---

# TIER 2 — Moderate Complexity

---

## FEATURE 4 — Structured Trace Collector + Failure-Biased Sampler

### Problem it solves
LangSmith tracing covers Phase II/III but lacks a structured `case-file` schema
for agent runs (multi-step plan + tool calls + approval events + cost + verdict).
Failures in long-horizon workflows are rare but critical; flat sampling misses
them. There is no way to replay a historical trace deterministically for
debugging.

### System components required
- Agent trace schema (JSON Schema + Pydantic model)
- `TraceCollector` service — accepts `TraceEvent` objects, assembles case files
- `FailureBiasedSampler` — config-driven capture rules (error, high cost, PII,
  policy-block)
- Trace store (JSONL file-based, indexed by trace_id; queryable by failure type)
- `TraceReplayer` — re-runs a trace in isolated environment for debugging
- Integration hooks in `RAGPipeline.query()` and MCP tool calls

### Detailed implementation plan

**Step 1 — Trace schema** (`src/observability/schemas.py`)
```python
@dataclass
class TraceEvent:
    event_type: str   # "query_start" | "retrieval" | "rerank" | "generate" |
                      # "tool_call" | "guardrail_block" | "pii_redact" | "verdict"
    timestamp: str
    duration_ms: float
    payload: dict     # event-specific fields (redacted query, chunk_ids, etc.)
    cost_usd: float
    error: str | None

@dataclass
class AgentTrace:
    trace_id: str
    session_id: str
    query_hash: str   # SHA-256 of raw query (no PII)
    model: str
    user_role: str
    events: list[TraceEvent]
    total_cost_usd: float
    verdict: str      # "success" | "guardrail_block" | "error" | "pii_redacted"
    capture_reason: str | None  # why failure-biased sampler captured this trace
    created_at: str
```

**Step 2 — TraceCollector** (`src/observability/collector.py`)
Context-manager API: `with TraceCollector(session_id) as tc:`. Internally uses
`contextvars.ContextVar` to make the active collector available within the async
call stack without threading issues. Exposes `tc.add_event(event)`. On exit,
assembles `AgentTrace`, applies `FailureBiasedSampler`, and writes to store.

**Step 3 — FailureBiasedSampler** (`src/observability/sampler.py`)
Reads `config/trace_rules.yaml`. Capture a trace if ANY rule matches:
```yaml
capture_rules:
  - condition: "verdict == 'error'"
    reason: "pipeline_error"
  - condition: "total_cost_usd > 0.05"
    reason: "high_cost"
  - condition: "'pii_redact' in event_types"
    reason: "pii_access"
  - condition: "'guardrail_block' in event_types"
    reason: "security_event"
  - condition: "random() < 0.01"   # 1% baseline sampling
    reason: "baseline_sample"
```

**Step 4 — Trace store** (`src/observability/store.py`)
`TraceStore` backed by `data/traces/` directory. Each trace is one JSONL file
`data/traces/{trace_id}.json`. An index file `data/traces/index.jsonl` contains
one line per trace: `{trace_id, created_at, verdict, capture_reason, total_cost_usd}`.
`TraceStore.query(verdict=None, capture_reason=None, limit=50)` scans the index.

**Step 5 — TraceReplayer** (`src/observability/replayer.py`)
`TraceReplayer.replay(trace_id)` loads the trace, re-feeds the original query
(reconstructed from `query_hash` — actually the raw query is stored in the
`query_start` event payload, redacted of PII) through the current pipeline, and
returns a new `AgentTrace` for diff comparison.

**Step 6 — Integration hooks**
Modify `RAGPipeline.query()` to open a `TraceCollector` context, call
`tc.add_event()` at each pipeline stage, and emit the final verdict.
Modify `src/collection/mcp/server.py` to add `tc.add_event(tool_call_event)`
inside each tool function.

**Step 7 — Trace API endpoint** (`app/server.py`)
```
GET /api/traces?verdict=error&limit=20        # list recent error traces
GET /api/traces/{trace_id}                    # full trace detail
POST /api/traces/{trace_id}/replay            # re-run trace
```

### Files to create or modify

| File | Action | Purpose |
|---|---|---|
| `src/observability/__init__.py` | Create | Package |
| `src/observability/schemas.py` | Create | TraceEvent, AgentTrace |
| `src/observability/collector.py` | Create | TraceCollector context manager |
| `src/observability/sampler.py` | Create | FailureBiasedSampler |
| `src/observability/store.py` | Create | TraceStore JSONL backend |
| `src/observability/replayer.py` | Create | TraceReplayer |
| `config/trace_rules.yaml` | Create | Capture rule definitions |
| `src/serving/pipeline.py` | Modify | Add TraceCollector integration |
| `src/collection/mcp/server.py` | Modify | Add tool call trace events |
| `app/server.py` | Modify | Trace API endpoints |
| `data/traces/.gitkeep` | Create | Trace store directory |

### API interfaces

```python
# Collector (context manager)
with TraceCollector(session_id="abc") as tc:
    tc.add_event(TraceEvent(event_type="query_start", ...))
    # ... pipeline runs ...
    tc.set_verdict("success")
# -> automatically sampled and written to store

# Store
store = TraceStore("data/traces")
traces = store.query(verdict="error", limit=20)

# Replayer
replayer = TraceReplayer(pipeline)
new_trace = replayer.replay(trace_id="abc-123")
```

### Data schemas
See `TraceEvent` and `AgentTrace` in implementation plan above.

### Security considerations
- Raw query text is stored in the `query_start` event payload. Apply `PIIFilter`
  before writing to the trace store (same redaction as pipeline output).
- Trace store is local-only by default; production deployments should write to
  an encrypted blob store with access logs.
- `TraceReplayer` runs in a read-only pipeline mode (no actual tool side-effects)
  to prevent replay attacks.

### Latency considerations
- `tc.add_event()` is a non-blocking append to an in-memory list; no I/O in the
  hot path. Write to disk happens after the pipeline returns.
- `FailureBiasedSampler` evaluation is O(N rules) synchronous CPU op (< 0.1 ms).
- Store writes are async (run in executor) if `asyncio` event loop is available.

### Context engineering strategy
- Traces store chunk IDs, not chunk text, to keep trace files small.
- The `query_hash` allows correlation with eval results without storing raw
  queries in a second location.

### Evaluation strategy
- **Failure capture rate**: synthetically inject 20 error traces and 20 PII-
  access traces; confirm sampler captures >= 95% of each type.
- **Replay fidelity**: replay 10 successful traces; compare retrieved chunk IDs
  (should match 100% given same index); compare answer similarity (target > 0.9
  cosine similarity).
- **MTTR demo**: with traces enabled, simulate a billing query regression;
  measure time to identify root cause using trace drill-down vs. log scanning.

### Implementation steps for Claude Code
1. Create `src/observability/` package.
2. Implement `schemas.py`, `collector.py`, `sampler.py`, `store.py` in order.
3. Implement `replayer.py` (load trace, extract query, re-run pipeline).
4. Modify `src/serving/pipeline.py`: wrap `query()` body with TraceCollector.
5. Write `config/trace_rules.yaml` with the 5 starter capture rules.
6. Add trace API endpoints to `app/server.py`.
7. Write a smoke test: run a known-bad query (one that triggers PromptGuard),
   confirm trace is written with `verdict=guardrail_block`.

### Estimated complexity: MEDIUM (5–7 days)
### Dependencies: Feature 1 (ABAC context for `user_role` in trace)
### Tests
- `tests/observability/test_collector.py`
- `tests/observability/test_sampler.py` — rule matching, 1% baseline sampling
- `tests/observability/test_replayer.py`

---

## FEATURE 5 — LLM-Judge Calibration Harness

### Problem it solves
The existing `LLMJudge` in `eval/judge.py` is used as the sole arbiter of
`faithfulness` and `correctness`. If the judge is miscalibrated (systematically
over- or under-scoring), all downstream production decisions are wrong. There is
no way to measure judge quality against human labels, no confusion matrix, and no
alert when judge drift occurs across model versions.

### System components required
- Human-label dataset (`eval/datasets/judge_labels.json` — 50 queries with
  human-annotated faithfulness + correctness scores)
- `JudgeCalibrator` class that runs the LLM judge on the labelled set
- Calibration reporter (precision, recall, F1, confusion matrix, Pearson r,
  mean absolute error)
- CLI command: `python -m eval.calibrate_judge`
- Drift alert: compare current calibration vs. stored baseline; fail if F1
  drops more than 0.05

### Detailed implementation plan

**Step 1 — Human label dataset** (`eval/datasets/judge_labels.json`)
Manually label 50 query-answer pairs sampled from existing eval results. Each
entry has:
```json
{
  "id": "cal_001",
  "query": "...",
  "answer": "...",
  "context_str": "...",
  "ground_truth": "...",
  "human_faithfulness": 0.9,
  "human_correctness": 0.8,
  "human_verdict": "accept"
}
```
Binarize at threshold 0.8 for F1 computation (faithful/not-faithful).

**Step 2 — JudgeCalibrator** (`eval/calibrate_judge.py`)
Runs `LLMJudge.score()` on all 50 labelled entries; collects `(human_score,
judge_score)` pairs. Computes:
- Binary precision, recall, F1 (binarize at 0.8 threshold for both)
- Pearson r (continuous agreement)
- Mean absolute error
- Confusion matrix (TP, FP, TN, FN counts)
- Per-category breakdown (billing vs. cross-source etc.)

**Step 3 — Baseline and drift detection**
Save calibration results to `eval/results/judge_calibration.json`. On subsequent
runs, load the baseline and flag drift: `if current_f1 < baseline_f1 - 0.05:
exit(1)`.

**Step 4 — Rich report output**
Use `rich.Table` to print calibration results (same style as existing
`print_report()` in `evaluator.py`).

### Files to create or modify

| File | Action | Purpose |
|---|---|---|
| `eval/calibrate_judge.py` | Create | JudgeCalibrator + CLI |
| `eval/datasets/judge_labels.json` | Create | 50 human-labelled examples |
| `eval/results/judge_calibration_baseline.json` | Create | Committed baseline |

### API interfaces

```bash
# Run calibration
python -m eval.calibrate_judge

# Update baseline
python -m eval.calibrate_judge --update-baseline

# Use a different judge model
python -m eval.calibrate_judge --judge-model gpt-4o
```

### Evaluation strategy
Meta: the calibration harness measures the quality of the evaluator itself.
- Initial run should show F1 >= 0.85 on the human-labelled set (acceptance
  criterion from `project_use_cases.md`).
- Simulate judge degradation by replacing `gpt-4o-mini` with a constant-score
  stub; confirm calibration harness reports low F1 and exits 1.

### Implementation steps for Claude Code
1. Manually annotate 50 query-answer pairs from `eval/results/` (reuse existing
   eval runs — choose 8–9 entries per category).
2. Implement `eval/calibrate_judge.py` with `JudgeCalibrator.run()` and CLI.
3. Run once to produce baseline; commit `judge_calibration_baseline.json`.
4. Add calibration run to `.github/workflows/eval_ci.yml` (runs weekly, not on
   every PR — it costs ~$0.05).

### Estimated complexity: LOW-MEDIUM (2–3 days)
### Dependencies: Feature 2 (shares CI infrastructure)
### Tests
- Unit test: stub `LLMJudge` to return known scores; verify calibration metrics.

---

## FEATURE 6 — Council Orchestrator (3-Agent Pattern)

### Problem it solves
Single-agent RAG answers are unchecked. For high-risk MSP decisions (contract
renewals, overdue escalations, billing disputes), a second opinion from a
different reasoning posture (creative vs. conservative) and a policy verifier
improves accuracy and reduces hallucination risk.

### System components required
- `CouncilOrchestrator` class
- Three agent roles: `FastCreativeAgent`, `ConservativeCheckerAgent`,
  `PolicyVerifierAgent` (each wraps a different LLM or prompt posture)
- `CouncilVerdict` dataclass (accepted answer + dissent summary)
- Deadlock detector (cycle/retry detection)
- Integration with `TraceCollector` for handoff observability
- CLI demo: `python -m src.agents.council --query "..."`

### Detailed implementation plan

**Step 1 — Agent base** (`src/agents/base.py`)
`BaseAgent(name, generator, system_prompt)` with one method:
`propose(query, context_bundle) -> AgentProposal(answer, confidence, reasoning)`.

**Step 2 — Agent roles** (`src/agents/roles.py`)
- `FastCreativeAgent`: low temperature (0.9), system prompt emphasising creative
  synthesis, uses `gpt-4o-mini`.
- `ConservativeCheckerAgent`: temperature 0.1, system prompt emphasising factual
  conservatism and citation grounding, uses `gpt-4o-mini`.
- `PolicyVerifierAgent`: uses the verifier prompt from `project_use_cases.md`
  (checks grounding + PII + authorization), uses `claude-haiku-4-5-20251001`.
  Returns `{"verdict": "accept|reject|escalate", "reasons": [...]}`.

**Step 3 — CouncilOrchestrator** (`src/agents/council.py`)
```
1. Run HybridRetriever + ContextManager to get shared ContextBundle.
2. Dispatch FastCreative and ConservativeChecker in parallel (asyncio.gather).
3. Pass both proposals + context to PolicyVerifier.
4. If PolicyVerifier accepts one proposal: return it.
5. If PolicyVerifier rejects both: escalate (return structured escalation message).
6. If PolicyVerifier requests clarification (escalate): retry up to 1 time with
   an augmented query; then escalate if still unresolved.
```

**Step 4 — DeadlockDetector** (`src/agents/deadlock.py`)
Maintains a retry counter per `(query_hash, session_id)` in memory (or Redis if
available). If retries >= 2, raises `CouncilDeadlock` exception which the
orchestrator catches and converts to a `CouncilVerdict(escalated=True)`.

**Step 5 — Pattern catalog** (`src/agents/patterns/`)
YAML definitions and example entry points:
- `sequential.yaml` + `sequential_runner.py`
- `hierarchical.yaml` + `hierarchical_runner.py`
- `council.yaml` + `council_runner.py` (the feature above)

**Step 6 — CLI demo** (`src/agents/council_cli.py`)
```bash
python -m src.agents.council_cli --query "Should we escalate Alpine Financial?"
```
Prints proposals from all 3 agents + verifier verdict + whether escalation was
triggered.

### Files to create or modify

| File | Action | Purpose |
|---|---|---|
| `src/agents/__init__.py` | Create | Package |
| `src/agents/base.py` | Create | BaseAgent + AgentProposal |
| `src/agents/roles.py` | Create | FastCreative, Conservative, PolicyVerifier |
| `src/agents/council.py` | Create | CouncilOrchestrator |
| `src/agents/deadlock.py` | Create | DeadlockDetector |
| `src/agents/patterns/council.yaml` | Create | Council pattern spec |
| `src/agents/patterns/sequential.yaml` | Create | Sequential pattern spec |
| `src/agents/council_cli.py` | Create | CLI demo |
| `app/server.py` | Modify | Add POST /api/council endpoint |

### API interfaces

```python
class CouncilOrchestrator:
    async def run(
        self,
        query: str,
        ctx: ABACContext,
        budget_tokens: int = 3000,
    ) -> CouncilVerdict: ...

@dataclass
class CouncilVerdict:
    accepted_answer: str
    winning_agent: str
    dissent_summary: str
    escalated: bool
    policy_reasons: list[str]
    total_cost_usd: float
    trace_id: str
```

```
POST /api/council
Body: {"message": "...", "user_role": "finance"}
Response: CouncilVerdict as JSON
```

### Security considerations
- Each agent call goes through the ASG (Feature 1): policy is evaluated per-
  agent, not once per council run.
- PolicyVerifier runs last and has `role=verifier` in ABAC context; only it may
  return `"accept"` decisions.
- Escalated verdicts are automatically added to the trace store with
  `verdict=escalated` for human review.

### Latency considerations
- FastCreative and ConservativeChecker run in parallel (asyncio.gather); wall
  time = max(fast, conservative) not sum. Target p95 parallel step < 1,500 ms.
- PolicyVerifier runs serially after the parallel step (~500 ms).
- Total council p95: ~2,000 ms (vs. 1,200 ms single agent). Explicitly
  documented as a quality-latency tradeoff; expose via `mode=council|single` in
  the API.

### Context engineering strategy
- All three agents share the same `ContextBundle` (computed once before
  dispatch). This ensures proposals are grounded in the same retrieved facts.
- PolicyVerifier receives the proposals as structured JSON in its context window,
  not free text, to reduce hallucination of verification criteria.

### Evaluation strategy
- **Council accuracy uplift**: run the 80-query eval set with `mode=council` and
  compare composite score vs. single-agent baseline. Target >= 5% uplift on
  `cross_source` category (the hardest, most multi-hop).
- **Coordination failure rate**: run 20 synthetic adversarial queries designed to
  produce conflicting proposals; confirm deadlock detector fires and escalation
  rate < 5%.
- **Policy verifier precision**: run 30 council outputs where one proposal
  contains a hallucinated claim; confirm PolicyVerifier rejects >= 90%.

### Implementation steps for Claude Code
1. Create `src/agents/` package; implement `base.py`, `roles.py`.
2. Implement `council.py` with async parallel dispatch.
3. Implement `deadlock.py` with in-memory retry counter.
4. Create `src/agents/patterns/` with YAML catalog.
5. Implement `council_cli.py`.
6. Add `POST /api/council` to `app/server.py`.
7. Write 5 council scenario tests (mix of accept, reject, escalate outcomes).

### Estimated complexity: MEDIUM-HIGH (7–10 days)
### Dependencies: Feature 1 (ASG for per-agent ABAC), Feature 3 (ContextManager),
                  Feature 4 (TraceCollector for handoff logging)

---

# TIER 3 — Advanced / Research Features

---

## FEATURE 7 — Edge Runtime Emulator + Latency SLO Harness

### Problem it solves
MSP field technicians operate in environments with intermittent connectivity.
There is no tested offline-capable agent path, no SLO measurement framework, and
no emulator to develop/test local decision-making without a live API connection.

### System components required
- Docker-based edge emulator with network partition simulation
- Local policy engine (subset of ASG Feature 1, no API calls)
- Local compact vector store (FAISS only, no reranker, pre-loaded index)
- Cloud-fallback stub (simulates reconnection after offline period)
- Latency SLO harness: measures p50/p95 for local-only vs. cloud-assisted paths
- Offline job queue (sensor alert → local triage → queued upload)

### Detailed implementation plan

**Step 1 — Dockerized emulator** (`docker/edge/`)
`Dockerfile.edge` based on `python:3.11-slim`. Copies:
- `data/index/faiss.index` + `data/index/chunks.json` (pre-built index)
- `config/policies.yaml` + local edge policy subset
- `src/context/`, `src/retrieval/retriever.py`, `src/embedding/`
- `src/security/` (policy engine only, no audit logger network calls)
- A minimal FastAPI server `edge/server.py` exposing `POST /edge/query`

`docker-compose.edge.yml`: two services — `edge` (the emulator) and
`toxiproxy` (for network fault injection: latency, packet loss, partition).

**Step 2 — Local RAG pipeline** (`edge/local_pipeline.py`)
`EdgeRAGPipeline` — strips out `LLMReranker` (requires OpenAI), uses only FAISS
search + `ContextManager` with `budget_tokens=512`. Generation uses a local
Anthropic model call (if connected) or returns a canned response stub (if
offline). The `offline_safe` parameter switches between these modes.

**Step 3 — Latency SLO harness** (`edge/latency_harness.py`)
Runs 100 synthetic queries through:
- `local_only` path (FAISS + ContextManager + stub LLM): measure round-trip ms
- `cloud_assisted` path (full pipeline via the emulated network): measure RTT
  including simulated latency

Outputs p50/p95/p99 and compares to SLO targets:
- Local decision p95 <= 250 ms
- Cloud-assisted p95 <= 1,200 ms

**Step 4 — Offline job queue** (`edge/job_queue.py`)
Simple SQLite-backed queue. `EdgeJob(type, payload, status, queued_at)`.
Sensor alert handler: `enqueue_job(type="sensor_alert", payload=event)`.
Background sync worker: when connectivity is restored (health-check loop),
drain queue and POST to central `/api/ingest`.

**Step 5 — Offline scenario tests** (`edge/test_scenarios.py`)
10 scripted scenarios: sensor fires during partition, triage runs locally,
job queues, connectivity restores, job syncs, answer matches cloud version.

### Files to create or modify

| File | Action | Purpose |
|---|---|---|
| `docker/edge/Dockerfile.edge` | Create | Edge container |
| `docker/edge/docker-compose.edge.yml` | Create | Edge + toxiproxy |
| `edge/__init__.py` | Create | Package |
| `edge/local_pipeline.py` | Create | EdgeRAGPipeline |
| `edge/server.py` | Create | Minimal FastAPI for edge |
| `edge/latency_harness.py` | Create | SLO benchmark |
| `edge/job_queue.py` | Create | SQLite offline queue |
| `edge/test_scenarios.py` | Create | 10 offline scenarios |

### Latency SLO targets
| Path | p50 | p95 |
|---|---|---|
| Local-only (FAISS + stub) | <= 80 ms | <= 250 ms |
| Cloud-assisted (full pipeline) | <= 600 ms | <= 1,200 ms |

### Estimated complexity: HIGH (10–14 days)
### Dependencies: Feature 3 (ContextManager for budget-constrained local path),
                  Feature 4 (TraceCollector for offline trace upload)

---

## FEATURE 8 — Tamper-Evident Immutable Audit Log (Production Hardening)

### Problem it solves
Feature 1 introduces HMAC-signed audit entries but the file-based JSONL approach
is not tamper-evident in a multi-process or multi-machine environment. For
compliance (SOC 2, HIPAA-adjacent MSP contracts), audit entries need a
cryptographic chain that can be externally verified.

### System components required
- Hash-chained audit log (each entry's HMAC includes the previous entry's hash)
- Merkle-tree verifier for bulk log verification
- Export endpoint: `GET /api/audit/export?from=T1&to=T2` (signed, paginated)
- Retention policy enforcer (TTL-based purge with proof-of-deletion receipt)
- SIEM integration stub (forwards events to Splunk/Datadog webhook)

### Detailed implementation plan

**Step 1 — Hash-chained entries** (`src/security/audit_logger.py` — extend)
Each entry includes a `prev_hash` field (SHA-256 of the previous raw entry
bytes). Entry 0 has `prev_hash = "genesis"`. Verification walks the chain and
confirms each `prev_hash` matches `sha256(prev_entry_bytes)`.

**Step 2 — Merkle verifier** (`src/security/merkle.py`)
Build a Merkle tree over all entry hashes in a time window. Store the root hash
in `data/audit/merkle_roots.jsonl`. An external auditor can verify any single
entry against the published root without downloading the full log.

**Step 3 — Export + retention** (`app/server.py` extension)
`GET /api/audit/export` returns entries as signed JSON. Retention policy purges
entries older than `config.audit.retention_days` (default 365) and writes a
`PurgeCertificate(entries_deleted, merkle_root_at_purge, timestamp, sig)`.

**Step 4 — SIEM webhook** (`src/security/siem_forwarder.py`)
On every high-severity event (guardrail block, policy deny, PII access), POST to
`SIEM_WEBHOOK_URL` from `.env`. Async, non-blocking, with retry.

### Estimated complexity: HIGH (8–12 days)
### Dependencies: Feature 1 (base audit logger)

---

## FEATURE 9 — Voice-First Agent Mini-Pipeline

### Problem it solves
MSP field technicians need hands-free access to TechVault RAG from mobile
devices. A voice interface (speech-to-text → RAG → text-to-speech) is a strong
differentiator but introduces new latency, PII (audio), and accessibility
requirements.

### System components required
- VAD (Voice Activity Detection) — WebRTC-based in-browser
- STT (Speech-to-Text) — OpenAI Whisper API
- RAG query path (existing pipeline)
- TTS (Text-to-Speech) — OpenAI TTS API
- Pronunciation lexicon (`config/lexicon.yaml` for MSP terms: FAISS, SLA, etc.)
- Latency meter (VAD end → TTS first byte p95)
- Privacy: audio retention TTL = 0 (never store audio), consent capture in trace

### Detailed implementation plan

**Step 1 — Browser VAD + STT** (`app/static/voice.html`)
Use the Web Audio API + a minimal VAD (silence detection) to capture audio
segments. POST audio blobs to `POST /api/voice/transcribe` (Whisper API proxy).

**Step 2 — Server STT endpoint** (`app/server.py`)
`POST /api/voice/transcribe` — receives audio blob, calls `openai.audio.
transcriptions.create(model="whisper-1")`, applies pronunciation post-correction
from lexicon, returns transcribed text. Audio blob is never written to disk.

**Step 3 — TTS endpoint** (`app/server.py`)
After RAG generates an answer, `GET /api/voice/speak?answer_id=X` calls
`openai.audio.speech.create(model="tts-1", voice="alloy")` and streams the
MP3 response. No caching (consent/TTL = 0).

**Step 4 — Latency meter** (`app/static/voice.html`)
JavaScript: measure `t_vad_end` (last audio sample) to `t_tts_first_byte`
(first byte of MP3 stream). Log to `POST /api/metrics/latency` and display
p95 in a debug overlay.

**Step 5 — Privacy controls**
Trace metadata for voice queries includes `consent_captured=true` flag (set by
UI checkbox before microphone access). Audio is never stored. PII filter applies
to the RAG answer before TTS synthesis.

### SLO targets
| Segment | p95 target |
|---|---|
| STT (Whisper API RTT) | <= 500 ms |
| RAG pipeline (fast path) | <= 400 ms |
| TTS first byte | <= 300 ms |
| **End-to-end** | **<= 1,200 ms** |

### Estimated complexity: MEDIUM-HIGH (7–10 days)
### Dependencies: Feature 3 (fast path for 400 ms RAG), Feature 4 (trace
                  `consent_captured` field)

---

# Feature Dependency Graph

```
Feature 1 (ASG / Policy Engine)
    |
    +-- Feature 2 (Gold-Task CI) ........... no code dep, shares CI infra
    |
    +-- Feature 4 (Trace Collector) ........ uses ABACContext in trace
    |       |
    |       +-- Feature 6 (Council) ........ uses TraceCollector for handoffs
    |
    +-- Feature 6 (Council) ............... per-agent ABAC enforcement
    |
    +-- Feature 8 (Immutable Audit) ........ extends Feature 1 audit logger

Feature 3 (ContextManager)
    |
    +-- Feature 6 (Council) ............... shared ContextBundle
    |
    +-- Feature 7 (Edge Runtime) ........... budget-constrained local path
    |
    +-- Feature 9 (Voice Pipeline) ......... fast path for latency SLO

Feature 5 (Judge Calibration) .............. standalone, depends on eval/
```

---

# Summary Table

| # | Feature | Tier | Complexity | Days | Depends on |
|---|---|---|---|---|---|
| 1 | Policy-Driven Agent Security Gateway | 1 | Low-Med | 3–5 | — |
| 2 | Gold-Task CI Runner | 1 | Low | 2–3 | — |
| 3 | ContextManager SDK | 1 | Low-Med | 3–4 | — |
| 4 | Trace Collector + Failure-Biased Sampler | 2 | Medium | 5–7 | F1 |
| 5 | LLM-Judge Calibration Harness | 2 | Low-Med | 2–3 | — |
| 6 | Council Orchestrator | 2 | Med-High | 7–10 | F1, F3, F4 |
| 7 | Edge Runtime Emulator + Latency SLO | 3 | High | 10–14 | F3, F4 |
| 8 | Tamper-Evident Immutable Audit Log | 3 | High | 8–12 | F1 |
| 9 | Voice-First Agent Mini-Pipeline | 3 | Med-High | 7–10 | F3, F4 |

**Recommended sprint order:**
- Sprint 1 (Phase A): Features 2 → 1 → 3 (CI gate first, then security, then context)
- Sprint 2 (Phase B): Features 5 → 4 → 6 (calibrate judge, add tracing, build council)
- Sprint 3 (Phase C): Features 7 → 8 → 9 (edge, audit hardening, voice)

---

# Cross-Cutting Non-Negotiables (all features)

Every feature implementation must satisfy:

1. **Security by design**
   - ABAC attributes evaluated at call time (not cached across requests)
   - All tool calls logged to audit store
   - No PII in logs, traces, or audit entries (PIIFilter applied before write)

2. **Latency SLOs declared and tested**
   - Synchronous fast path (ContextManager `budget_tokens=1024`) must be under
     400 ms total for Tier 1 features
   - Async heavy path (full reranking + council) documented at p95 with test

3. **Context discipline**
   - All features use `ContextManager.get_context()` rather than raw chunk lists
   - Progressive disclosure: working tier data always placed at context boundaries
   - Memory writes (future longterm tier) require PII tag + ACL before persist

4. **Eval integration**
   - Every new feature writes a metric to the eval harness or trace store
   - Feature 2 (Gold-Task CI) gates every PR; new features must not lower
     baseline composite score
   - Council and edge features have their own eval categories added to
     `eval/datasets/`
