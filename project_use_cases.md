# Use cases and Feature Roadmap — Agentic Security, Latency, Context Engineering & Eval

This markdown file translates the conference "business takeaways" into a set of concrete use cases and implementable demo features for Claude Code to develop. Each use case emphasizes **security**, **latency**, **context engineering**, and **agentic evals** and contains:

- A short value proposition
- Concrete integration points with the existing TechVault RAG pipeline (CLAUDE.md)
- Demo features for Claude Code to implement
- Acceptance criteria, KPIs, and a minimal test plan

---

## 1) Agent Quality & Evaluation Platform (Core)

**Why**: Evaluation must be first-class to avoid silent regressions, measure multi-agent coordination, and enforce deployment gates.

**High-level idea**
- Build a modular eval harness that can run *offline unit tests* (gold tasks), *workflow tests* (multi-step), and *production monitors* (sampled/triggered checks). Include LLM-as-judge support but require calibrated human audits for high-risk categories.

**Integration points**
- Reuse `eval/run_eval` infrastructure in CLAUDE.md Phase `eval/` and extend dataset schema to include multi-agent orchestration cases and security-oriented tests (prompt-injection, tool misuse). (phase: `eval.run_eval`).

**Demo features for Claude Code**
1. **Gold-task CI runner**: CLI + GitHub Actions job that runs a small selection of gold tasks on PRs and fails the PR on regressions.
2. **Multi-agent scenario runner**: Orchestrates 2–3 mock agents (planner, executor, verifier) against scripted workflows and measures handoff success and deadlocks.
3. **LLM-judge calibration harness**: Runs judge predictions vs human labels and reports judge calibration (precision/recall, confusion matrix).

**Acceptance criteria & KPIs**
- Regression gate: failing >1% of gold tasks triggers `exit 1` for CI.
- Judge calibration: judge F1 >= 0.85 on holdout set before allowing automated passes.
- Multi-agent success: handoff success >= 90% on the demo scenarios.

**Minimal test plan**
- Provide 20 gold tasks covering billing/contract queries (reuse `eval/datasets/*.json`).
- Add 5 multi-agent scenarios for sequential and council patterns.

---

## 2) Agent Security Gateway (Policy + Runtime Enforcement)

**Why**: Agents + tools dramatically expand attack surface (prompt injection, data exfiltration). Enforceable gateway policies are required.

**High-level idea**
- Implement a centralized Agent Security Gateway (ASG) sitting between agents and tools: enforces allowlists, ABAC checks, step-up auth, token budgets, and immutable audit logs.

**Integration points**
- Intercept calls from `MCP server` (`src/collection/mcp/server.py`) and `RAGGenerator` tool calls; record tool call events in a traceable log store.

**Demo features for Claude Code**
1. **Policy engine prototype**: simple policy DSL (YAML) implementing allowlists, required justification fields, and ABAC attribute checks (user.role, data.classification, env).
2. **Prompt-guard integration**: pre-flight prompt scanner that blocks common injection patterns and returns `GuardrailResult(passed=False)` with reasons.
3. **Audit logger**: tamper-evident append-only log (signed entries) for all tool calls that can be exported for compliance.

**Security-specific acceptance criteria**
- Gateway blocks a curated suite of prompt-injection tests (100 cases) with true positive >= 98% and false positive <= 2% on a benign sample.
- All sensitive tool calls require ABAC attributes; missing attribute → blocked.
- Audit logs are append-only and include SHA-256 of call + signer.

**KPIs**
- Prompt injection attempt blocked rate
- % tool calls with ABAC evaluation
- Time-to-approve sensitive tool action (p95)

---

## 3) Context-Optimized RAG & Memory (Cost + Accuracy)

**Why**: Context is the new bottleneck — both for model accuracy (“lost in the middle”) and for token economics.

**High-level idea**
- Implement progressive-disclosure context engineering: a **working context** (small, immediate), a **retrieval layer** (hybrid FAISS+BM25), and a **long-term memory** (tiered, ACL governed). Use dynamic chunking and pruning heuristics.

**Integration points**
- AdaptiveChunker and HybridRetriever (CLAUDE.md Phase II/III) — add metadata for context freshness, TTL, and vector decay.

**Demo features for Claude Code**
1. **Context Playbook SDK**: helpers to mark context items as `working|ephemeral|longterm`, and an API `ContextManager.get_context(query, budget_tokens)` that returns a cost-optimized bundle.
2. **Progressive disclosure UI**: show which context pieces were included, explain why (relevance score + freshness), and allow “request more context” button.
3. **Memory safety layer**: automated PII tagging and redaction policy for long-term memory writes (configurable retention & ACLs).

**Latency & cost considerations**
- Budget token parameter ensures low-latency fallbacks; for p95 low-latency queries, set `budget_tokens <= 1024`. Implement synchronous path for working-context-only queries and async path for heavy retrieval.

**Acceptance criteria & KPIs**
- Tokens per successful task reduced by >= 30% vs naive full-context baseline on demo corpus.
- Retrieval precision: % retrieved context cited in answers >= 80%.
- Memory leak incidents = 0 in testing (test suite with synthetic PII).

---

## 4) Low-Latency Edge Agent (IoT / Field Ops Demo)

**Why**: Latency-sensitive workflows require edge or hybrid deployments and explicit fallback logic.

**High-level idea**
- Build a lightweight agent runtime for edge devices that can operate offline with a compact policy model and perform safe local actions; cloud fallback handles heavier reasoning or audit logs upload.

**Integration points**
- Implement a simplified RAG/ContextManager that runs on a tiny vector store + small local model (or distilled system) and syncs traces to central observability.

**Demo features for Claude Code**
1. **Edge runtime emulator**: Dockerized emulator that simulates intermittent connectivity; includes local policy engine and local fallback prompts.
2. **Latency SLO harness**: synthetic tests measuring round-trip latency p50/p95 from edge→cloud and decision latency local-only vs cloud-assisted.
3. **Offline-safe job types**: create a sample workflow (sensor alert → local triage → queued upload) demonstrating correctness under partition.

**Latency acceptance criteria**
- Local decision latency p95 <= 250ms for defined critical actions.
- Cloud-assisted decisions p95 <= 1,200ms under good connectivity.

**KPIs**
- Offline success rate (tasks completed correctly under partition)
- Extra cloud calls saved per task (cost reduction)

---

## 5) Multi-Agent Orchestration Patterns + Council Demo

**Why**: Single agents fail at scale; patterns (sequential, hierarchical, council) improve reliability and allow specialization.

**High-level idea**
- Provide a pattern library and a runnable council demo where multiple agents propose answers and a verifier agent adjudicates.

**Integration points**
- Use the MCP server tool framework to spawn isolated agent instances; trace all handoffs into the observability store.

**Demo features for Claude Code**
1. **Council orchestrator**: 3 agents (fast-creative, conservative-checker, policy-verifier) submit candidate answers; a verifier uses majority + policy checks to accept or ask for clarification.
2. **Deadlock detector**: small scheduler that detects cycles/retries and raises a recoverable exception + human escalation.
3. **Pattern catalog**: YAML definitions and example code for sequential, hierarchical, and council patterns.

**Evaluation & KPIs**
- Council accuracy uplift vs single agent (target >= 10% for high-risk question set).
- Coordination failure rate < 5% in demo runs.

---

## 6) Observability & Tail-Capture for Agent Runs

**Why**: Trace sampling misses rare but critical failures in long-horizon workflows.

**High-level idea**
- Implement a trace schema for agents (redacted prompt, plan steps, tool calls, approvals, costs, and verdict). Use failure-biased capture rules and a small indexed trace store.

**Integration points**
- Hook into `RAGPipeline.query()` and MCP tool calls; export traces to a searchable store and link traces to eval runs.

**Demo features for Claude Code**
1. **Trace Schema + Collector**: JSON schema + small collector service that stores case-file traces.
2. **Failure-biased sampler**: config-driven rules to capture all runs with (error OR cost > threshold OR PII access OR policy-block).
3. **Trace replay tool**: re-run a trace deterministically in an isolated environment for debugging.

**Acceptance criteria**
- Incident MTTR reduced in demo runs by >= 30% vs no-trace baseline.
- Failure-biased sampler captures ≥95% of synthetic critical failures.

---

## 7) Voice-First Agent Mini-Pipeline (Optional)

**Why**: Voice-first flow introduces high UX and privacy requirements but is a strong differentiator for field and support use cases.

**Demo features for Claude Code**
- Modular pipeline (VAD → STT → RAG LLM → TTS) with a pronunciation lexicon and latency meter.
- Evaluate WER on key domain terms and measure end‑to‑end p95 latency.

**Privacy & Security**
- Enforce audio retention TTL and encryption at rest; require consent captures in trace metadata.

---

## Implementation Roadmap & Priorities (3-phase)

**Phase A — Core Controls & Eval (0–6 weeks)**
- Agent Security Gateway prototype + prompt-guard tests.
- Gold-task CI runner + judge calibration harness.
- Context Playbook SDK (working/ephemeral/longterm) and simple ContextManager API.

**Phase B — Orchestration & Observability (6–12 weeks)**
- Council orchestrator + pattern catalog.
- Trace collector + failure-biased sampler + replay tool.
- Progressive-disclosure UI for context visualization.

**Phase C — Edge / Voice / Production Hardening (12–24 weeks)**
- Edge runtime emulator + latency SLO harness.
- Voice mini-pipeline (if prioritized).
- Harden ASG for production: signed audit logs, ABAC integration with identity.

---

## Developer Handoff: Prompts, API sketches, and Test Data

**Prompt examples (for policy-verifier)**
```
SYSTEM: You are the policy verifier. Given a candidate answer and the associated tool calls, accept the answer only if: (1) all claims are grounded in provided citations, (2) no PII was leaked, (3) action requests are authorized by policy. Return JSON: {"verdict": "accept|reject|escalate", "reasons": [...]}.
```

**ContextManager API sketch**
```py
class ContextManager:
    def get_context(query: str, budget_tokens: int) -> List[ContextPiece]:
        """Returns ranked context pieces with metadata: {id, text, tokens, freshness, relevance} """
```

**Policy DSL example (YAML)**
```yaml
policies:
  - id: allow_billing_read
    action: read_billing
    abac:
      required_attrs: [user.role, data.classification]
      rules:
        - if: "data.classification == 'sensitive' and user.role != 'finance'"
          effect: deny
```

**Minimal test datasets**
- 20 gold queries from `eval/datasets` (billing/contract/psa) for acceptance.
- 100 crafted prompt-injection strings for gateway testing.
- 10 edge-offline scenarios (sensor events) for latency/emulation.

---

## Quick wins for Claude Code to deliver in 1–2 sprints

1. **Gold-task CI runner** (CI + 20 tasks)
2. **Prompt-guard + policy DSL prototype** (gateway blocking tests)
3. **ContextManager.get_context() + progressive-disclosure UI** (small web UI)
4. **Council orchestrator demo with 3 agents**

---

## Final notes

This document is intentionally implementation-focused so Claude Code can convert ideas into PR-sized features. The three cross-cutting non-negotiables for every feature are:
- **Security by design** (ABAC, audit logs, prompt-guard)
- **Latency SLOs** (explicit budgets and sync/async paths)
- **Context discipline** (progressive disclosure, memory ACLs)
- **Eval integration** (every feature writes to the eval harness or trace store so behavior can be measured and regressed)

If you want, I can also generate initial seed code for the Gold-task runner, a policy-engine skeleton, or the ContextManager API next.

