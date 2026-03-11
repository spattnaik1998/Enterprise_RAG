# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**TechVault MSP Enterprise RAG Intelligence Hub** — a 6-stage RAG pipeline (collect → clean → chunk → embed → index → serve) over both AI/ML research sources and synthetic MSP back-office data (billing, PSA, CRM, comms, contracts).

## Pipeline Commands

```bash
# One-time: generate synthetic enterprise data (seed=42, run once)
python scripts/generate_enterprise_data.py

# Phase I — collect from all 8 sources + validate
python -m src.main phase1
python -m src.main phase1 --dry-run          # validate config only
python -m src.main phase1 --skip-health      # skip source health checks

# Phase II — chunk, embed (OpenAI), build FAISS + BM25 index
python -m src.main phase2

# Phase III — CLI query interface (interactive or single-shot)
python -m src.main phase3
python -m src.main phase3 --query "Which clients have overdue invoices?"

# Check phase checkpoint state
python -m src.main status

# Phase III — Web UI + REST API (run from project root; uses Supabase index)
uvicorn app.server:app --reload --port 8000

# Run MCP server standalone (for Claude Desktop or custom MCP clients)
python -m src.collection.mcp.server

# Council Orchestrator — 3-agent voting CLI
python -m src.agents.council_cli --query "Should we escalate Alpine Financial?"
python -m src.agents.council_cli  # interactive mode

# Migrate local FAISS + enterprise JSON data to Supabase (idempotent)
python scripts/migrate_to_supabase.py
python scripts/migrate_to_supabase.py --only chunks  # partial re-run
```

## Evaluation Framework

```bash
# Smoke test: 1 model, billing queries, 5 samples (~$0.01)
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5

# Full production eval: all 4 models, all 80 queries (~$2.82)
python -m eval.run_eval

# Multi-model, multi-category (space-separated or repeated flags both work)
python -m eval.run_eval --models gpt-4o-mini gpt-4o --category contracts crm

# Skip reranking for rapid iteration
python -m eval.run_eval --no-rerank --models gpt-4o-mini --sample 5

# CI/CD integration (exit 0 = all pass, exit 1 = any fail)
python -m eval.run_eval --models gpt-4o-mini && echo "PASS" || echo "FAIL"
```

### Eval CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--models TEXT` | all 4 | Models to test (space-separated or repeat flag) |
| `--category TEXT` | all 6 | Categories: `billing`, `contracts`, `crm`, `psa`, `communications`, `cross_source` |
| `--sample INT` | 0 (all) | Queries per category to sample; 0 = use all |
| `--no-rerank` | off | Skip LLM reranking (faster, cheaper, lower quality) |
| `--output PATH` | auto-timestamped | JSON report path (saved to `eval/results/`) |
| `--judge-model TEXT` | `gpt-4o-mini` | OpenAI model used as LLM judge |
| `--top-k INT` | 20 | Candidates retrieved before reranking |
| `--rerank-top-k INT` | 10 | Chunks kept after reranking |
| `--index-dir PATH` | `data/index` | FAISS index directory |
| `--seed INT` | 42 | RNG seed for reproducible sampling |
| `--quiet` | off | Suppress progress bar |

### Production Thresholds (all must pass)

| Metric | Threshold | What it measures |
|---|---|---|
| Recall@10 | >= 80% | Any `expected_keyword` found in top-10 citation titles/sources or answer text |
| Source Type Hit | >= 85% | Any citation `source_type` matches `expected_source_types` |
| Faithfulness | >= 85% | LLM judge: all claims grounded in retrieved context |
| Correctness | >= 75% | LLM judge: answer matches ground truth |
| Composite | >= 82% | Mean of the four metrics above |

### Eval Dataset

80 queries across 6 categories in `eval/datasets/*.json`. Query schema:
```json
{
  "id": "billing_001",
  "query": "...",
  "ground_truth": "...",
  "expected_source_types": ["billing"],
  "expected_keywords": ["ClientA", "ClientB"],
  "difficulty": "easy|medium|hard"
}
```

### Interpreting Failing Metrics

| Failing metric | Likely cause | Fix |
|---|---|---|
| Recall@10 < 80% | Index quality | Review chunking strategy or re-run Phase II |
| Source Type Hit < 85% | RRF weight imbalance | Tune `retrieval.dense_weight` in `config/config.yaml` |
| Faithfulness < 85% | Model hallucinating | Review system prompt in `src/generation/prompts.py` |
| Correctness < 75% | Reranking quality | Increase `rerank_top_k` |
| Cross-source queries low | Multi-hop retrieval weakness | Increase `top_k` |

### Eval Cost Estimates

| Scope | Models | Queries | Cost |
|---|---|---|---|
| Smoke test | gpt-4o-mini | 5 (billing) | ~$0.01 |
| Single model | gpt-4o-mini | 80 (all) | ~$0.13 |
| Full eval | All 4 models | 80 each | ~$2.82 |

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env` and fill in all required keys (see table below)
3. Run `python scripts/generate_enterprise_data.py` (creates `data/enterprise/*.json`)
4. Run phases in order: `phase1` → review `data/validated/` → `phase2`

### Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Embeddings (Phase II), reranker, OpenAI generation |
| `ANTHROPIC_API_KEY` | Yes | Claude generation (Haiku, Sonnet) |
| `SUPABASE_URL` | Yes (web) | PostgreSQL + pgvector backend |
| `SUPABASE_SERVICE_KEY` | Yes (web) | **Must be `service_role` JWT**, not `anon` key — anon key fails all inserts due to RLS |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `LANGSMITH_TRACING` | Optional | Set to `true` to enable traces |
| `LANGSMITH_PROJECT` | Optional | Default: `"Enterprise RAG"` |

**Windows note**: The codebase patches `sys.stdout/stderr` to UTF-8 at startup in `src/main.py`, `src/embedding/pipeline.py`, `app/server.py`, `eval/run_eval.py`, and `src/agents/council_cli.py`. Any new entry points on Windows need the same reconfigure block at the top.

**TimeFM note**: The forecasting feature requires a local build of TimeFM. The server returns a 503 with install instructions if the package is missing. NLTK punkt tokenizer is auto-downloaded on the first Phase II run.

**Testing note**: `pytest` and `pytest-asyncio` are in `requirements.txt` but no test suite exists yet in the project. There is no `tests/` directory to run.

## Architecture

### Data Flow (per phase)

**Phase I** (`src/collection/` → `src/validation/`):
- `CollectionPipeline` (`src/collection/pipeline.py`) runs 8 collectors concurrently via asyncio
- Each collector extends `BaseCollector` and returns `list[RawDocument]`
- `DocumentValidator` runs 7 checks (min/max length, language, alpha ratio, boilerplate, dedup by SHA-256 checksum, quality score)
- `PhaseICheckpoint` saves state to `data/checkpoint_phase1.json` — pipeline stops here for human review
- Output: `data/validated/*.json`, `data/rejected/*.json`, `data/checkpoint_phase1.json`

**Phase II** (`src/chunking/` → `src/embedding/`):
- `AdaptiveChunker` selects strategy per document: `keep_whole` (structured short docs), `fixed_overlap` (CRM/medium), `sentence_window` (Wikipedia/ArXiv/RSS)
- `Embedder` calls OpenAI `text-embedding-3-small` in batches, L2-normalises output
- `FAISSIndex` stores `IndexFlatIP` (inner product = cosine on normalised vecs) + `BM25Okapi` corpus
- Output: `data/index/faiss.index`, `data/index/chunks.json`, `data/index/bm25_corpus.json`

**Phase III** (`src/retrieval/` + `src/generation/` + `src/serving/`):
- `RAGPipeline` in `src/serving/pipeline.py` orchestrates the full query lifecycle
- Two generator implementations with identical `generate(query, reranked_chunks)` interface: `RAGGenerator` (OpenAI) and `AnthropicGenerator`
- `pipeline.query(user_query, generator=None)` — optional `generator` arg lets callers swap the LLM at request time (used by the web API)
- **`LLMReranker` always uses OpenAI (`gpt-4o-mini`) regardless of which generator model the user selects for generation**
- CLI output: `data/index/` must exist (run Phase II first)

**Phase III Web UI** (`app/`):
- `app/server.py` — FastAPI app; startup loads `SupabaseIndex` then `RAGPipeline(index=supabase_index)` via `lifespan`
- `GET /api/health` — pipeline status, vector count, available models by provider
- `POST /api/chat` — accepts `{message, provider, model, session_id}`, swaps generator per request
- `GET /api/clients` — list all clients from invoice data (for forecasting UI)
- `GET /api/forecast/{client_id}?horizon=N` — TimeFM revenue forecast; `horizon` clamped to [1, 12], default 6
- `GET /api/logs` — paginated chat history (`?limit=50&offset=0`)
- `GET /api/logs/stats` — aggregated dashboard stats (calls `get_chat_stats()` RPC)
- `GET /` — serves `app/static/index.html` (Obsidian Terminal chat UI)
- `GET /forecast` — serves `app/static/forecast.html` (revenue forecasting visualization)
- `GET /logs` — serves `app/static/logs.html` (monitoring dashboard, auto-refreshes every 60s)
- Blocking `pipeline.query()` and forecast inference run in `loop.run_in_executor` to avoid stalling the async event loop
- Chat interactions are logged to Supabase `chat_logs` table via `app/chat_logger.py`; logging errors are swallowed (never degrade API response)

### Key Source Modules

| Module | Source system | Count |
|---|---|---|
| `ArXivCollector` | arxiv API | ~77 papers |
| `WikipediaCollector` | Wikipedia API | 8 topics |
| `RSSCollector` | 5 RSS feeds | ~433 entries |
| `BillingCollector` | QuickBooks (synthetic) | ~688 invoices |
| `PSACollector` | ConnectWise (synthetic) | ~732 tickets |
| `CRMCollector` | HubSpot (synthetic) | 50 profiles |
| `CommsCollector` | Exchange Online (synthetic) | ~142 emails |
| `ContractsCollector` | SharePoint (synthetic) | 50 contracts |

### MCP Server (17 tools)

Defined in `src/collection/mcp/server.py`. Exposes tools for research (arxiv, wikipedia, rss, web) and enterprise systems (billing AR, PSA tickets, CRM profiles, comms history, contracts, `get_client_360` cross-source aggregation).

### Core Schemas (`src/schemas.py`)

Three document states flow through the pipeline:
- `RawDocument` → collected, no quality guarantees
- `ValidatedDocument` → passed all 7 checks (adds `quality_score`, `validation_notes`)
- `RejectedDocument` → failed one or more checks (adds `rejection_reasons`)

All three share `checksum` (SHA-256), `word_count`, `char_count` as `@computed_field`. The `Chunk` dataclass (`src/chunking/schemas.py`) is produced by Phase II and flows through embedding, retrieval, reranking, and generation.

### Dual Index Mode (FAISS vs Supabase)

`FAISSIndex` (CLI/eval) and `SupabaseIndex` (web server) both implement the same `search_hybrid(query_vec, query_text, top_k, dense_weight, sparse_weight)` interface. `RAGPipeline.__init__` accepts `index=None` (loads local FAISS from disk) or `index=<SupabaseIndex>`. `HybridRetriever` type-hints `FAISSIndex` but satisfies duck-typing at runtime.

- **CLI / eval**: uses `FAISSIndex` (local `data/index/`) — no Supabase needed
- **Web server**: uses `SupabaseIndex` — connects to Supabase at startup, queries `ntotal` for health
- **RRF formula**: `weight / (rank + 60)`, dense_weight=0.7, sparse_weight=0.3

Supabase RPC functions: `match_chunks` (pgvector `<=>` cosine) and `search_chunks_fts` (PostgreSQL FTS). Both return TEXT ids (not UUID — columns were altered after initial creation).

**Common Supabase gotchas**:
- `chunk_id`/`doc_id` are TEXT not UUID — functions must declare TEXT return types or PostgREST throws error 42804
- Recreating a function without dropping all overloads causes PGRST203 (ambiguous). Always `DROP FUNCTION IF EXISTS` with full signature first.
- Batch size = 5 rows per insert (1536 floats × 5 ≈ 150KB, safely under PostgREST 1MB limit)

### Forecasting (`src/forecasting/`)

`InvoiceForecaster` uses **TimeFM 2.5 200M** (Google, via HuggingFace) to predict monthly revenue per client:
- Aggregates 13 months of invoice totals from `data/enterprise/invoices.json`
- Lazy-loads the ~800 MB PyTorch model on first call
- Returns historical actuals + point forecast + 10th/90th percentile bounds
- Called by `GET /api/forecast/{client_id}`; inference runs in thread-pool executor

### Sprint 2 Modules

**Agent Security Gateway** (`src/security/`):
- `AgentSecurityGateway` (`gateway.py`) — central enforcement: PromptGuard check → YAML policy evaluation → execute → HMAC-signed audit log
- `PolicyEngine` (`policy_engine.py`) — evaluates ABAC rules from `config/policies.yaml`; `ABACContext` carries `user_role`, `data_classification`, `username`
- `AuditLogger` (`audit_logger.py`) — append-only JSONL at `data/audit/audit.jsonl`; each entry HMAC-signed
- `@asg_tool(action, classification)` decorator — wraps MCP tool functions; extracts `_abac_ctx` kwarg or falls back to `ABACContext.anonymous()`
- Web server integrates via `gw.handle(query, abac_ctx, generator=generator, pipeline=pipeline)` instead of `pipeline.query()` directly

**ContextManager SDK** (`src/context/`):
- `ContextManager.get_context(query, chunks, budget_tokens=3000, fast_path=False)` — takes reranked chunks, scores by freshness + tier, greedily packs within token budget, reorders for "lost in the middle" mitigation
- `FreshnessScorer` (`freshness.py`) — scores chunks 0-1 based on metadata date fields
- `TierClassifier` (`tiers.py`) — classifies source_type into priority tiers; `reorder_for_lim()` puts highest-priority chunks at start and end
- Combined score: `relevance * (0.7 + 0.3 * freshness)`
- `fast_path=True` enforces 1024-token budget (latency optimisation)

**TraceCollector** (`src/observability/`):
- `TraceCollector` (`collector.py`) — context manager; records `TraceEvent` objects and writes sampled `AgentTrace` to `data/traces/`
- `FailureBiasedSampler` (`sampler.py`) — always samples failures; probabilistically samples successes
- `TraceStore` (`store.py`) — JSONL-per-session storage; `TraceReplayer` (`replayer.py`) for offline analysis
- Access active collector in nested calls: `get_active_collector()` (uses `ContextVar`, safe across async/threads)

**Council Orchestrator** (`src/agents/`):
- `CouncilOrchestrator` (`council.py`) — 3-agent voting: runs shared `HybridRetriever + ContextManager` once, then dispatches `FastCreativeAgent` and `ConservativeCheckerAgent` in parallel (`asyncio.gather`), then `PolicyVerifierAgent` reviews both
- `DeadlockDetector` (`deadlock.py`) — detects retry loops; retries once on "escalate", returns escalated `CouncilVerdict` if still unresolved
- `CouncilVerdict` — final output: `accepted_answer`, `winning_agent`, `dissent_summary`, `escalated`, `hallucination_detected`, `pii_concern`, `total_cost_usd`, `trace_id`
- CLI: `python -m src.agents.council_cli`

### Vercel Deployment

- `api/index.py` — Vercel entry point: `from app.server import app` (ASGI export)
- `vercel.json` — routes all traffic to `api/index.py`, `maxDuration: 60` (requires Pro plan)
- `requirements-vercel.txt` — serving layer only; excludes `faiss-cpu`, `rank-bm25`, `tiktoken`, `nltk`, `langdetect`, `faker`, `mcp`, `arxiv`, `wikipedia-api`, `feedparser`
- No FAISS binary at startup — Supabase connection only (~1-2s cold start vs ~10s+ with FAISS)
- Required env vars in Vercel settings: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`

### Observability

LangSmith tracing via `@traceable` on the full Phase III chain: `RAGPipeline.query()` (top-level), `HybridRetriever.retrieve()`, `LLMReranker.rerank()`, `RAGGenerator.generate()`, and `Embedder.embed_texts()` (Phase II). Requires `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` in `.env`.

## Configuration

All pipeline parameters live in `config/config.yaml`:
- `collection.*` — enable/disable sources, file paths, query lists
- `validation.*` — thresholds (min/max length, quality score, language)
- `chunking.*` — `max_tokens=512`, `keep_whole_threshold=600`, window sizes
- `embedding.*` — model, dimensions, batch size
- `retrieval.*` — `dense_weight=0.7`, `sparse_weight=0.3`, `top_k=20`, `rerank_top_k=10`

## Phase III Query Lifecycle

```
user query -> PromptGuard -> HybridRetriever -> LLMReranker -> RAGGenerator -> PIIFilter -> QueryResult
```

1. **PromptGuard** (`src/retrieval/guardrails.py`): 13 regex patterns covering jailbreaks, role overrides, instruction-ignoring. Returns `GuardrailResult(passed=False)` on hit.
2. **HybridRetriever** (`src/retrieval/retriever.py`): calls `Embedder.embed_query()` then `FAISSIndex.search_hybrid()` — RRF fusion of FAISS dense + BM25 sparse, returns top_k candidates.
3. **LLMReranker** (`src/retrieval/reranker.py`): single OpenAI call with all candidates in one prompt; scores 0-10 per chunk; sorts and truncates to rerank_top_k.
4. **RAGGenerator / AnthropicGenerator** (`src/generation/generator.py`): both build numbered `[1]..[N]` context from reranked chunks + `SYSTEM_PROMPT` from `src/generation/prompts.py`; return `RAGResponse` with answer + citations + token counts. Anthropic uses `system=` parameter (not inside messages list). Supported models: `gpt-4o-mini`, `gpt-4o`, `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`.
5. **PIIFilter** (`src/retrieval/guardrails.py`): regex-subs email/phone/SSN/CC/IP from generated answer with `[REDACTED_<TYPE>]`.

### Key Classes

| Class | File | Role |
|---|---|---|
| `RAGPipeline` | `src/serving/pipeline.py` | Orchestrator — loads index, wires all components |
| `CollectionPipeline` | `src/collection/pipeline.py` | Runs all 8 collectors concurrently via asyncio |
| `AdaptiveChunker` | `src/chunking/chunker.py` | Selects chunking strategy per document type |
| `FAISSIndex` | `src/embedding/faiss_index.py` | Local FAISS+BM25 index with RRF hybrid search (CLI/eval) |
| `SupabaseIndex` | `src/embedding/supabase_index.py` | pgvector + PostgreSQL FTS index (web/Vercel) |
| `HybridRetriever` | `src/retrieval/retriever.py` | Query embed + hybrid search (duck-types both indexes) |
| `LLMReranker` | `src/retrieval/reranker.py` | Batch LLM relevance scoring (always OpenAI) |
| `RAGGenerator` | `src/generation/generator.py` | OpenAI grounded answer synthesis |
| `AnthropicGenerator` | `src/generation/generator.py` | Anthropic grounded answer synthesis (same interface) |
| `PromptGuard` | `src/retrieval/guardrails.py` | Injection detection |
| `PIIFilter` | `src/retrieval/guardrails.py` | Output redaction |
| `PhaseICheckpoint` | `src/checkpoint.py` | Saves/loads Phase I state to disk |
| `QueryResult` | `src/serving/pipeline.py` | Full result dataclass (answer, citations, timings, cost) |
| `InvoiceForecaster` | `src/forecasting/invoice_forecaster.py` | TimeFM 2.5 monthly revenue forecasting |
| `AgentSecurityGateway` | `src/security/gateway.py` | Central ABAC policy + audit enforcement layer |
| `ContextManager` | `src/context/manager.py` | Budget-aware context assembly with freshness scoring |
| `TraceCollector` | `src/observability/collector.py` | Context-manager that records and samples pipeline traces |
| `CouncilOrchestrator` | `src/agents/council.py` | 3-agent voting (FastCreative + ConservativeChecker + PolicyVerifier) |

### Phase III CLI Flags

| Flag | Default | Effect |
|---|---|---|
| `--query TEXT` | None | Single-shot mode (omit for interactive loop) |
| `--top-k INT` | 10 | Candidates retrieved before reranking |
| `--rerank-top-k INT` | 5 | Chunks kept after LLM reranking |
| `--model TEXT` | gpt-4o-mini | OpenAI model for reranking + generation (CLI only; web UI selects per request) |
| `--no-rerank` | False | Skip LLM reranking (faster, cheaper) |
| `--no-pii` | False | Disable PII redaction |
| `--json` | False | JSON output (single-shot only) |

## Adding a New LLM Model

Two files must be updated together:
1. `src/generation/generator.py` — add entry to `_MODEL_PRICING` dict (input $/M, output $/M)
2. `app/server.py` — add model ID to `_OPENAI_MODELS` or `_ANTHROPIC_MODELS` set

The `ChatRequest` validator in `app/server.py` cross-checks provider/model pairs at request time, so both sets must be consistent.

## Enterprise Data Field Names

These differ from intuitive assumptions — consult the enterprise data files directly (`data/enterprise/`):
- **invoices.json**: `invoice_date`, `line_items[].amount` (not `line_total`)
- **psa_tickets.json**: `type` (not `ticket_type`), `title`, `technician`, `hours_billed`, `resolved_date`, `resolution_note` (string)
- **crm_profiles.json**: `account_health` = `RED`/`YELLOW`/`GREEN`, contacts = `{cfo, it_manager, ar_contact}`
- **contracts.json**: `effective_date`/`expiry_date`, `monthly_value`/`annual_value`, `sla_response_time` = dict
- **communications.json**: invoice reminder email history (file is `communications.json`, not `comms.json`)
- **clients.json**: master client list used by the forecasting UI (`/api/clients`)
