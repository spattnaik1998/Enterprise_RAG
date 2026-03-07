# SKILLS.md — Project Understanding Reference

This document is a deep-dive reference for any AI code session working in this repository.
It complements CLAUDE.md (commands + architecture overview) with full implementation detail
on every feature, every file, every schema, and every known quirk.

---

## 1. What This Project Is

**TechVault MSP Enterprise RAG Intelligence Hub**

A production-grade, portfolio-quality Retrieval-Augmented Generation system built for a
fictional Managed Service Provider (MSP) called TechVault. It ingests data from 8 sources
(3 AI/ML research + 5 synthetic enterprise back-office), stores everything in Supabase
(pgvector + PostgreSQL FTS), and serves a chat UI + monitoring dashboard over a FastAPI
backend deployable to Vercel.

**Purpose**: Demonstrate the full RAG engineering stack — from raw data collection through
embedding, indexing, hybrid retrieval, LLM reranking, grounded generation, PII filtering,
observability, and cloud deployment.

---

## 2. Complete Directory Map

```
Enterprise_RAG/
|
|-- api/
|   `-- index.py              # Vercel entry point: imports app from app.server
|
|-- app/
|   |-- __init__.py
|   |-- server.py             # FastAPI app, lifespan, all REST routes
|   |-- chat_logger.py        # Supabase chat_logs write/read (replaces JSONL)
|   `-- static/
|       |-- index.html        # Chat UI (Obsidian Terminal design)
|       |-- forecast.html     # TimeFM revenue forecasting page
|       `-- logs.html         # Monitoring dashboard (Chart.js)
|
|-- config/
|   `-- config.yaml           # All pipeline parameters (collection, chunking, retrieval)
|
|-- data/                     # GITIGNORED - local runtime data
|   |-- enterprise/           # Synthetic JSON files (generated once, seed=42)
|   |   |-- clients.json
|   |   |-- invoices.json
|   |   |-- psa_tickets.json
|   |   |-- crm_profiles.json
|   |   |-- communications.json
|   |   `-- contracts.json
|   |-- raw/                  # Phase I output: one JSON per collected document
|   |-- validated/            # Phase I output: documents that passed all 7 checks
|   |-- rejected/             # Phase I output: documents that failed checks
|   |-- index/                # Phase II output: faiss.index, chunks.json, bm25_corpus.json
|   |-- chat_logs/            # Legacy JSONL (now Supabase handles this)
|   `-- checkpoint_phase1.json
|
|-- eval/
|   |-- __init__.py
|   |-- datasets/             # 80 queries across 6 JSON files (billing/crm/psa/contracts/comms/cross)
|   |-- results/              # Auto-timestamped JSON eval reports (gitignored)
|   |-- evaluator.py          # RAGEvaluator class: loads pipeline, swaps generator per model
|   |-- judge.py              # LLMJudge: GPT-4o-mini grades faithfulness + correctness
|   `-- run_eval.py           # Typer CLI + rich progress + rich table; exit 0=pass, 1=fail
|
|-- scripts/
|   |-- generate_enterprise_data.py   # Creates data/enterprise/*.json (run once)
|   `-- migrate_to_supabase.py        # Migrates all local data to Supabase (idempotent)
|
|-- src/
|   |-- schemas.py            # RawDocument, ValidatedDocument, RejectedDocument (Pydantic v2)
|   |-- checkpoint.py         # PhaseICheckpoint: saves/loads Phase I state to JSON
|   |-- main.py               # CLI entry: phase1, phase2, phase3, status subcommands
|   |
|   |-- collection/           # Phase I: 8 collectors
|   |   |-- base_collector.py
|   |   |-- arxiv_collector.py
|   |   |-- wikipedia_collector.py
|   |   |-- rss_collector.py
|   |   |-- billing_collector.py
|   |   |-- psa_collector.py
|   |   |-- crm_collector.py
|   |   |-- comms_collector.py
|   |   |-- contracts_collector.py
|   |   |-- pipeline.py       # CollectionPipeline: runs all 8 concurrently via asyncio
|   |   `-- mcp/
|   |       |-- server.py     # 17-tool MCP server (research + enterprise tools)
|   |       `-- client.py     # MCP client for testing
|   |
|   |-- validation/           # Phase I: 7-check document quality gate
|   |   |-- validator.py      # DocumentValidator: orchestrates all 7 checks
|   |   |-- quality_checks.py # Individual check implementations
|   |   `-- report.py         # ValidationReport: human-readable summary
|   |
|   |-- chunking/             # Phase II: adaptive chunking
|   |   |-- chunker.py        # AdaptiveChunker: selects strategy per document type
|   |   `-- schemas.py        # Chunk dataclass (chunk_id, doc_id, text, source provenance)
|   |
|   |-- embedding/            # Phase II: embedding + indexing
|   |   |-- embedder.py       # Embedder: OpenAI text-embedding-3-small, L2-normalised
|   |   |-- faiss_index.py    # FAISSIndex: IndexFlatIP + BM25Okapi (local/CLI mode)
|   |   |-- supabase_index.py # SupabaseIndex: pgvector + FTS (web/Vercel mode)
|   |   `-- pipeline.py       # Phase II orchestrator: chunk -> embed -> index -> save
|   |
|   |-- retrieval/            # Phase III: retrieval + safety layers
|   |   |-- retriever.py      # HybridRetriever: embeds query, calls search_hybrid()
|   |   |-- reranker.py       # LLMReranker: batch OpenAI scoring in one call
|   |   `-- guardrails.py     # PromptGuard (13 injection patterns) + PIIFilter (5 types)
|   |
|   |-- generation/           # Phase III: answer synthesis
|   |   |-- generator.py      # RAGGenerator (OpenAI) + AnthropicGenerator (same interface)
|   |   `-- prompts.py        # SYSTEM_PROMPT, CITATION_TEMPLATE, NO_CONTEXT_RESPONSE
|   |
|   |-- serving/
|   |   `-- pipeline.py       # RAGPipeline: orchestrates the full query lifecycle
|   |
|   |-- forecasting/
|   |   `-- invoice_forecaster.py  # InvoiceForecaster: TimeFM 2.5 200M monthly revenue
|   |
|   `-- utils/
|       |-- logger.py         # setup_logger() with loguru + file rotation
|       `-- helpers.py        # Shared utility functions
|
|-- CLAUDE.md                 # Commands + architecture (read by Claude Code automatically)
|-- SKILLS.md                 # This file: deep-dive reference
|-- README.md                 # Top-level project summary
|-- requirements.txt          # Full local development dependencies
|-- requirements-vercel.txt   # Serving-layer only (no faiss/nltk/langdetect/faker/mcp)
|-- vercel.json               # Vercel serverless config (routes -> api/index.py)
|-- .env.example              # Template for required environment variables
`-- .gitignore
```

---

## 3. Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Embeddings (Phase II), reranker, OpenAI generation |
| `ANTHROPIC_API_KEY` | Yes | Claude generation (Haiku, Sonnet) |
| `SUPABASE_URL` | Yes (web) | PostgreSQL + pgvector backend |
| `SUPABASE_SERVICE_KEY` | Yes (web) | Service role key (bypasses RLS) — NOT the anon key |
| `LANGSMITH_API_KEY` | Optional | LangSmith tracing |
| `LANGSMITH_TRACING` | Optional | Set to `true` to enable traces |
| `LANGSMITH_PROJECT` | Optional | Project name (default: "Enterprise RAG") |

**Critical**: `SUPABASE_SERVICE_KEY` must be the `service_role` JWT (starts with `eyJ`, role claim = `"service_role"`),
not the `anon` key. The anon key will fail on all inserts due to RLS.

---

## 4. The Six Pipeline Phases

### Phase I — Collection (`src/collection/`)

**Entry**: `python -m src.main phase1`

`CollectionPipeline.run()` in `collection/pipeline.py` fires all 8 collectors concurrently
using `asyncio.gather`. Each collector extends `BaseCollector` which provides:
- `collect() -> list[RawDocument]` (abstract)
- HTTP retry logic via `tenacity`
- Uniform `RawDocument` output with `source`, `source_type`, `title`, `content`, `url`, `metadata`

After collection, `DocumentValidator` applies 7 checks in order:
1. `min_length` (>= 50 chars)
2. `max_length` (<= 500k chars)
3. `language` (langdetect, must be `en`)
4. `alpha_ratio` (>= 0.3 alphabetic characters)
5. `boilerplate` (rejects navigation-only / near-empty HTML)
6. `duplicate` (SHA-256 checksum dedup across the run)
7. `quality_score` (composite 0-1 score, threshold 0.3)

Output written to `data/validated/*.json`, `data/rejected/*.json`, `data/checkpoint_phase1.json`.
Pipeline halts here — human reviews validated data before Phase II.

**Result**: 2,195 collected, 2,180 validated (99.3% pass rate).

### Phase II — Chunking + Embedding (`src/chunking/`, `src/embedding/`)

**Entry**: `python -m src.main phase2`

`AdaptiveChunker` selects strategy per document:
- `keep_whole`: structured short docs (billing, PSA, comms, contracts < 600 tokens) — 1,851 chunks
- `sentence_window`: Wikipedia, ArXiv, long RSS (8-sentence windows, 2-sentence overlap) — 145 chunks
- `fixed_overlap`: CRM profiles, medium structured docs (512-token max, 64-token stride)

`Embedder` calls `text-embedding-3-small` (1536 dims) in batches, L2-normalises each vector
so cosine similarity == inner product (FAISS `IndexFlatIP` stores these).

Two indexes built locally:
- `data/index/faiss.index` — binary FAISS file
- `data/index/chunks.json` — full chunk metadata (text + provenance)
- `data/index/bm25_corpus.json` — BM25Okapi tokenised corpus

**Result**: 1,996 chunks, 1,996 vectors, cost $0.011 USD.

### Phase III — Query Lifecycle (`src/serving/pipeline.py`)

**Entry**: `uvicorn app.server:app` (web) or `python -m src.main phase3` (CLI)

Full query path:
```
user query
    -> PromptGuard.check()          # 13 regex injection patterns
    -> HybridRetriever.retrieve()   # embed query + search_hybrid() + RRF
    -> LLMReranker.rerank()         # single OpenAI batch call, score 0-10
    -> RAGGenerator/AnthropicGenerator.generate()  # grounded answer
    -> PIIFilter.redact()           # redact email/phone/SSN/CC/IP
    -> QueryResult                  # answer + citations + timings + tokens + cost
```

`RAGPipeline.__init__` accepts `index=None` (loads local FAISS) or `index=<SupabaseIndex>`
(skips disk load). This dual-mode design lets CLI use FAISS and web server use Supabase
with zero code duplication.

---

## 5. Supabase Database Schema

### Tables

**`rag_chunks`** — the vector index (replaces FAISS + BM25 locally)
```sql
chunk_id        TEXT  PRIMARY KEY    -- e.g. "inv_2025_1001_clt-00"
doc_id          TEXT                 -- parent document id
chunk_index     INT
chunk_strategy  TEXT                 -- keep_whole | sentence_window | fixed_overlap
text            TEXT
token_count     INT
source          TEXT                 -- e.g. "billing:quickbooks:INV-2025-1001"
source_type     TEXT                 -- billing | psa | crm | contracts | communications | arxiv | wikipedia | rss
title           TEXT
url             TEXT
metadata        JSONB
embedding       vector(1536)         -- pgvector, L2-normalised
```

**`chat_logs`** — every RAG interaction
```sql
id                    UUID  DEFAULT gen_random_uuid()
created_at            TIMESTAMPTZ DEFAULT now()
session_id            TEXT
query                 TEXT
answer_length         INT
provider              TEXT
model                 TEXT
blocked               BOOLEAN
blocked_reason        TEXT
citation_count        INT
source_types          TEXT[]
citations             JSONB          -- [{title, source_type, relevance_score}]
latency_retrieval_ms  FLOAT
latency_rerank_ms     FLOAT
latency_generation_ms FLOAT
latency_total_ms      FLOAT
tokens_prompt         INT
tokens_completion     INT
tokens_total          INT
estimated_cost_usd    FLOAT
pii_redacted_count    INT
```

**Enterprise data tables** (migrated from `data/enterprise/*.json`):
- `clients` — 50 clients with industry, contact info
- `invoices` — 688 invoices (fields: `invoice_date`, `line_items[].amount`)
- `invoice_line_items` — individual line items
- `psa_tickets` — 732 tickets (fields: `type`, `title`, `technician`, `hours_billed`, `resolved_date`)
- `crm_profiles` — 50 profiles (fields: `account_health` = RED/YELLOW/GREEN, contacts = {cfo, it_manager, ar_contact})
- `communications` — 142 email reminders
- `contracts` — 50 contracts (fields: `effective_date`/`expiry_date`, `monthly_value`, `sla_response_time` dict)

### PostgreSQL Functions (RPC)

**`match_chunks(query_embedding vector, match_count int)`**
- Uses pgvector `<=>` operator (cosine distance) on `embedding` column
- Returns chunks sorted by `1 - (embedding <=> query_embedding)` (similarity score)
- Return types: all TEXT ids (not UUID — the columns were altered after initial creation)

**`search_chunks_fts(query_text text, match_count int)`**
- Uses `websearch_to_tsquery('english', query_text)` on `title || ' ' || text`
- Returns chunks with `ts_rank` score
- Rejects empty strings (guarded in `SupabaseIndex._search_sparse`)

**`get_chat_stats()`**
- Returns single JSONB row with aggregated monitoring stats
- Called by `compute_stats()` in `chat_logger.py`

### Common Supabase Gotchas

- **UUID vs TEXT**: `chunk_id` and `doc_id` are TEXT, not UUID. The functions must declare
  `TEXT` return types or PostgREST throws error `42804`.
- **Function overloads**: If you recreate a function without dropping ALL overloads first,
  PostgREST throws `PGRST203` (ambiguous). Always `DROP FUNCTION IF EXISTS` with full
  signature before `CREATE FUNCTION`.
- **Service key**: Using the `anon` key causes all writes to fail silently due to RLS.
  Always use `service_role` key on the backend.
- **Batch size**: PostgREST has a ~1MB body limit. Chunk migration uses `CHUNK_BATCH = 5`
  rows at a time (1536 floats × 5 rows × 20 bytes ≈ 150KB per batch).

---

## 6. SupabaseIndex vs FAISSIndex

Both implement the identical `search_hybrid(query_vec, query_text, top_k, dense_weight, sparse_weight)` interface.

| Aspect | FAISSIndex | SupabaseIndex |
|---|---|---|
| Dense search | FAISS `IndexFlatIP.search()` | `match_chunks` RPC (pgvector `<=>`) |
| Sparse search | `BM25Okapi.get_scores()` | `search_chunks_fts` RPC (PostgreSQL FTS) |
| RRF formula | `weight / (rank + 60)` | identical |
| `ntotal` | `self.faiss_index.ntotal` | Supabase `count` query (cached) |
| Load | `FAISSIndex.load(path)` from disk | `SupabaseIndex()` connects at init |
| Used by | CLI (`python -m src.main phase3`) | Web server (`uvicorn app.server:app`) |

`HybridRetriever` type-hints `index: FAISSIndex` but Python doesn't enforce this at runtime.
`SupabaseIndex` satisfies the duck type with no changes to `HybridRetriever`.

---

## 7. Web Server Routes (`app/server.py`)

| Method | Path | Description |
|---|---|---|
| GET | `/` | Serves `app/static/index.html` (chat UI) |
| GET | `/forecast` | Serves `app/static/forecast.html` (TimeFM page) |
| GET | `/logs` | Serves `app/static/logs.html` (monitoring dashboard) |
| GET | `/api/health` | Pipeline status, vector count, available models |
| POST | `/api/chat` | Full RAG query — accepts `{message, provider, model, session_id}` |
| GET | `/api/clients` | List all client IDs and names (for forecasting UI) |
| GET | `/api/forecast/{client_id}` | TimeFM revenue forecast, `?horizon=N` (clamped 1-12) |
| GET | `/api/logs` | Paginated chat history (`?limit=50&offset=0`) |
| GET | `/api/logs/stats` | Aggregated dashboard stats |

**Startup sequence** (via `lifespan` async context manager):
1. `SupabaseIndex()` — connects to Supabase, queries `ntotal`
2. `RAGPipeline(index=supabase_index)` — wires retriever/reranker/generator
3. Server ready

**Generator selection per request**: `_make_generator(provider, model)` instantiates
`AnthropicGenerator` or `RAGGenerator` fresh per request. The reranker always uses
`gpt-4o-mini` regardless of the user's model choice.

**Supported models**:
- OpenAI: `gpt-4o-mini`, `gpt-4o`
- Anthropic: `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`

To add a new model: update `_MODEL_PRICING` in `src/generation/generator.py` AND
`_OPENAI_MODELS` or `_ANTHROPIC_MODELS` in `app/server.py`.

---

## 8. Chat Logger (`app/chat_logger.py`)

Three public functions (server.py imports these directly):

- `log_interaction(**kwargs)` — inserts one row into `chat_logs` table; all errors caught
  and logged as warnings (never propagates to the API response)
- `load_logs(limit, offset)` — paginated SELECT ordered by `created_at DESC`
- `compute_stats()` — calls `get_chat_stats()` RPC; handles PostgREST scalar wrapping
  (result may come as `list[dict]`, `dict`, or nested under function name key)

Uses a module-level lazy singleton `_sb`. `load_dotenv()` is called at the top of the
module so env vars are available when `_SUPABASE_URL` and `_SUPABASE_KEY` are read,
regardless of import order in `server.py`.

---

## 9. Frontend Pages (`app/static/`)

### `index.html` — Chat UI
- Dark terminal theme ("Obsidian Terminal")
- Model selector: provider dropdown + model dropdown (dynamically populated from `/api/health`)
- Citation panel: source type badges, relevance scores, expandable chunk text
- Latency + token + cost display per response
- Nav links: Chat / Forecast / Monitoring

### `forecast.html` — Revenue Forecasting
- Client selector (populated from `/api/clients`)
- Horizon slider (1–12 months)
- Chart.js line chart: historical actuals (solid) + forecast (dashed) + 10th/90th percentile band
- Calls `GET /api/forecast/{client_id}?horizon=N`
- 503 handling: shows install instructions if TimeFM not installed

### `logs.html` — Monitoring Dashboard
- 6 KPI cards: total queries, avg latency, avg cost, blocked queries, PII redactions, avg citations
- Chart.js charts: query timeline, model usage doughnut, source type bar, latency breakdown, provider split
- Recent queries table (last 50 rows, newest first)
- Auto-refreshes every 60 seconds
- Export JSON button

---

## 10. Forecasting (`src/forecasting/invoice_forecaster.py`)

`InvoiceForecaster` uses **TimeFM 2.5 200M** (Google, ~800 MB PyTorch model from HuggingFace).

- Reads `data/enterprise/invoices.json` (field: `invoice_date`, `line_items[].amount`)
- Aggregates monthly totals per `client_id` (13 months of history)
- Lazy-loads TimeFM on first call; singleton cached for subsequent requests
- Returns: `{client_id, historical: [{month, revenue}], forecast: [{month, point, p10, p90}]}`
- Inference runs in `loop.run_in_executor` (thread pool) to avoid blocking the async event loop
- If TimeFM not installed: `ImportError` → HTTP 503 with pip install instructions

**TimeFM install**: `pip install -e "C:/Users/91838/Downloads/TimeFM/timesfm[torch]"` (local build).

---

## 11. MCP Server (`src/collection/mcp/server.py`)

17 tools exposed via Model Context Protocol for external clients (e.g., Claude Desktop):

**Research tools (5)**:
- `search_arxiv(query, max_results)` — ArXiv API
- `fetch_wikipedia(topic)` — Wikipedia summary + full article
- `fetch_rss_feed(url, max_entries)` — Any RSS feed
- `fetch_webpage(url)` — Web scrape with BeautifulSoup
- `list_available_sources()` — Lists all configured sources

**Billing tools (4)**:
- `billing_get_overdue_invoices(days_overdue, client_id)` — AR aging
- `billing_get_aged_receivables()` — Bucketed 0-30/31-60/61-90/90+ days
- `billing_get_client_statement(client_id)` — Full invoice history
- `billing_get_invoice_details(invoice_id)` — Single invoice breakdown

**PSA tools (2)**:
- `psa_get_client_tickets(client_id, status)` — Service tickets
- `psa_get_unbilled_work(client_id)` — Hours billed but not yet invoiced

**CRM tools (2)**:
- `crm_get_client_profile(client_id)` — Full profile with account health
- `crm_get_at_risk_accounts(health_status)` — RED/YELLOW filter

**Comms tool (1)**:
- `comms_get_invoice_history(client_id)` — Email reminder log

**Contracts tool (1)**:
- `contracts_get_terms(client_id)` — SLA, auto-renewal, values

**Cross-source (1)**:
- `get_client_360(client_id)` — Aggregates billing + PSA + CRM + comms + contracts in one call

---

## 12. Evaluation Framework (`eval/`)

### Dataset: 80 queries in `eval/datasets/`
- `billing_queries.json` (20), `contracts_queries.json` (15), `crm_queries.json` (12)
- `psa_queries.json` (15), `communications_queries.json` (10), `cross_source_queries.json` (8)

Each query has: `id`, `query`, `ground_truth`, `expected_source_types`, `expected_keywords`, `difficulty`

### Metrics (all must pass for PRODUCTION READY gate)
- **Recall@10** >= 80%: any `expected_keyword` in top-10 citation titles/sources or answer text
- **Source Type Hit** >= 85%: any citation `source_type` matches `expected_source_types`
- **Faithfulness** >= 85%: LLM judge (GPT-4o-mini) — all claims grounded in retrieved context
- **Correctness** >= 75%: LLM judge — answer matches ground truth
- **Composite** >= 82%: mean of all four

### Key classes
- `RAGEvaluator` (`evaluator.py`): loads pipeline once, calls `pipeline.query()` per sample,
  swaps generator per model, computes Recall@10 + Source Type Hit inline
- `LLMJudge` (`judge.py`): single GPT-4o-mini call per query, temperature=0, returns JSON
  `{faithfulness_score, correctness_score, reasoning}`

### CLI
```bash
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5  # ~$0.01
python -m eval.run_eval                                                        # all models, ~$2.82
```
Exit 0 = all thresholds pass. Exit 1 = any fail. Integrates with CI/CD.

---

## 13. Data Migration (`scripts/migrate_to_supabase.py`)

Migrates local JSON + FAISS to Supabase in dependency order:
1. `clients` → 2. `invoices` → 3. `invoice_line_items` → 4. `psa_tickets`
5. `crm_profiles` → 6. `communications` → 7. `contracts`
8. `rag_chunks` (extracts vectors via `faiss.reconstruct(i)`, rounded to 6dp)
9. `chat_logs` (legacy JSONL if it exists)

Uses `upsert(on_conflict="chunk_id")` — fully idempotent, safe to re-run.
`--only <step>` flag for partial re-runs (e.g. `--only chunks`).
`CHUNK_BATCH = 5` to stay under PostgREST 1MB body limit.

---

## 14. Vercel Deployment

**Files**:
- `api/index.py` — `from app.server import app` (Vercel detects the `app` ASGI export)
- `vercel.json` — routes all traffic to `api/index.py`, `maxDuration: 60` (requires Pro plan)
- `requirements-vercel.txt` — serving layer only; excludes `faiss-cpu`, `rank-bm25`, `tiktoken`,
  `nltk`, `langdetect`, `faker`, `mcp`, `arxiv`, `wikipedia-api`, `feedparser`

**Required env vars in Vercel project settings**:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`

**Cold start**: no FAISS binary to load — Supabase connection only (~1-2s vs ~10s+ with FAISS).

---

## 15. Key Design Decisions

| Decision | Why |
|---|---|
| Supabase replaces FAISS for web | Eliminates 200MB binary from serverless bundle; pgvector query is fast enough |
| Dual index mode (FAISS for CLI, Supabase for web) | CLI/eval doesn't need Supabase running; preserves local dev workflow |
| `index=` parameter on `RAGPipeline` | Duck-type injection — zero changes to HybridRetriever |
| LLMReranker always uses OpenAI | Reranker needs to score chunks fast and cheaply; not user-facing model choice |
| `log_interaction` errors swallowed | Chat logging must never degrade the API response |
| Batch size = 5 for chunk migration | PostgREST 1MB body limit; 1536 floats × 5 rows ≈ 150KB safe margin |
| `load_dotenv()` in `chat_logger.py` | Module may be imported before `server.py` calls its own `load_dotenv()` |
| `NO_CONTEXT_RESPONSE` on empty retrieval | Honest fallback; system prompt also instructs "say so" — no hallucination |

---

## 16. Known Limitations

1. **Cross-source aggregation queries** (e.g. "total unbilled work for expiring clients") return
   "no information" — semantic RAG cannot perform SQL JOINs + date math. This is by design.
   Fix would require a dedicated SQL query layer or metadata filtering on chunks.

2. **Vector count discrepancy**: Supabase has ~1,771 vectors vs 1,996 original chunks.
   Some batches may have failed silently during migration. Re-running
   `python scripts/migrate_to_supabase.py --only chunks` will fill gaps (upsert is safe).

3. **TimeFM requires local build**: The forecasting feature only works in local dev
   (`pip install -e ".../TimeFM/timesfm[torch]"`). Not available on Vercel (too large).
   The server returns HTTP 503 with instructions when TimeFM is missing.

4. **No test suite**: pytest is installed but no tests exist. Eval framework (`eval/`)
   is the closest substitute for integration tests.

5. **`rerank_top_k=5`** default in web mode may be too low for cross-source queries.
   Increasing to 10 via `RAGPipeline(rerank_top_k=10)` in `server.py` may improve recall.

---

## 17. Windows-Specific Notes

- All entry points patch `sys.stdout/stderr` to UTF-8: `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`
  Required files: `src/main.py`, `src/embedding/pipeline.py`, `app/server.py`, `eval/run_eval.py`
- Any new entry point on Windows needs the same reconfigure block at the top
- Never use non-ASCII characters in Python source files (box-drawing, em-dashes, bullets crash cp1252)
- Use forward slashes or `pathlib.Path` for all file paths

---

## 18. LangSmith Observability

`@traceable` decorators on:
- `RAGPipeline.query()` — top-level chain trace
- `HybridRetriever.retrieve()` — retrieval span
- `LLMReranker.rerank()` — reranking span
- `RAGGenerator.generate()` / `AnthropicGenerator.generate()` — LLM span
- `Embedder.embed_texts()` — Phase II embedding span

Enable with `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`.
Project name set via `LANGSMITH_PROJECT` (default: "Enterprise RAG").
