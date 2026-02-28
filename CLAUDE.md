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

# Check phase checkpoint state
python -m src.main status

# Run MCP server standalone (for Claude Desktop or custom MCP clients)
python -m src.collection.mcp.server
```

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env` and fill in `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LANGSMITH_API_KEY`
3. Run `python scripts/generate_enterprise_data.py` (creates `data/enterprise/*.json`)
4. Run phases in order: `phase1` → review `data/validated/` → `phase2`

**Windows note**: The codebase patches `sys.stdout/stderr` to UTF-8 at startup in `src/main.py` and `src/embedding/pipeline.py` to handle emoji in RSS content on cp1252 terminals. Any new entry points on Windows need the same reconfigure block.

## Architecture

### Data Flow (per phase)

**Phase I** (`src/collection/` → `src/validation/`):
- `CollectionPipeline` runs 8 collectors concurrently via asyncio
- Each collector extends `BaseCollector` and returns `list[RawDocument]`
- `DocumentValidator` runs 7 checks (min/max length, language, alpha ratio, boilerplate, dedup by SHA-256 checksum, quality score)
- Output: `data/validated/*.json`, `data/rejected/*.json`, `data/checkpoint_phase1.json`

**Phase II** (`src/chunking/` → `src/embedding/`):
- `AdaptiveChunker` selects strategy per document: `keep_whole` (structured short docs), `fixed_overlap` (CRM/medium), `sentence_window` (Wikipedia/ArXiv/RSS)
- `Embedder` calls OpenAI `text-embedding-3-small` in batches, L2-normalises output
- `FAISSIndex` stores `IndexFlatIP` (inner product = cosine on normalised vecs) + `BM25Okapi` corpus
- Output: `data/index/faiss.index`, `data/index/chunks.json`, `data/index/bm25_corpus.json`

**Phase III** (planned): load `FAISSIndex.load("data/index")` → `search_hybrid()` → rerank → generate with OpenAI/Anthropic → return answer + citations

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

All three share `checksum` (SHA-256), `word_count`, `char_count` as `@computed_field`.

### Hybrid Retrieval

`FAISSIndex.search_hybrid()` fuses dense (FAISS) and sparse (BM25) results via Reciprocal Rank Fusion:
- Default weights: `dense=0.7`, `sparse=0.3`
- RRF formula: `weight / (rank + 60)` — robust to score-scale mismatch across indexes
- Configured in `config/config.yaml` under `retrieval:`

### Observability

LangSmith tracing via `@traceable` on `Embedder.embed_texts()`. Requires `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` in `.env`.

## Configuration

All pipeline parameters live in `config/config.yaml`:
- `collection.*` — enable/disable sources, file paths, query lists
- `validation.*` — thresholds (min/max length, quality score, language)
- `chunking.*` — `max_tokens=512`, `keep_whole_threshold=600`, window sizes
- `embedding.*` — model, dimensions, batch size
- `retrieval.*` — `dense_weight`, `sparse_weight`, `top_k`, `rerank_top_k`

## Phase III Architecture

**CLI**: `python -m src.main phase3` (interactive loop) or `python -m src.main phase3 --query "..."` (single-shot)

### Query Lifecycle

```
user query -> PromptGuard -> HybridRetriever -> LLMReranker -> RAGGenerator -> PIIFilter -> QueryResult
```

1. **PromptGuard** (`src/retrieval/guardrails.py`): 13 regex patterns covering jailbreaks, role overrides, instruction-ignoring. Returns `GuardrailResult(passed=False)` on hit.
2. **HybridRetriever** (`src/retrieval/retriever.py`): calls `Embedder.embed_query()` then `FAISSIndex.search_hybrid()` — RRF fusion of FAISS dense + BM25 sparse, returns top_k=10 candidates.
3. **LLMReranker** (`src/retrieval/reranker.py`): single OpenAI call with all candidates in one prompt; scores 0-10 per chunk; sorts and truncates to rerank_top_k=5.
4. **RAGGenerator** (`src/generation/generator.py`): builds numbered `[1]..[N]` context from reranked chunks + `SYSTEM_PROMPT` from `src/generation/prompts.py`; calls `gpt-4o-mini`; returns `RAGResponse` with answer + citations + token counts.
5. **PIIFilter** (`src/retrieval/guardrails.py`): regex-subs email/phone/SSN/CC/IP from generated answer with `[REDACTED_<TYPE>]`.

### Key Classes

| Class | File | Role |
|---|---|---|
| `RAGPipeline` | `src/serving/pipeline.py` | Orchestrator — loads index, wires all components |
| `HybridRetriever` | `src/retrieval/retriever.py` | Query embed + hybrid FAISS+BM25 search |
| `LLMReranker` | `src/retrieval/reranker.py` | Batch LLM relevance scoring |
| `RAGGenerator` | `src/generation/generator.py` | Grounded answer synthesis |
| `PromptGuard` | `src/retrieval/guardrails.py` | Injection detection |
| `PIIFilter` | `src/retrieval/guardrails.py` | Output redaction |
| `QueryResult` | `src/serving/pipeline.py` | Full result dataclass (answer, citations, timings, cost) |

### Observability

`RAGPipeline.query()` is `@traceable(name="rag_query", run_type="chain")`. The child spans — `retrieve`, `rerank`, `generate` — are also `@traceable`, so LangSmith shows the full chain tree with latency and token cost per step.

### Phase III CLI Flags

| Flag | Default | Effect |
|---|---|---|
| `--query TEXT` | None | Single-shot mode (omit for interactive loop) |
| `--top-k INT` | 10 | Candidates retrieved before reranking |
| `--rerank-top-k INT` | 5 | Chunks kept after LLM reranking |
| `--model TEXT` | gpt-4o-mini | OpenAI model for reranking + generation |
| `--no-rerank` | False | Skip LLM reranking (faster, cheaper) |
| `--no-pii` | False | Disable PII redaction |
| `--json` | False | JSON output (single-shot only) |

## Enterprise Data Field Names

These differ from intuitive assumptions — consult the enterprise data files directly:
- **invoices.json**: `invoice_date`, `line_items[].amount` (not `line_total`)
- **psa_tickets.json**: `type` (not `ticket_type`), `title`, `technician`, `hours_billed`, `resolved_date`, `resolution_note` (string)
- **crm_profiles.json**: `account_health` = `RED`/`YELLOW`/`GREEN`, contacts = `{cfo, it_manager, ar_contact}`
- **contracts.json**: `effective_date`/`expiry_date`, `monthly_value`/`annual_value`, `sla_response_time` = dict
