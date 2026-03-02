# TechVault MSP Enterprise RAG Intelligence Hub

A production-grade, 6-stage Retrieval-Augmented Generation (RAG) pipeline built over real-world MSP (Managed Service Provider) back-office data. Combines AI/ML research sources with synthetic enterprise data (billing, PSA, CRM, communications, contracts) to deliver grounded, cited answers via a chat UI, REST API, and CLI.

---

## Architecture Overview

```
Phase I   Collect    8 sources (ArXiv, Wikipedia, RSS + 5 enterprise systems)
   |
Phase II  Process    Chunk -> Embed (OpenAI) -> FAISS + BM25 dual index
   |
Phase III Serve      Query -> Rerank -> Generate -> PII Filter -> Answer + Citations
                     Web UI  |  REST API  |  CLI  |  MCP Server (17 tools)
   |
Eval      Measure    80 ground-truth queries x 4 models -> PASS/FAIL production gate
```

### Query Lifecycle (Phase III)

```
User Query
    |
    v
PromptGuard          (13 regex patterns -- injection detection)
    |
    v
HybridRetriever      (FAISS dense + BM25 sparse -> RRF fusion, top_k=20)
    |
    v
LLMReranker          (single GPT-4o-mini call scores all candidates, keep top 10)
    |
    v
RAGGenerator         (OpenAI or Anthropic -- grounded answer with numbered citations)
    |
    v
PIIFilter            (redact email / phone / SSN / CC / IP from output)
    |
    v
QueryResult          (answer + citations + latency + token cost)
```

---

## Data Sources

### Research Sources (public, no API key required)
| Collector | Source | Documents |
|---|---|---|
| `ArXivCollector` | arxiv.org API | ~77 papers (cs.AI / CL / IR) |
| `WikipediaCollector` | Wikipedia API | 8 topics (RAG, FAISS, embeddings, etc.) |
| `RSSCollector` | 5 RSS feeds | ~433 entries (ArXiv, The Gradient, Ahead of AI) |

### Enterprise MSP Sources (TechVault synthetic data, seed=42)
| Collector | System | Documents |
|---|---|---|
| `BillingCollector` | QuickBooks Enterprise | ~688 invoices |
| `PSACollector` | ConnectWise Manage | ~732 service tickets |
| `CRMCollector` | HubSpot CRM | 50 client profiles |
| `CommsCollector` | Exchange Online | ~142 email reminders |
| `ContractsCollector` | SharePoint | 50 service agreements |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | Python 3.11, asyncio, Pydantic v2 |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Vector Store | FAISS `IndexFlatIP` (cosine via L2-normalised inner product) |
| Sparse Index | BM25Okapi (rank-fusion via RRF) |
| LLM Generation | OpenAI (gpt-4o-mini, gpt-4o) + Anthropic (Haiku, Sonnet) |
| Forecasting | TimeFM 2.5 200M (Google, HuggingFace) -- monthly revenue prediction |
| Web Framework | FastAPI + Uvicorn |
| Observability | LangSmith (`@traceable` on full RAG chain) |
| MCP Server | 17 tools for Claude Desktop / custom MCP clients |
| Evaluation | Custom framework -- LLM-as-judge + retrieval metrics |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/spattnaik1998/Enterprise_RAG.git
cd Enterprise_RAG
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   LANGSMITH_API_KEY=ls__...   (optional, for tracing)
#   LANGSMITH_TRACING=true      (optional)
```

### 3. Generate synthetic enterprise data (run once)

```bash
python scripts/generate_enterprise_data.py
# Creates data/enterprise/*.json -- 50 clients, 13 months billing history
```

### 4. Run the pipeline

```bash
# Phase I: collect from all 8 sources + validate
python -m src.main phase1

# Phase II: chunk, embed, build FAISS + BM25 index
python -m src.main phase2

# Phase III: interactive CLI
python -m src.main phase3

# Phase III: Web UI + REST API
uvicorn app.server:app --reload --port 8000
# Then open http://localhost:8000
```

---

## Pipeline Commands

```bash
# Single-shot CLI query
python -m src.main phase3 --query "Which clients have overdue invoices?"

# Check pipeline state
python -m src.main status

# Dry-run (validate config only)
python -m src.main phase1 --dry-run

# Skip source health checks
python -m src.main phase1 --skip-health
```

---

## Web UI & REST API

Start the server:

```bash
uvicorn app.server:app --reload --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Obsidian Terminal chat UI |
| `GET /forecast` | GET | TimeFM revenue forecasting dashboard |
| `POST /api/chat` | POST | RAG query (`{message, provider, model}`) |
| `GET /api/health` | GET | Pipeline status + available models |
| `GET /api/clients` | GET | List all clients (for forecasting UI) |
| `GET /api/forecast/{client_id}` | GET | 3-month revenue forecast for a client |

### Supported Models (selectable per request)

| Model | Provider | Speed | Quality |
|---|---|---|---|
| `gpt-4o-mini` | OpenAI | Fast | Good |
| `gpt-4o` | OpenAI | Moderate | Excellent |
| `claude-haiku-4-5-20251001` | Anthropic | Fast | Good |
| `claude-sonnet-4-6` | Anthropic | Moderate | Excellent |

---

## MCP Server (17 Tools)

Exposes enterprise data tools for Claude Desktop or custom MCP clients:

```bash
python -m src.collection.mcp.server
```

| Category | Tools |
|---|---|
| Research | `search_arxiv`, `fetch_wikipedia`, `fetch_rss_feed`, `fetch_webpage`, `list_available_sources` |
| Billing | `billing_get_overdue_invoices`, `billing_get_aged_receivables`, `billing_get_client_statement`, `billing_get_invoice_details` |
| PSA | `psa_get_client_tickets`, `psa_get_unbilled_work` |
| CRM | `crm_get_client_profile`, `crm_get_at_risk_accounts` |
| Communications | `comms_get_invoice_history` |
| Contracts | `contracts_get_terms` |
| Cross-source | `get_client_360` (aggregates billing + PSA + CRM + comms + contracts) |

---

## Evaluation Framework

A rigorous eval suite measuring hybrid search quality and LLM answer quality across all four models, with a binary PASS/FAIL production gate.

### Quick start

```bash
# Smoke test: 1 model, billing queries, 5 samples (~$0.01)
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5

# Full production eval: all 4 models, all 80 queries (~$2.82)
python -m eval.run_eval

# Side-by-side comparison
python -m eval.run_eval --models gpt-4o-mini --models claude-haiku-4-5-20251001 --sample 10
```

### Metrics & Thresholds

| Metric | Threshold | Description |
|---|---|---|
| Retrieval Recall@10 | >= 80% | Expected keyword found in top-10 citations or answer |
| Source Type Hit Rate | >= 85% | Citation source_type matches expected data system |
| Answer Faithfulness | >= 85% | LLM judge: claims grounded in retrieved context |
| Answer Correctness | >= 75% | LLM judge: answer matches ground truth |
| Composite Score | >= 82% | Mean of all four metrics |

All five must pass for a model to be **PRODUCTION READY**. The CLI exits with code `0` if all tested models pass, `1` if any fail (CI/CD friendly).

### Query Dataset

80 ground-truth queries across 6 categories:

| Category | Queries | Sample Topics |
|---|---|---|
| `billing` | 20 | Overdue balances, invoice IDs, payment terms |
| `contracts` | 15 | SLA response times, expiry dates, monthly values |
| `crm` | 12 | Account health (RED/YELLOW/GREEN), contact names |
| `psa` | 15 | Technician hours, ticket types, client ticket counts |
| `communications` | 10 | Reminder sequences, client responses, escalation |
| `cross_source` | 8 | Multi-system synthesis (billing + CRM + contracts) |

See [eval/README.md](eval/README.md) for full CLI reference, cost estimates, and result interpretation.

---

## Project Structure

```
Enterprise_RAG/
|
|-- src/
|   |-- collection/          8 collectors + MCP server (17 tools)
|   |-- validation/          7-check document validator
|   |-- chunking/            AdaptiveChunker (keep_whole / sentence_window / fixed_overlap)
|   |-- embedding/           OpenAI embedder + FAISS index builder
|   |-- retrieval/           HybridRetriever, LLMReranker, PromptGuard, PIIFilter
|   |-- generation/          RAGGenerator (OpenAI) + AnthropicGenerator
|   |-- serving/             RAGPipeline orchestrator + QueryResult schema
|   |-- forecasting/         TimeFM invoice revenue forecaster
|   |-- schemas.py           RawDocument, ValidatedDocument, RejectedDocument
|   `-- main.py              CLI entry point (phase1 / phase2 / phase3 / status)
|
|-- app/
|   |-- server.py            FastAPI app (chat + forecast endpoints)
|   `-- static/              index.html (chat UI) + forecast.html
|
|-- eval/
|   |-- datasets/            80 ground-truth queries across 6 JSON files
|   |-- judge.py             LLMJudge (GPT-4o-mini, faithfulness + correctness)
|   |-- evaluator.py         RAGEvaluator (metrics, aggregation, reporting)
|   |-- run_eval.py          CLI entry point (typer + rich)
|   |-- results/             JSON eval reports (gitignored)
|   `-- README.md            Eval framework documentation
|
|-- scripts/
|   `-- generate_enterprise_data.py   Synthetic MSP data generator (seed=42)
|
|-- data/
|   `-- enterprise/          Synthetic source data (committed)
|       |-- invoices.json
|       |-- psa_tickets.json
|       |-- crm_profiles.json
|       |-- comms.json
|       `-- contracts.json
|
|-- config/
|   `-- config.yaml          All pipeline parameters (chunking, retrieval weights, etc.)
|
|-- .env.example             Environment variable template
|-- requirements.txt         Python dependencies
|-- CLAUDE.md                AI assistant instructions
`-- README.md                This file
```

---

## Configuration

All pipeline parameters live in `config/config.yaml`:

```yaml
retrieval:
  dense_weight: 0.7      # FAISS weight in RRF fusion
  sparse_weight: 0.3     # BM25 weight in RRF fusion
  top_k: 10              # Candidates before reranking
  rerank_top_k: 5        # Chunks kept after reranking

chunking:
  max_tokens: 512
  keep_whole_threshold: 600

embedding:
  model: text-embedding-3-small
  dimensions: 1536
```

---

## Observability

LangSmith tracing is enabled on the full Phase III chain. Set in `.env`:

```
LANGSMITH_API_KEY=ls__...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=Enterprise RAG
```

Traced spans: `rag_query` (top-level) -> `retrieve` -> `rerank` -> `generate` -> `embed_texts`

---

## Windows Note

The codebase patches `sys.stdout/stderr` to UTF-8 at startup in all entry points (`src/main.py`, `app/server.py`, `eval/run_eval.py`) to handle emoji in RSS content on cp1252 terminals.

---

## License

MIT
