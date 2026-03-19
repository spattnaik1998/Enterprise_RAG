# Enterprise RAG — Complete Execution Guide

## Quick Reference: All Commands

```bash
# One-time setup
pip install -r requirements.txt
cp .env.example .env              # Fill in API keys
python scripts/generate_enterprise_data.py

# Phase I: Collection + Validation
python -m src.main phase1                    # Full pipeline with health checks
python -m src.main phase1 --dry-run          # Validate config without running
python -m src.main phase1 --skip-health      # Skip health checks (faster)

# Phase II: Chunking + Embedding + Indexing
python -m src.main phase2

# Phase III: Query Interface (CLI)
python -m src.main phase3                    # Interactive query loop
python -m src.main phase3 --query "YOUR QUERY"  # Single-shot query
python -m src.main phase3 --query "YOUR QUERY" --json  # JSON output

# Phase III: Web UI + REST API
uvicorn app.server:app --reload --port 8000

# MCP Server (standalone)
python -m src.collection.mcp.server

# Council Orchestrator (3-agent voting)
python -m src.agents.council_cli             # Interactive mode
python -m src.agents.council_cli --query "YOUR QUERY"  # Single query

# Evaluation Framework
python -m eval.run_eval                      # Full eval (all 4 models, 80 queries)
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5  # Smoke test
python -m eval.run_eval --models gpt-4o-mini gpt-4o --category contracts --no-rerank

# Supabase Migration
python scripts/migrate_to_supabase.py        # Full migration
python scripts/migrate_to_supabase.py --only chunks  # Partial re-run

# Check system status
python -m src.main status
```

---

## Detailed Execution Instructions

### 1. Environment Setup

#### Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: On Windows, the pip install may take 10-15 minutes for FAISS and other compiled packages.

#### Configure Environment Variables
```bash
# Create .env file from template
cp .env.example .env

# Edit .env with your keys:
# OPENAI_API_KEY=sk-xxx...
# ANTHROPIC_API_KEY=sk-ant-xxx...
# SUPABASE_URL=https://xxx.supabase.co
# SUPABASE_SERVICE_KEY=eyJxxx...
```

**Required Keys**:
- `OPENAI_API_KEY` — For embeddings, reranking, generation
- `ANTHROPIC_API_KEY` — For Claude models
- `SUPABASE_URL` — For web server only (optional for CLI)
- `SUPABASE_SERVICE_KEY` — Must be `service_role` key, not `anon`

**Optional Keys**:
- `LANGSMITH_API_KEY` — For tracing
- `LANGSMITH_TRACING=true` — Enable tracing
- `LANGSMITH_PROJECT=Enterprise RAG` — Custom project name

#### Generate Synthetic Enterprise Data (One-Time)
```bash
# Creates 50 clients, 13 months billing, 9 industries
# Seed=42 ensures reproducibility
python scripts/generate_enterprise_data.py

# Output: data/enterprise/*.json
#   - invoices.json (688 records)
#   - psa_tickets.json (732 records)
#   - crm_profiles.json (50 records)
#   - contracts.json (50 records)
#   - communications.json (142 records)
```

---

### 2. Phase I: Data Collection + Validation

#### Full Collection Pipeline (with health checks)
```bash
python -m src.main phase1
```

**What it does**:
- Collects from 8 sources (3 research, 5 MSP)
- Validates 7 criteria per document
- Saves validated docs to `data/validated/`
- Saves rejected docs to `data/rejected/`
- **Pauses here** for human review before proceeding to Phase II

**Output**:
```
[Phase I] Collected 2,195 documents
[Phase I] Validated 2,180 (99.3% pass rate)
[Phase I] Rejected 15 (low quality, boilerplate, duplicates)
[Phase I] Checkpoint saved → data/checkpoint_phase1.json
```

**Expected Runtime**: ~2-3 minutes

#### Dry-Run (Validate Config Only)
```bash
python -m src.main phase1 --dry-run
```

Validates config without making API calls. Useful for checking credentials.

#### Skip Health Checks (Faster)
```bash
python -m src.main phase1 --skip-health
```

Skips source availability checks (faster if sources are unavailable).

---

### 3. Phase II: Chunking + Embedding + Indexing

#### Build FAISS Index (Required for Phase III CLI)
```bash
python -m src.main phase2
```

**What it does**:
1. Loads 2,180 validated documents
2. Chunks using adaptive strategy (keep_whole, fixed_overlap, sentence_window)
3. Embeds with OpenAI `text-embedding-3-small`
4. Creates FAISS IndexFlatIP + BM25Okapi dual index
5. Saves to `data/index/`

**Output**:
```
[Phase II] Processing 2,180 docs → 1,996 chunks
[Phase II] Embedding 1,996 chunks (batch_size=32)
[Phase II] Building FAISS IndexFlatIP (1,996 vectors, 1536 dims)
[Phase II] Building BM25 corpus (1,996 docs)
[Phase II] Index saved → data/index/
```

**Files Created**:
- `data/index/faiss.index` — FAISS binary index
- `data/index/chunks.json` — Chunk metadata + provenance
- `data/index/bm25_corpus.json` — BM25 tokenized corpus

**Cost**: ~$0.011 USD
**Expected Runtime**: 2-5 minutes

---

### 4. Phase III: Query Interface (CLI)

#### Interactive Query Loop
```bash
python -m src.main phase3
```

Opens an interactive prompt:
```
[RAGPipeline] Ready | 1,996 vectors | model=gpt-4o-mini | rerank=True
>>> Enter your query (or 'quit' to exit):
```

Type queries and get grounded answers with citations:
```
>>> Which clients have overdue invoices?

[RAG] Query: Which clients have overdue invoices?
[Retriever] Found 10 candidates
[Reranker] Scored and ranked to top 5
[Generator] Claude Haiku synthesizing answer...

Answer:
Based on the retrieved data, the following clients have overdue invoices:

1. Alpine Financial (45 days overdue, $12,500)
   [Source: invoices.json]

2. Lakewood Tech (32 days overdue, $8,750)
   [Source: invoices.json]

Cost: $0.0012 USD
Latency: 1,234ms
```

#### Single-Shot Query (Batch Mode)
```bash
python -m src.main phase3 --query "What clients have overdue invoices?"
```

**Output**: Answer printed to stdout, then exits

#### JSON Output (for scripting)
```bash
python -m src.main phase3 --query "What is ClientA's invoice total?" --json
```

**Output**:
```json
{
  "query": "What is ClientA's invoice total?",
  "answer": "ClientA's invoice total is $45,230.",
  "citations": [
    {
      "index": 1,
      "title": "Invoice INV-2024-001",
      "source": "invoices.json",
      "chunk_id": "billing_chunk_042"
    }
  ],
  "latency_ms": {
    "retrieval": 234,
    "rerank": 156,
    "generation": 523,
    "total": 913
  },
  "estimated_cost_usd": 0.0012
}
```

#### Phase III Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--query TEXT` | None | Single-shot query (omit for interactive) |
| `--top-k INT` | 10 | Candidates before reranking |
| `--rerank-top-k INT` | 5 | Chunks kept after reranking |
| `--model TEXT` | gpt-4o-mini | OpenAI model for generation |
| `--no-rerank` | False | Skip LLM reranking (faster, cheaper) |
| `--no-pii` | False | Disable PII redaction |
| `--json` | False | JSON output (single-shot only) |

#### Example Queries by Category

**Billing/AR**:
```bash
python -m src.main phase3 --query "Which clients have overdue invoices?"
python -m src.main phase3 --query "Show aged receivables over 60 days"
python -m src.main phase3 --query "What is Alpine Financial's total outstanding balance?"
```

**PSA/Tickets**:
```bash
python -m src.main phase3 --query "List unresolved support tickets"
python -m src.main phase3 --query "How many hours were billed this month?"
```

**CRM/Profiles**:
```bash
python -m src.main phase3 --query "Which accounts are at risk?"
python -m src.main phase3 --query "Get ClientA's account health status"
```

**Contracts**:
```bash
python -m src.main phase3 --query "Which contracts expire in the next 30 days?"
python -m src.main phase3 --query "Show SLA response times for all contracts"
```

**Cross-Source**:
```bash
python -m src.main phase3 --query "Get a complete view of Alpine Financial"
python -m src.main phase3 --query "Which at-risk accounts have unpaid invoices?"
```

---

### 5. Web UI + REST API

#### Start Web Server
```bash
# From project root
uvicorn app.server:app --reload --port 8000
```

**Output**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

**Access Points**:
- **Chat UI**: http://localhost:8000/
- **Forecast UI**: http://localhost:8000/forecast
- **Logs Dashboard**: http://localhost:8000/logs
- **API Docs**: http://localhost:8000/docs (Swagger UI)

#### API Endpoints

**Chat Endpoint**:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Which clients have overdue invoices?",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "session_id": "user123"
  }'
```

**Response**:
```json
{
  "answer": "Based on the data...",
  "citations": [...],
  "provider": "openai",
  "model": "gpt-4o-mini",
  "cost_usd": 0.0012,
  "latency_ms": 1234
}
```

**Health Check**:
```bash
curl http://localhost:8000/api/health
```

**List Clients** (for forecasting dropdown):
```bash
curl http://localhost:8000/api/clients
```

**Get Revenue Forecast**:
```bash
curl http://localhost:8000/api/forecast/client_123?horizon=6
```

**View Chat Logs**:
```bash
curl "http://localhost:8000/api/logs?limit=50&offset=0"
```

**Dashboard Stats**:
```bash
curl http://localhost:8000/api/logs/stats
```

---

### 6. MCP Server (Model Context Protocol)

#### Start Standalone MCP Server
```bash
python -m src.collection.mcp.server
```

**Output**:
```
[MCP] Server listening on stdio
[MCP] Registered 17 tools:
  - search_arxiv
  - fetch_wikipedia
  - get_client_360
  - billing_get_overdue_invoices
  - ... (13 more)
```

**Use with Claude Desktop or custom MCP clients** to access:
- 5 research tools (search, fetch, browse)
- 4 billing tools (invoices, AR, statements)
- 2 PSA tools (tickets, unbilled work)
- 2 CRM tools (profiles, at-risk accounts)
- 1 comms tool (email history)
- 1 contracts tool
- 1 cross-source aggregation (get_client_360)

---

### 7. Council Orchestrator (3-Agent Voting)

#### Interactive Mode
```bash
python -m src.agents.council_cli
```

**Output**:
```
Council Orchestrator — 3-Agent Voting System
>>> Enter your query (or 'quit' to exit):
```

Type complex queries for multi-agent consensus:
```
>>> Should we escalate Alpine Financial?

[FastCreativeAgent] Processing...
[ConservativeCheckerAgent] Processing...
[PolicyVerifierAgent] Reviewing...

COUNCIL VERDICT:
Agent Consensus: 2 of 3 agree → Accept
Winning Agent: ConservativeCheckerAgent
Answer: Based on their 45-day overdue balance and RED account health status,
        I recommend escalation for immediate collection follow-up.
Escalated: No
PII Concerns: None
Total Cost: $0.0082 USD
Latency: 9,234 ms
```

#### Single Query
```bash
python -m src.agents.council_cli --query "Should we escalate Alpine Financial?"
```

---

### 8. Evaluation Framework

#### Full Production Evaluation (all 4 models, 80 queries)
```bash
python -m eval.run_eval
```

**Cost**: ~$2.82 USD
**Runtime**: ~30 minutes
**Output**: `eval/results/results_TIMESTAMP.json` with metrics

#### Smoke Test (quick validation, $0.01)
```bash
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5 --no-rerank
```

**Cost**: ~$0.01 USD
**Runtime**: ~2 minutes
**Use**: Quick sanity check during development

#### Multi-Model Comparison
```bash
python -m eval.run_eval --models gpt-4o-mini gpt-4o claude-haiku-4-5-20251001 --sample 10
```

#### Single Category (18 contracts queries)
```bash
python -m eval.run_eval --models gpt-4o-mini --category contracts
```

#### All Categories with Custom Output
```bash
python -m eval.run_eval \
  --models gpt-4o-mini \
  --output results/custom_eval.json \
  --judge-model gpt-4o-mini \
  --top-k 20 \
  --rerank-top-k 10 \
  --seed 42
```

#### Eval Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--models TEXT` | all 4 | Space-separated: gpt-4o-mini, gpt-4o, claude-haiku-4-5-20251001, claude-sonnet-4-6 |
| `--category TEXT` | all 6 | billing, contracts, crm, psa, communications, cross_source |
| `--sample INT` | 0 (all) | Queries per category to sample |
| `--no-rerank` | off | Skip LLM reranking (faster, cheaper) |
| `--output PATH` | auto | JSON report path |
| `--judge-model TEXT` | gpt-4o-mini | LLM judge model |
| `--top-k INT` | 20 | Retrieval candidates |
| `--rerank-top-k INT` | 10 | Chunks after reranking |
| `--seed INT` | 42 | RNG seed |
| `--quiet` | off | Suppress progress bar |

**Example Combinations**:
```bash
# Quick smoke test
python -m eval.run_eval --models gpt-4o-mini --sample 1 --no-rerank

# Compare models on contracts
python -m eval.run_eval --models gpt-4o-mini gpt-4o --category contracts --sample 5

# Full evaluation with custom settings
python -m eval.run_eval --top-k 30 --rerank-top-k 15 --judge-model gpt-4o
```

---

### 9. Check System Status

#### View Checkpoint State
```bash
python -m src.main status
```

**Output**:
```
[RAGPipeline] Status Report
├─ Phase I Checkpoint: data/checkpoint_phase1.json
│  ├─ Status: COMPLETE
│  ├─ Collected: 2,195 documents
│  ├─ Validated: 2,180 documents
│  └─ Rejected: 15 documents
├─ Phase II Index: data/index/
│  ├─ Status: READY
│  ├─ Total vectors: 1,996
│  ├─ Embedding model: text-embedding-3-small
│  └─ Index size: 4.2 MB
└─ Phase III Pipeline: Ready to serve queries
```

---

### 10. Supabase Migration (Optional)

#### Full Migration (upload all data to Supabase)
```bash
python scripts/migrate_to_supabase.py
```

**Prerequisites**:
- `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in .env
- `SUPABASE_SERVICE_KEY` must be `service_role` key (not `anon`)

**What it does**:
1. Uploads all chunks and embeddings to pgvector
2. Configures full-text search
3. Idempotent (safe to re-run)

**Output**:
```
[Supabase] Inserting 1,996 chunks in batches of 5...
[Supabase] Batch 1/400: 5 chunks → 250KB
[Supabase] Complete: 1,996 chunks, 48.5 MB
```

#### Partial Re-run (update only chunks)
```bash
python scripts/migrate_to_supabase.py --only chunks
```

---

### 11. Testing (Validation of Review Brief v2 Implementation)

#### Phase 1-2: Smoke + Unit Tests (No Dependencies)
```bash
# All tests pass ✓
python -c "
from src.agents.router import RouteType
from src.security.gateway import AgentSecurityGateway
print('PASS: All imports work')
print('RouteType:', [e.value for e in RouteType])
"
```

#### Phase 3: Integration Tests (Requires FAISS)
```bash
python << 'EOF'
import asyncio
from src.serving.pipeline import RAGPipeline
from src.agents.router import DirectRAGAgent

pipeline = RAGPipeline(index_dir="data/index")
agent = DirectRAGAgent(pipeline)

async def test():
    verdict = await agent.run("What clients have overdue invoices?")
    print(f"PASS: DirectRAGAgent executed in {verdict.latency_ms:.0f}ms")

asyncio.run(test())
EOF
```

#### Phase 4: End-to-End Test (Requires API Keys)
```bash
# NO_CONTEXT_RESPONSE fallback test
python -m src.main phase3 --query "What is the difference between machine learning and AI?"

# Router logging test
python -m src.agents.council_cli --query "What is ClientA's invoice total?"

# Skill execution test
python << 'EOF'
import asyncio
from src.skills.registry import SkillRegistry
from src.skills.base import SkillContext

async def test():
    registry = SkillRegistry()
    skill = registry.get('ar_risk_report')
    if skill:
        ctx = SkillContext(query="Generate AR risk report")
        result = await skill.execute(ctx)
        print(f"PASS: Skill executed in {result.latency_ms:.0f}ms")

asyncio.run(test())
EOF
```

#### Phase 5: Regression Tests
```bash
# Standard RAG pipeline
python -m src.main phase3 --query "What is ClientA's account health?"

# Council orchestrator
python -m src.agents.council_cli --query "Should we escalate this client?"

# Eval framework
python -m eval.run_eval --models gpt-4o-mini --sample 1 --no-rerank
```

---

## Common Use Cases

### Use Case 1: Query Single Data Point
```bash
python -m src.main phase3 --query "What is Alpine Financial's total outstanding balance?" --json
```

### Use Case 2: Analyze Risk
```bash
python -m src.agents.council_cli --query "Which of our top 10 clients are at highest risk?"
```

### Use Case 3: Generate Report (Skill)
```bash
# Interactive mode
python -m src.agents.council_cli

>>> Generate AR risk report for clients with 30+ day overdue
```

### Use Case 4: Monitor System
```bash
# Terminal 1: Start server
uvicorn app.server:app --reload --port 8000

# Terminal 2: Access UI
# Open http://localhost:8000
# Chat, view forecasts, monitor logs
```

### Use Case 5: Validate Changes
```bash
# After code changes, run smoke tests
python -m eval.run_eval --models gpt-4o-mini --sample 1 --no-rerank

# On success, run full eval
python -m eval.run_eval
```

---

## Troubleshooting

### "FAISS index not found"
```bash
# Need to run Phase II first
python -m src.main phase2
```

### "API key missing or invalid"
```bash
# Check .env file
cat .env

# Verify keys are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### "Windows encoding error"
The code automatically patches `sys.stdout` to UTF-8 on Windows startup. If you see encoding errors, ensure:
```bash
# Run in UTF-8 terminal
chcp 65001  # Windows Command Prompt
# or use PowerShell (native UTF-8 support)
```

### "Port 8000 already in use"
```bash
# Use different port
uvicorn app.server:app --reload --port 8001
```

### "Supabase connection error"
```bash
# Check credentials
cat .env | grep SUPABASE

# Verify service_role key (not anon)
# In Supabase settings: Settings → API → service_role secret
```

---

## Performance Targets

| Component | Latency | Cost |
|-----------|---------|------|
| DirectRAGAgent | < 3s | ~$0.001 |
| Full Council (3 agents) | 8-12s | ~$0.008 |
| Phase I Collection | 2-3 min | Free (local) |
| Phase II Embedding | 2-5 min | ~$0.011 |
| Full Evaluation | ~30 min | ~$2.82 |

---

## Next Steps

1. **Setup** (5 min): Install, .env, generate data
2. **Phase I** (3 min): Collect and validate
3. **Phase II** (5 min): Chunk and embed
4. **Phase III - CLI** (interactive): Test with queries
5. **Phase III - Web** (5 min): Start server, explore UI
6. **Evaluation** (30 min): Run full eval or smoke test
7. **Testing** (varies): Run Phase 3-6 tests (see TEST_QUICK_START.md)

**Total time**: ~1 hour to full system operational

