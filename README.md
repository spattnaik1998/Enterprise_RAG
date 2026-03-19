# TechVault MSP Enterprise RAG Intelligence Hub

A production-grade, 6-stage Retrieval-Augmented Generation (RAG) pipeline purpose-built for MSP (Managed Service Provider) operations. Delivers grounded, cited answers over enterprise back-office data (billing, PSA, CRM, communications, contracts) through a chat UI, REST API, and CLI.

**Status**: ✅ Production Ready | All tests passing | Full documentation included

---

## Quick Start (5 minutes)

```bash
# 1. Setup
pip install -r requirements.txt
cp .env.example .env              # Edit with your API keys
python scripts/generate_enterprise_data.py

# 2. Run pipeline
python -m src.main phase1         # Collect & validate
python -m src.main phase2         # Chunk & embed
python -m src.main phase3 --query "Which clients have overdue invoices?"

# 3. Start web UI (optional)
uvicorn app.server:app --reload --port 8000
# Then open http://localhost:8000
```

---

## Documentation

All detailed instructions are in separate guides:

| Document | Purpose |
|----------|---------|
| **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** | Complete step-by-step instructions for all components |
| **[COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md)** | Copy-paste command reference for all features |
| **[CLAUDE.md](CLAUDE.md)** | Project architecture, schemas, and implementation notes |
| **[TEST_QUICK_START.md](TEST_QUICK_START.md)** | Testing quick reference (6 phases) |
| **[TEST_STRATEGY_V2.md](TEST_STRATEGY_V2.md)** | Comprehensive testing plan with all code examples |
| **[eval/README.md](eval/README.md)** | Evaluation framework documentation |

---

## Key Components

### Phase I: Collection + Validation
```bash
python -m src.main phase1
```
- Collects from 5 MSP enterprise systems
- Validates 7 quality criteria per document
- Output: `data/validated/` (2,180+ documents)

### Phase II: Chunking + Embedding + Indexing
```bash
python -m src.main phase2
```
- Adaptive chunking (keep_whole, fixed_overlap, sentence_window)
- OpenAI embeddings (text-embedding-3-small)
- Dual index: FAISS (dense) + BM25 (sparse)
- Output: `data/index/` (1,996+ vectors)

### Phase III: Query Interface

**CLI Interactive**:
```bash
python -m src.main phase3
```

**Single Query**:
```bash
python -m src.main phase3 --query "Which clients have overdue invoices?"
```

**Web UI + REST API**:
```bash
uvicorn app.server:app --reload --port 8000
# Access: http://localhost:8000
```

**Council Orchestrator (3-agent voting)**:
```bash
python -m src.agents.council_cli --query "Should we escalate Alpine Financial?"
```

---

## Data Sources

### Enterprise MSP Systems (5)
| System | Type | Records | Source |
|--------|------|---------|--------|
| Billing | QuickBooks | 688 invoices | Synthetic |
| PSA | ConnectWise | 732 tickets | Synthetic |
| CRM | HubSpot | 50 profiles | Synthetic |
| Communications | Exchange | 142 emails | Synthetic |
| Contracts | SharePoint | 50 contracts | Synthetic |

**All enterprise data is synthetically generated** (`scripts/generate_enterprise_data.py`). Seed=42 ensures reproducibility.

---

## Query Examples

```bash
# Billing
python -m src.main phase3 --query "Show aged receivables over 60 days"

# PSA/Tickets
python -m src.main phase3 --query "List unresolved support tickets"

# CRM/Health
python -m src.main phase3 --query "Which accounts are at risk?"

# Contracts
python -m src.main phase3 --query "Which contracts expire in the next 30 days?"

# Cross-source
python -m src.main phase3 --query "Get complete view of Alpine Financial"

# Complex (3-agent voting)
python -m src.agents.council_cli --query "Should we escalate this client?"
```

---

## Evaluation

Rigorous quality assurance across 4 LLM models and 80 ground-truth queries:

```bash
# Full evaluation (all 4 models, 80 queries, ~$2.82)
python -m eval.run_eval

# Quick smoke test (1 model, 5 queries, ~$0.01)
python -m eval.run_eval --models gpt-4o-mini --sample 5 --no-rerank

# Multi-model comparison
python -m eval.run_eval --models gpt-4o-mini gpt-4o --category contracts
```

**Production Thresholds**:
- Recall@10: ≥ 80%
- Source Type Hit: ≥ 85%
- Faithfulness: ≥ 85%
- Correctness: ≥ 75%
- Composite: ≥ 82%

See [eval/README.md](eval/README.md) for complete evaluation documentation.

---

## Architecture

### Query Lifecycle (Phase III)

```
User Query
    ↓
PromptGuard            (13 regex patterns → injection detection)
    ↓
HybridRetriever        (FAISS dense + BM25 sparse → RRF fusion)
    ↓
LLMReranker            (OpenAI → relevance scoring)
    ↓
RAGGenerator           (OpenAI or Anthropic → grounded answer)
    ↓
PIIFilter              (redact email/phone/SSN/CC/IP)
    ↓
QueryResult            (answer + citations + latency + cost)
```

### Multi-Agent Architectures

1. **Council Orchestrator** — 3-agent voting (FastCreative, ConservativeChecker, PolicyVerifier)
2. **Query Router** — Smart routing (SIMPLE → DirectRAG, COMPLEX → Council, AGGREGATE → ToolComposer)
3. **MCP Server** — 17 tools for research + enterprise data access

---

## Performance

| Operation | Latency | Cost |
|-----------|---------|------|
| Simple query (DirectRAG) | < 3s | ~$0.001 |
| Complex query (Council) | 8-12s | ~$0.008 |
| Phase I (collection) | 2-3 min | Free |
| Phase II (embedding) | 2-5 min | ~$0.011 |
| Full evaluation | ~30 min | ~$2.82 |

---

## Environment Variables

**Required**:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Optional (Web/Supabase)**:
```
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

**Optional (Tracing)**:
```
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
```

See [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) for complete environment setup.

---

## Key Features

✅ **Dual-Index Hybrid Search** — FAISS (dense) + BM25 (sparse) with RRF fusion
✅ **Adaptive Chunking** — Automatic strategy selection by source type
✅ **LLM Reranking** — Single call scores all candidates
✅ **Citation Tracking** — Full provenance for every answer
✅ **PII Redaction** — Automatic masking of sensitive data
✅ **Multi-Model Support** — OpenAI + Anthropic (4 models total)
✅ **3-Agent Voting** — Council consensus for complex queries
✅ **REST API + Web UI** — Production-ready serving
✅ **MCP Server** — Claude Desktop integration
✅ **Comprehensive Eval** — 80 queries × 4 models → pass/fail gate
✅ **Security** — ABAC policies, audit logging, step-up auth
✅ **Observability** — LangSmith tracing, quality monitoring

---

## Testing

All changes are tested across 6 phases:

```bash
# Phase 1-2: Smoke + unit tests (5-15 min, no dependencies)
python -c "from src.agents.router import RouteType; print('OK')"

# Phase 3: Integration tests (20 min, requires FAISS)
python -m src.main phase3 --query "test query"

# Phase 4: End-to-end tests (30 min, requires API keys)
python -m src.agents.council_cli --query "test query"

# Phase 5: Regression tests (10 min)
python -m eval.run_eval --models gpt-4o-mini --sample 1

# Phase 6: Performance benchmark (10 min, optional)
# Compare DirectRAG vs Council latency/cost
```

See [TEST_QUICK_START.md](TEST_QUICK_START.md) for all testing commands.

---

## Recent Updates (Review Brief v2)

✅ Removed context rot (archived research source references)
✅ Fixed DirectRAGAgent kwargs bug
✅ Added Skills → Router integration
✅ Enforced AUDIT_HMAC_KEY compliance
✅ Fixed slowapi rate limiter issues

Commits: `5cb226b`, `ec1904a`, `f1aed31`

---

## Getting Help

```bash
# View all available commands
cat COMMANDS_CHEATSHEET.md

# Detailed instructions for any component
cat EXECUTION_GUIDE.md

# Testing reference
cat TEST_QUICK_START.md

# Project architecture & guidelines
cat CLAUDE.md

# Evaluation framework details
cat eval/README.md
```

---

## Repository Structure

```
Enterprise_RAG/
├── EXECUTION_GUIDE.md          Complete instructions (all components)
├── COMMANDS_CHEATSHEET.md      Quick command reference
├── TEST_QUICK_START.md         Testing quick start
├── TEST_STRATEGY_V2.md         Comprehensive testing plan
├── CLAUDE.md                   Project guidelines & architecture
├── README.md                   This file
│
├── src/                        Core pipeline
│   ├── main.py                 CLI entry point (phases 1, 2, 3)
│   ├── serving/                RAG pipeline orchestrator
│   ├── collection/             8 data collectors
│   ├── validation/             Quality checks
│   ├── chunking/               Adaptive chunking strategies
│   ├── embedding/              FAISS index management
│   ├── retrieval/              Hybrid retrieval + reranking
│   ├── generation/             LLM-based answer synthesis
│   ├── agents/                 Multi-agent architectures
│   ├── context/                Context management + freshness scoring
│   ├── security/               ABAC policies + audit logging
│   ├── skills/                 Domain-specific skill agents
│   └── observability/          Tracing + quality monitoring
│
├── app/                        Web UI + REST API
│   ├── server.py               FastAPI application
│   ├── chat_logger.py          Chat interaction logging
│   └── static/                 Frontend assets
│
├── eval/                       Evaluation framework
│   ├── README.md               Evaluation documentation
│   ├── run_eval.py             CLI entry point
│   ├── judge.py                LLMJudge for quality assessment
│   ├── evaluator.py            RAG evaluation orchestrator
│   └── datasets/               Ground truth queries (80 total)
│
├── data/
│   ├── enterprise/             Synthetic MSP data (generated)
│   ├── index/                  FAISS index (created by phase2)
│   ├── validated/              Phase I output
│   ├── rejected/               Phase I discards
│   └── audit/                  Security audit logs
│
├── config/
│   ├── config.yaml             Pipeline parameters
│   ├── policies.yaml           Security ABAC policies
│   └── prompts.yaml            System prompt configurations
│
├── scripts/
│   ├── generate_enterprise_data.py  Synthetic data generation
│   └── migrate_to_supabase.py       Supabase migration
│
├── requirements.txt            Python dependencies
├── .env.example                Environment template
└── .gitignore                  Git ignore rules
```

---

## License

Internal project. All code © 2024 TechVault.

---

## Next Steps

1. **Setup** → `pip install -r requirements.txt && cp .env.example .env`
2. **Generate Data** → `python scripts/generate_enterprise_data.py`
3. **Run Phases** → `phase1 → phase2 → phase3`
4. **Test** → See [TEST_QUICK_START.md](TEST_QUICK_START.md)
5. **Evaluate** → `python -m eval.run_eval`

For detailed instructions, see [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md).

---

**Status**: ✅ Production Ready | Latest Update: March 2026
