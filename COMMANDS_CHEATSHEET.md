# Enterprise RAG — Commands Cheat Sheet

## Setup (One-Time)
```bash
pip install -r requirements.txt
cp .env.example .env              # Edit with your API keys
python scripts/generate_enterprise_data.py
```

---

## Phase I: Collection + Validation
```bash
python -m src.main phase1                    # Full pipeline with checks
python -m src.main phase1 --dry-run          # Config validation only
python -m src.main phase1 --skip-health      # Skip health checks
```

---

## Phase II: Chunking + Embedding + Indexing
```bash
python -m src.main phase2                    # Build FAISS index
```

---

## Phase III: Query Interface (CLI)

### Interactive Mode
```bash
python -m src.main phase3
```
Then type queries and get answers.

### Single Query
```bash
# Standard output
python -m src.main phase3 --query "Which clients have overdue invoices?"

# JSON output
python -m src.main phase3 --query "What is ClientA's status?" --json

# With custom parameters
python -m src.main phase3 --query "..." --top-k 20 --rerank-top-k 10 --no-rerank
```

### Query Examples
```bash
# Billing
python -m src.main phase3 --query "Show clients with 30+ day overdue"

# PSA
python -m src.main phase3 --query "List unresolved support tickets"

# CRM
python -m src.main phase3 --query "Which accounts are at risk?"

# Contracts
python -m src.main phase3 --query "Which contracts expire soon?"

# Cross-source
python -m src.main phase3 --query "Get full profile of Alpine Financial"
```

---

## Web UI + API
```bash
# Start server
uvicorn app.server:app --reload --port 8000

# Access endpoints
# Chat UI:     http://localhost:8000
# Forecast:    http://localhost:8000/forecast
# Logs:        http://localhost:8000/logs
# API Docs:    http://localhost:8000/docs

# API calls
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"...", "provider":"openai", "model":"gpt-4o-mini"}'

curl http://localhost:8000/api/health
curl http://localhost:8000/api/clients
curl "http://localhost:8000/api/logs?limit=50"
```

---

## Council Orchestrator (3-Agent Voting)
```bash
# Interactive mode
python -m src.agents.council_cli

# Single query
python -m src.agents.council_cli --query "Should we escalate Alpine Financial?"
```

---

## Evaluation Framework

### Full Evaluation
```bash
# All 4 models, all 80 queries (~$2.82, 30 min)
python -m eval.run_eval

# Smoke test (~$0.01, 2 min)
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5 --no-rerank

# Multi-model comparison
python -m eval.run_eval --models gpt-4o-mini gpt-4o --sample 10

# Custom settings
python -m eval.run_eval \
  --models gpt-4o-mini \
  --category contracts \
  --sample 5 \
  --top-k 30 \
  --rerank-top-k 15 \
  --judge-model gpt-4o
```

**Categories**: billing, contracts, crm, psa, communications, cross_source

---

## MCP Server
```bash
python -m src.collection.mcp.server
```
Use with Claude Desktop or custom MCP clients (17 tools available).

---

## System Status
```bash
python -m src.main status
```

---

## Supabase Migration
```bash
python scripts/migrate_to_supabase.py        # Full migration
python scripts/migrate_to_supabase.py --only chunks  # Chunks only
```

---

## Testing

### Phase 1-2: Smoke + Unit Tests
```bash
python -c "
from src.agents.router import RouteType
from src.security.gateway import AgentSecurityGateway
print('Imports OK')
"
```

### Phase 3: Integration Tests
```bash
python << 'EOF'
import asyncio
from src.serving.pipeline import RAGPipeline
from src.agents.router import DirectRAGAgent

pipeline = RAGPipeline(index_dir="data/index")
agent = DirectRAGAgent(pipeline)

async def test():
    verdict = await agent.run("What clients have overdue invoices?")
    print(f"PASS: {verdict.latency_ms:.0f}ms")

asyncio.run(test())
EOF
```

### Phase 4: End-to-End Tests
```bash
# NO_CONTEXT_RESPONSE test
python -m src.main phase3 --query "What is RAG?"

# Router logging test
python -m src.agents.council_cli --query "What is ClientA's invoice total?"

# Skill test
python << 'EOF'
import asyncio
from src.skills.registry import SkillRegistry
from src.skills.base import SkillContext

async def test():
    skill = SkillRegistry().get('ar_risk_report')
    if skill:
        result = await skill.execute(SkillContext(query="Generate AR risk report"))
        print(f"PASS: {result.latency_ms:.0f}ms")

asyncio.run(test())
EOF
```

### Phase 5: Regression Tests
```bash
python -m src.main phase3 --query "What is ClientA's status?"
python -m src.agents.council_cli --query "Should we escalate?"
python -m eval.run_eval --models gpt-4o-mini --sample 1 --no-rerank
```

---

## Common Workflows

### Quick Query + Check Cost
```bash
python -m src.main phase3 --query "Which clients have overdue invoices?" --json | jq '.estimated_cost_usd'
```

### Monitor Real-Time
```bash
# Terminal 1: Start server
uvicorn app.server:app --reload --port 8000

# Terminal 2: Watch logs
tail -f data/audit/audit.jsonl | jq '.'
```

### Batch Query Processing
```bash
# Create queries.txt
cat > queries.txt << 'EOF'
Which clients have overdue invoices?
What is Alpine Financial's account health?
List all expiring contracts
EOF

# Process each
while read query; do
  python -m src.main phase3 --query "$query" --json
done < queries.txt
```

### Cost Estimation
```bash
# Smoke test before full eval
python -m eval.run_eval --models gpt-4o-mini --sample 1 --no-rerank
# Output: $0.0003
# Full eval: $0.0003 × 80 = $0.024

# Compare model costs
python -m eval.run_eval --models gpt-4o-mini gpt-4o --sample 5 --no-rerank
```

### A/B Test Reranking
```bash
# Without reranking (faster)
python -m eval.run_eval --models gpt-4o-mini --sample 5 --no-rerank

# With reranking (higher quality)
python -m eval.run_eval --models gpt-4o-mini --sample 5
```

---

## Environment Variables

### Required
```bash
OPENAI_API_KEY=sk-...              # Embeddings, reranking, generation
ANTHROPIC_API_KEY=sk-ant-...       # Claude models
```

### Optional (Web/Supabase)
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...        # Must be service_role, not anon
```

### Optional (Tracing)
```bash
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="Enterprise RAG"
```

### Optional (Other)
```bash
ENVIRONMENT=dev|staging|production
AUDIT_HMAC_KEY=...                 # Required in production
```

---

## Quick Diagnostics

```bash
# Check Python version
python --version               # Should be 3.11+

# Test imports
python -c "import faiss; import torch; print('OK')"

# Check API keys
python -c "import os; print(os.environ.get('OPENAI_API_KEY', 'NOT SET')[:20])"

# Check FAISS index
python -c "
from src.embedding.faiss_index import FAISSIndex
from pathlib import Path
idx = FAISSIndex.load(Path('data/index'))
print(f'Index ready: {idx.ntotal} vectors')
"

# Test RAG pipeline
python -c "
from src.serving.pipeline import RAGPipeline
p = RAGPipeline(index_dir='data/index')
result = p.query('What is ClientA?')
print(f'Answer length: {len(result.answer)} chars')
"

# Test OpenAI connection
python -c "
from openai import OpenAI
client = OpenAI()
response = client.models.list()
print(f'OpenAI OK: {len(list(response.data))} models available')
"

# Test Anthropic connection
python -c "
from anthropic import Anthropic
client = Anthropic()
models = client.models.list()
print(f'Anthropic OK: models available')
"
```

---

## Flags Reference

### Phase III Flags
```bash
--query TEXT              # Single-shot query (omit for interactive)
--top-k INT              # Retrieval candidates (default: 10)
--rerank-top-k INT       # After reranking (default: 5)
--model TEXT             # Generator model (default: gpt-4o-mini)
--no-rerank              # Skip LLM reranking
--no-pii                 # Disable PII redaction
--json                   # JSON output
```

### Eval Flags
```bash
--models TEXT            # Space-separated: gpt-4o-mini gpt-4o ...
--category TEXT          # billing contracts crm psa communications cross_source
--sample INT             # Queries per category (0 = all)
--no-rerank              # Skip reranking
--output PATH            # Output JSON file
--judge-model TEXT       # Judge model (default: gpt-4o-mini)
--top-k INT              # Retrieval candidates (default: 20)
--rerank-top-k INT       # After reranking (default: 10)
--seed INT               # RNG seed (default: 42)
--quiet                  # Suppress progress bar
```

---

## Performance Targets

| Operation | Latency | Cost |
|-----------|---------|------|
| Simple query (DirectRAG) | < 3s | $0.0008 |
| Complex query (Council) | 8-12s | $0.0080 |
| Phase I | 2-3 min | Free |
| Phase II | 2-5 min | $0.011 |
| Full Eval | ~30 min | $2.82 |
| Smoke Eval | ~2 min | $0.01 |

---

## Key Files

| File | Purpose |
|------|---------|
| `EXECUTION_GUIDE.md` | Detailed instructions (this doc) |
| `TEST_QUICK_START.md` | Testing quick reference |
| `TEST_STRATEGY_V2.md` | Full testing plan |
| `CLAUDE.md` | Project guidelines |
| `.env` | API keys (create from .env.example) |
| `data/index/` | FAISS index (created by Phase II) |
| `data/enterprise/` | Synthetic data (created by setup script) |
| `data/validated/` | Phase I output |
| `eval/results/` | Evaluation reports |
| `app/` | Web UI and API |
| `src/main.py` | CLI entry point |

---

## Useful One-Liners

```bash
# Count documents
find data/validated -name "*.json" -type f | wc -l

# Check index size
du -sh data/index/

# List all API endpoints
python -c "
from app.server import app
for route in app.routes:
    print(f'{route.methods} {route.path}')
" | grep api

# View recent costs
jq '.total_cost_usd' eval/results/*.json | sort -rn | head

# Check latest audit log
tail -5 data/audit/audit.jsonl | jq '.'

# Export chat history
jq '.messages[]' data/chat_logs.jsonl > chat_export.json

# Count errors
grep -c "ERROR" data/*.log

# View system load during eval
time python -m eval.run_eval --sample 1

# Estimate full eval cost
python -c "
cost_per_query = 0.0015
total_queries = 80
print(f'Estimated: ${cost_per_query * total_queries:.2f}')
"
```

---

## Getting Help

```bash
# Show all CLI commands
python -m src.main --help

# Show Phase III options
python -m src.main phase3 --help

# Show eval options
python -m eval.run_eval --help

# API documentation
# Open http://localhost:8000/docs (requires running server)

# View code examples
grep -r "example\|Example" src/ --include="*.py" | head -20
```

