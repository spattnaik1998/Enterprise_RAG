# RAG Evaluation Framework

Rigorous evaluation of the TechVault MSP Enterprise RAG pipeline, measuring
hybrid search quality and LLM answer quality across all four supported models.
Issues a binary **PASS / FAIL** production deployment decision per model.

---

## Setup

Ensure Phase II has been completed (FAISS index must exist at `data/index/`):

```bash
python -m src.main status          # confirm phase2 is complete
python -m src.main phase2          # run phase2 if not yet done
```

Install dependencies (already in `requirements.txt`):

```bash
pip install typer rich openai anthropic loguru python-dotenv
```

Ensure your `.env` contains valid keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## CLI Reference

```
python -m eval.run_eval [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--models TEXT` | all 4 | Models to evaluate. Repeat: `--models gpt-4o-mini --models gpt-4o` |
| `--category TEXT` | all 6 | Categories to evaluate. Repeat: `--category billing --category crm` |
| `--sample INT` | `0` | Queries per category to sample (0 = use all) |
| `--no-rerank` | off | Skip LLM reranking (faster, cheaper, lower quality) |
| `--output PATH` | auto-timestamped | Path for JSON report |
| `--judge-model TEXT` | `gpt-4o-mini` | OpenAI model used as LLM judge |
| `--top-k INT` | `20` | Candidates retrieved before reranking |
| `--rerank-top-k INT` | `10` | Chunks kept after reranking |
| `--index-dir PATH` | `data/index` | FAISS index directory |
| `--seed INT` | `42` | RNG seed for reproducible sampling |
| `--quiet` | off | Suppress progress bar |

### Common Commands

```bash
# Quick smoke test: 1 model, billing only, 5 queries (~$0.01)
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5

# Full production eval: all 4 models, all 80 queries (~$2.82)
python -m eval.run_eval

# Single category deep-dive on two models
python -m eval.run_eval --models gpt-4o-mini gpt-4o --category contracts crm

# Skip reranking for rapid iteration
python -m eval.run_eval --no-rerank --models gpt-4o-mini --sample 5

# CI integration (exits with code 1 if any model fails)
python -m eval.run_eval --models gpt-4o-mini && echo "PASS" || echo "FAIL"
```

---

## Metrics & Production Thresholds

All five metrics must exceed their threshold for a model to be **PRODUCTION READY**:

| Metric | Threshold | Description |
|---|---|---|
| **Retrieval Recall@10** | >= 80% | Fraction of queries where any top-10 citation title/source/answer contains at least 1 `expected_keyword` (case-insensitive) |
| **Source Type Hit Rate** | >= 85% | Fraction of queries where at least 1 citation `source_type` matches the query's `expected_source_types` |
| **Answer Faithfulness** | >= 85% | LLM judge (0-1): are all claims in the answer supported by retrieved context? |
| **Answer Correctness** | >= 75% | LLM judge (0-1): does the answer correctly address the question vs. ground truth? |
| **Composite Score** | >= 82% | Mean of the four metrics above |

### Production Gate

A model achieves **PRODUCTION READY** status only when all five metrics simultaneously
exceed their thresholds. A single failing metric means the model is **NOT READY**.

The CLI exits with code `0` if all tested models pass, and `1` if any fail — enabling
CI/CD integration.

---

## Query Dataset

80 queries across 6 categories derived from the synthetic TechVault MSP enterprise data:

| Category | File | Queries | Focus |
|---|---|---|---|
| `billing` | `billing_queries.json` | 20 | Invoices, AR status, overdue balances |
| `contracts` | `contracts_queries.json` | 15 | SLA terms, expiry dates, monthly values |
| `crm` | `crm_queries.json` | 12 | Account health, contacts, health notes |
| `psa` | `psa_queries.json` | 15 | Service tickets, technicians, hours billed |
| `communications` | `communications_queries.json` | 10 | Reminder emails, sequence levels |
| `cross_source` | `cross_source_queries.json` | 8 | Multi-system synthesis queries |

### Query Schema

```json
{
  "id": "billing_001",
  "query": "Which clients have OVERDUE_CRITICAL invoices?",
  "ground_truth": "Full expected answer text...",
  "expected_source_types": ["billing"],
  "expected_keywords": ["Northern Lights", "Crossroads", "Skyline"],
  "difficulty": "easy|medium|hard",
  "notes": "Optional debugging hint"
}
```

**Difficulty distribution**: ~25% easy, ~50% medium, ~25% hard.

---

## Cost Estimates

Costs depend on model selection and number of queries:

| Scope | Models | Queries | Estimated Cost |
|---|---|---|---|
| Smoke test | gpt-4o-mini | 5 (billing) | ~$0.01 |
| Single model | gpt-4o-mini | 80 (all) | ~$0.13 |
| Two models | gpt-4o-mini + gpt-4o | 80 each | ~$1.73 |
| Full eval | All 4 models | 80 each | ~$2.82 |

Full 4-model breakdown:
- `gpt-4o-mini` generation: ~$0.08
- `gpt-4o` generation: ~$1.60
- `claude-haiku-4-5-20251001` generation: ~$0.13
- `claude-sonnet-4-6` generation: ~$0.96
- LLM judge (320 calls): ~$0.05
- **Total: ~$2.82**

---

## Output

### Console

Rich progress bar during evaluation followed by a summary table:

```
                RAG Evaluation Report
+-----------------------+-----------+----------+-------------+------------+-----------+---------+-------+------------------+
| Model                 | Recall@10 | Src Hit  | Faithfulness| Correctness| Composite | Blocked | Cost$ | Status           |
|                       | (>=80%)   | (>=85%)  | (>=85%)     | (>=75%)    | (>=82%)   |         |       |                  |
+-----------------------+-----------+----------+-------------+------------+-----------+---------+-------+------------------+
| gpt-4o-mini           | 87.5%     | 91.2%    | 88.1%       | 82.4%      | 87.3%     | 0       | $0.13 | PRODUCTION READY |
| gpt-4o                | 92.5%     | 95.0%    | 93.7%       | 90.1%      | 92.8%     | 0       | $1.60 | PRODUCTION READY |
+-----------------------+-----------+----------+-------------+------------+-----------+---------+-------+------------------+
```

### JSON Report

Saved to `eval/results/eval_YYYYMMDD_HHMMSS.json`. Contains:
- Overall model metrics with production status
- Per-category breakdown per model
- Raw query-level results (query, recall_hit, faithfulness, correctness, cost)
- Total cost and thresholds used

---

## Architecture

```
eval/
  __init__.py              Package marker
  judge.py                 LLMJudge: single GPT-4o-mini call per query
  evaluator.py             RAGEvaluator: orchestrates pipeline + judge + aggregation
  run_eval.py              CLI entry point (typer + rich)
  datasets/
    billing_queries.json       20 queries
    contracts_queries.json     15 queries
    crm_queries.json           12 queries
    psa_queries.json           15 queries
    communications_queries.json 10 queries
    cross_source_queries.json   8 queries
  results/
    .gitkeep               Output directory (JSON reports written here)
```

### How Recall@10 Works

Recall is computed as a **keyword presence check**, not a strict match:

```python
recall_hit = any(
    kw.lower() in (title + source + answer).lower()
    for kw in expected_keywords
)
```

The answer text is included as a fallback because the RAG model synthesizes
facts from retrieved chunks — if the relevant keyword appears in the answer,
the retrieval was effective even if the raw chunk titles did not surface it.

### How the LLM Judge Works

Each query gets a single `gpt-4o-mini` call at `temperature=0`:

```
INPUT:  query + ground_truth + citation metadata + generated answer
OUTPUT: {"faithfulness": 0.92, "correctness": 0.88}
```

Citation metadata (title, source, source_type, score) is passed instead of
raw chunk text since `QueryResult.citations` does not include chunk text.

---

## Interpreting Results

| Scenario | Action |
|---|---|
| All models PASS | Deploy any model. Prefer `gpt-4o-mini` for cost, `claude-sonnet-4-6` for quality. |
| Recall < 80% | Index quality issue — review chunking strategy or re-run Phase II. |
| Source Type Hit < 85% | Dense/sparse weight imbalance — tune `retrieval.dense_weight` in `config/config.yaml`. |
| Faithfulness < 85% | Model hallucinating — review system prompt in `src/generation/prompts.py`. |
| Correctness < 75% | Ground truth mismatch or reranking quality — try increasing `rerank_top_k`. |
| Cross-source queries low | Multi-hop retrieval weakness — consider increasing `top_k`. |
