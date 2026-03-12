# Latency Testing Guide

Comprehensive latency analysis suite for RAG and Agent systems with 40 test queries across 4 complexity levels.

## Quick Start

### Test RAG System (Simple Queries)
```bash
python -m eval.latency_benchmark --mode rag --complexity simple
```

### Test Agent System (All Queries)
```bash
python -m eval.latency_benchmark --mode agent --complexity all --sample 5
```

### Test Both Systems (Moderate Complexity)
```bash
python -m eval.latency_benchmark --mode both --complexity moderate
```

### Full Benchmark with Custom Output
```bash
python -m eval.latency_benchmark --mode both --complexity all --output my_results.json
```

## Query Categories

### Simple Queries (8 queries)
**Expected Latency**: RAG ~800ms, Agent ~7s

Direct lookups and single-source queries:
- `simple_001`: "List all overdue invoices" (billing)
- `simple_002`: "What invoices are more than 60 days overdue?" (billing)
- `simple_003`: "Show me all open service tickets" (PSA)
- `simple_004`: "Which clients have RED account health?" (CRM)
- `simple_005`: "List all active contracts" (contracts)
- `simple_006`: "Show recent invoice reminders" (communications)
- `simple_007`: "What is the total outstanding balance?" (billing)
- `simple_008`: "Which unresolved tickets are assigned to a technician?" (PSA)

**Best For**: Baseline performance, cache efficiency, single-model testing

---

### Moderate Queries (8 queries)
**Expected Latency**: RAG ~1.2s, Agent ~10s

Multi-hop reasoning and dual-source queries:
- `moderate_001`: "Which clients have overdue invoices AND RED account health?" (billing + CRM)
- `moderate_002`: "What is the renewal status for Alpine Financial?" (contracts)
- `moderate_003`: "Show me high-priority tickets from clients with YELLOW health" (PSA + CRM)
- `moderate_004`: "Which clients have overdue invoices exceeding $10,000?" (billing)
- `moderate_005`: "List clients with expiring contracts in the next 90 days" (contracts)
- `moderate_006`: "What are the SLA response times for critical service contracts?" (contracts)
- `moderate_007`: "How many unresolved tickets do we have per client?" (PSA)
- `moderate_008`: "Which clients received invoice reminders but haven't paid?" (billing + communications)

**Best For**: Real-world scenarios, cost analysis, decision support testing

---

### Complex Queries (8 queries)
**Expected Latency**: RAG ~2.0s, Agent ~14s

Multi-source aggregation and complex reasoning:
- `complex_001`: "Which clients have overdue invoices, RED account health, AND unresolved high-priority tickets?" (3 sources)
- `complex_002`: "Show me clients at risk of churn: overdue invoices, declining health, and no recent successful ticket resolutions" (3 sources)
- `complex_003`: "Provide a client 360 view for Northern Lights Healthcare: invoices, tickets, contracts, and contact info" (4 sources)
- `complex_004`: "Which clients are due for contract renewal AND have RED health AND have more than 5 unresolved tickets?" (3 sources)
- `complex_005`: "Identify escalation candidates: clients with high-value contracts, critical SLAs, AND recent support issues" (3 sources)
- `complex_006`: "Calculate revenue at risk: show clients with overdue invoices over $5k who also have expiring contracts" (2 sources)
- `complex_007`: "What is the health trend for clients with large outstanding balances? Who should we reach out to?" (3 sources)
- `complex_008`: "Show me the top 3 at-risk accounts by composite score: (invoice aging + health status + ticket backlog + contract renewal)" (4 sources)

**Best For**: Executive reporting, risk analysis, CEO dashboards, decision intelligence

---

### Edge Cases (8 queries)
**Expected Latency**: RAG ~600ms, Agent ~6s

Boundary conditions and error handling:
- `edge_001`: "Show me clients with exactly zero invoices" (boundary condition)
- `edge_002`: "What is the oldest overdue invoice?" (aggregation)
- `edge_003`: "Show me contracts with auto-renewal disabled" (boolean filter)
- `edge_004`: "Which clients have both RED AND GREEN accounts simultaneously?" (logical impossibility)
- `edge_005`: "List tickets with zero hours billed" (boundary condition)
- `edge_006`: "Show me contracts expiring in the past" (historical)
- `edge_007`: "What is the status of a non-existent client ABC123?" (not found)
- `edge_008`: "Find invoices with negative amounts" (credit memos/refunds)

**Best For**: Error handling, edge case coverage, robustness testing

---

## Benchmark Modes

### RAG Mode
Tests the direct RAG pipeline:
- Query → Retriever → Reranker → Generator → Result
- Measures: latency, citations, token count, cost
- Single LLM call (generation only)
- ~800ms to ~2s per query

```bash
python -m eval.latency_benchmark --mode rag --complexity simple --sample 8
```

### Agent Mode
Tests the 3-agent Council pattern:
- Query → FastCreative (parallel with ConservativeChecker) → PolicyVerifier
- Measures: latency, winning agent, escalations, cost
- 2-3 LLM calls per query
- ~6s to ~14s per query

```bash
python -m eval.latency_benchmark --mode agent --complexity complex --sample 4
```

### Both Modes
Runs both RAG and Agent on all queries (sequential):
- Provides direct latency comparison
- Shows cost delta between architectures
- Total time: ~40-60 minutes for all 40 queries

```bash
python -m eval.latency_benchmark --mode both --complexity all
```

---

## Output Metrics

### Per-Query Results

**RAG Output**:
```json
{
  "query_id": "simple_001",
  "query": "List all overdue invoices",
  "category": "billing",
  "success": true,
  "latency_ms": 847.3,
  "answer_length": 1245,
  "citation_count": 5,
  "cost_usd": 0.000342,
  "tokens": 1894
}
```

**Agent Output**:
```json
{
  "query_id": "simple_001",
  "query": "List all overdue invoices",
  "category": "billing",
  "success": true,
  "latency_ms": 7142.1,
  "answer_length": 1389,
  "winning_agent": "ConservativeChecker",
  "escalated": false,
  "cost_usd": 0.00521
}
```

### Summary Statistics

```
System     Queries   Success Rate   Mean Latency (ms)   P95 Latency (ms)   Total Cost (USD)
RAG        8         100%           921                 1847               0.002734
Agent      8         100%           8456                13921              0.038645
```

---

## Performance Targets

### SLA Targets

| Complexity | RAG (ms) | Agent (ms) | RAG P95 | Agent P95 | RAG Cost | Agent Cost |
|------------|----------|-----------|---------|-----------|----------|-----------|
| Simple     | < 800    | < 7,000   | < 1,500 | < 12,000  | < $0.001 | < $0.005  |
| Moderate   | < 1,200  | < 10,000  | < 2,000 | < 15,000  | < $0.003 | < $0.010  |
| Complex    | < 2,000  | < 14,000  | < 3,500 | < 20,000  | < $0.005 | < $0.020  |

### Success Targets
- Success Rate: >= 95% (all queries must return answers)
- Error Rate: <= 5% (timeouts, API failures)
- Hallucination Rate (Agent): <= 10% (detected by PolicyVerifier)

---

## Usage Scenarios

### Scenario 1: Quick Smoke Test (5 minutes)
```bash
python -m eval.latency_benchmark --mode both --complexity simple --sample 3
```

**Output**:
- 3 RAG queries (2-3 seconds)
- 3 Agent queries (20-25 seconds)
- Cost: ~$0.02 total
- Verifies both systems are working

### Scenario 2: Complexity Analysis (15 minutes)
```bash
python -m eval.latency_benchmark --mode both --complexity all --sample 2
```

**Output**:
- 2 queries × 4 complexity levels = 8 queries per system
- Shows latency scaling by complexity
- Cost: ~$0.15 total

### Scenario 3: Full Benchmark Suite (60+ minutes)
```bash
python -m eval.latency_benchmark --mode both --complexity all
```

**Output**:
- All 40 test queries on both RAG and Agent systems
- Comprehensive statistics and outlier detection
- Cost: ~$1.50-$2.00 total
- Full regression test coverage

### Scenario 4: Agent Escalation Analysis (10 minutes)
```bash
python -m eval.latency_benchmark --mode agent --complexity complex --sample 5
```

**Output**:
- Focus on complex, multi-source queries
- Identifies when PolicyVerifier escalates
- Shows decision distribution (FastCreative vs ConservativeChecker wins)

---

## Result Files

Results are saved to `eval/results/latency_benchmark_*.json` with structure:

```json
{
  "timestamp": "2026-03-12T02:45:30.123456",
  "rag_results": [ ... 40 results ... ],
  "agent_results": [ ... 40 results ... ],
  "summary": {
    "rag": {
      "total_queries": 40,
      "successful": 40,
      "success_rate": 1.0,
      "latency_ms": {
        "min": 612.3,
        "max": 3421.1,
        "mean": 1456.7,
        "p50": 1234.5,
        "p95": 2876.3
      },
      "cost_usd": {
        "total": 0.154321,
        "mean": 0.003858
      }
    },
    "agent": { ... similar structure ... }
  }
}
```

---

## Analyzing Results

### Key Metrics to Track

1. **Latency Percentiles**
   - P50: Typical query performance
   - P95: Worst-case (for SLA planning)
   - Max: Outliers and failures

2. **Cost Per Query**
   - RAG typically cheaper (1 LLM call)
   - Agent more expensive (2-3 LLM calls)
   - Track cost/quality ratio

3. **Success Rate**
   - Should be >= 95%
   - Investigate failures (timeouts, API errors)

4. **Agent Decision Distribution**
   - Track which agent wins most often
   - Identify escalation patterns
   - Monitor PolicyVerifier rejection rate

### Trend Analysis

Compare results over time to detect:
- **Regressions**: Latency increases (API degradation, data growth)
- **Improvements**: Latency decreases (caching, optimization)
- **Cost Changes**: Per-token pricing updates, model changes
- **Reliability**: Success rate trends, error patterns

### Optimization Opportunities

1. **High P95 latency**
   - Consider query routing (simple → RAG, complex → Agent)
   - Increase rerank_top_k for low-recall queries
   - Add caching for frequent queries

2. **High cost per query**
   - Experiment with cheaper models (gpt-4o-mini vs gpt-4o)
   - Reduce context tokens (adjust top_k, rerank_top_k)
   - Consider Agent router (skip Council for simple queries)

3. **Low success rate**
   - Debug timeout causes
   - Check API rate limits
   - Monitor Supabase connection health

---

## Integration with CI/CD

### Pre-deployment Check
```bash
# Run quick smoke test before production deploy
python -m eval.latency_benchmark --mode both --complexity simple --sample 2

# Only proceed if success_rate >= 0.95 and mean latency within SLA
```

### Nightly Regression Test
```bash
# Full benchmark suite every night
python -m eval.latency_benchmark --mode both --complexity all \
  --output results/nightly_$(date +%Y%m%d).json
```

### A/B Testing
```bash
# Test new model or configuration against baseline
python -m eval.latency_benchmark --mode agent --complexity moderate \
  --output results/variant_a.json
# Then compare with baseline results
```

---

## Troubleshooting

### Connection Issues
```python
# If SupabaseIndex fails to connect
export SUPABASE_URL="your_url"
export SUPABASE_SERVICE_KEY="your_key"
python -m eval.latency_benchmark --mode rag --complexity simple --sample 1
```

### Out of Memory
```bash
# Run in batches instead of all at once
python -m eval.latency_benchmark --mode rag --complexity simple
python -m eval.latency_benchmark --mode rag --complexity moderate
python -m eval.latency_benchmark --mode rag --complexity complex
# Then merge results manually
```

### API Rate Limits
```bash
# Slower rate if hitting OpenAI/Anthropic limits
# Add sleep between queries by modifying latency_benchmark.py:
# time.sleep(1)  # Between queries
```

---

## Next Steps

1. **Run smoke test**: `python -m eval.latency_benchmark --mode both --complexity simple --sample 3`
2. **Analyze results**: Check `eval/results/latency_benchmark_*.json`
3. **Compare with targets**: Use SLA table above
4. **Identify bottlenecks**: High latency or high error rate
5. **Optimize**: Adjust parameters, models, or architecture
6. **Re-test**: Run full benchmark to verify improvements
