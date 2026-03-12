# Quick Query Reference Card

## Test Queries by Category

### SIMPLE QUERIES (Expected: RAG 800ms, Agent 7s)

**Billing**
- `List all overdue invoices`
- `What invoices are more than 60 days overdue?`
- `What is the total outstanding balance?`

**PSA/Tickets**
- `Show me all open service tickets`
- `Which unresolved tickets are assigned to a technician?`

**CRM**
- `Which clients have RED account health?`

**Contracts**
- `List all active contracts`

**Communications**
- `Show recent invoice reminders`

---

### MODERATE QUERIES (Expected: RAG 1.2s, Agent 10s)

**Billing**
- `Which clients have overdue invoices AND RED account health?`
- `Which clients have overdue invoices exceeding $10,000?`
- `Which clients received invoice reminders but haven't paid?`

**Contracts**
- `What is the renewal status for Alpine Financial?`
- `List clients with expiring contracts in the next 90 days`
- `What are the SLA response times for critical service contracts?`

**PSA**
- `Show me high-priority tickets from clients with YELLOW health`
- `How many unresolved tickets do we have per client?`

---

### COMPLEX QUERIES (Expected: RAG 2.0s, Agent 14s)

**High-Value**
- `Provide a client 360 view for Northern Lights Healthcare: invoices, tickets, contracts, and contact info`
- `Show me the top 3 at-risk accounts by composite score: (invoice aging + health status + ticket backlog + contract renewal)`

**Risk Analysis**
- `Which clients have overdue invoices, RED account health, AND unresolved high-priority tickets?`
- `Show me clients at risk of churn: overdue invoices, declining health, and no recent successful ticket resolutions`
- `Calculate revenue at risk: show clients with overdue invoices over $5k who also have expiring contracts`

**Escalation**
- `Which clients are due for contract renewal AND have RED health AND have more than 5 unresolved tickets?`
- `Identify escalation candidates: clients with high-value contracts, critical SLAs, AND recent support issues`

**Outreach**
- `What is the health trend for clients with large outstanding balances? Who should we reach out to?`

---

### EDGE CASES (Expected: RAG 600ms, Agent 6s)

- `Show me clients with exactly zero invoices`
- `What is the oldest overdue invoice?`
- `Show me contracts with auto-renewal disabled`
- `Which clients have both RED AND GREEN accounts simultaneously?`
- `List tickets with zero hours billed`
- `Show me contracts expiring in the past`
- `What is the status of a non-existent client ABC123?`
- `Find invoices with negative amounts`

---

## Run Commands Cheat Sheet

```bash
# Quick test (2 min)
python -m eval.latency_benchmark --mode both --complexity simple --sample 2

# Smoke test (5 min)
python -m eval.latency_benchmark --mode both --complexity simple

# Single complexity level (10 min)
python -m eval.latency_benchmark --mode both --complexity moderate

# All levels - short run (15 min)
python -m eval.latency_benchmark --mode both --complexity all --sample 2

# Full suite (90 min)
python -m eval.latency_benchmark --mode both --complexity all

# RAG only (faster)
python -m eval.latency_benchmark --mode rag --complexity all --sample 5

# Agent only (slower)
python -m eval.latency_benchmark --mode agent --complexity complex

# Custom output file
python -m eval.latency_benchmark --mode both --complexity all \
  --output my_benchmark_$(date +%Y%m%d).json
```

---

## Expected Results

| Test | RAG Mean | RAG P95 | Agent Mean | Agent P95 | RAG Cost | Agent Cost |
|------|----------|---------|-----------|-----------|----------|-----------|
| Simple (8q) | 850ms | 1,500ms | 7,200ms | 12,000ms | $0.003 | $0.035 |
| Moderate (8q) | 1,100ms | 2,000ms | 9,800ms | 15,000ms | $0.008 | $0.065 |
| Complex (8q) | 1,900ms | 3,500ms | 13,200ms | 20,000ms | $0.015 | $0.110 |
| Edge (8q) | 750ms | 1,200ms | 6,500ms | 10,000ms | $0.002 | $0.025 |
| **ALL (32q)** | **1,150ms** | **2,400ms** | **9,175ms** | **15,000ms** | **$0.028** | **$0.235** |

---

## Top Use Cases

### 1. **Billing Risk Queries** (CEO/CFO Dashboard)
- "Which clients have overdue invoices exceeding $10,000?"
- "Calculate revenue at risk: show clients with overdue invoices over $5k who also have expiring contracts"
- **Best For**: Revenue forecasting, collections strategy

### 2. **Client Health Queries** (Account Managers)
- "Which clients have RED account health?"
- "Show me clients at risk of churn: overdue invoices, declining health, and no recent successful ticket resolutions"
- **Best For**: Retention, upsell opportunities

### 3. **Escalation Queries** (Support Managers)
- "Which clients have overdue invoices, RED account health, AND unresolved high-priority tickets?"
- "Identify escalation candidates: clients with high-value contracts, critical SLAs, AND recent support issues"
- **Best For**: Incident response, SLA management

### 4. **Contract Renewal Queries** (Sales/Legal)
- "List clients with expiring contracts in the next 90 days"
- "Which clients are due for contract renewal AND have RED health AND have more than 5 unresolved tickets?"
- **Best For**: Pipeline planning, renewal strategy

### 5. **Performance Baseline Queries** (Operations)
- `simple_001`: "List all overdue invoices"
- `moderate_001`: "Which clients have overdue invoices AND RED account health?"
- `complex_001`: Multi-source aggregation
- **Best For**: System health checks, regression testing

---

## How to Use RAG vs Agent

| Use Case | Recommended | Latency | Cost |
|----------|-------------|---------|------|
| **Quick lookup** (simple_001) | RAG | ~850ms | ~$0.0004 |
| **Single condition filter** (simple_004) | RAG | ~750ms | ~$0.0003 |
| **Two-source join** (moderate_001) | RAG | ~1.1s | ~$0.001 |
| **Business decision** (complex_002) | Agent | ~13s | ~$0.008 |
| **High-stakes escalation** (complex_001) | Agent | ~13s | ~$0.008 |
| **Requires verification** (complex_003) | Agent | ~13s | ~$0.008 |

---

## Latency Tips

### Make Queries Faster

1. **Keep it simple**: Simple queries run 10x faster than complex
2. **Use RAG**: RAG is 10x faster than Agent for single-source queries
3. **Cache results**: Re-run the same query within 5 minutes
4. **Limit scope**: "overdue > 60 days" faster than "overdue > 30 days"

### Monitor Performance

1. **P95 > 3s (RAG)**: Check data size, network latency
2. **P95 > 20s (Agent)**: Check API rate limits, PolicyVerifier rejections
3. **Cost > $0.01**: Consider Agent router to skip Council for simple queries
4. **Error rate > 5%**: Investigate API failures, timeout settings

---

## Files for Reference

- **Full test suite**: `eval/test_queries.json` (40 queries)
- **Benchmark script**: `eval/latency_benchmark.py`
- **Detailed guide**: `LATENCY_TESTING_GUIDE.md`
- **Results location**: `eval/results/latency_benchmark_*.json`

---

## Next Steps

1. **Run smoke test**
   ```bash
   python -m eval.latency_benchmark --mode both --complexity simple
   ```

2. **Check results**
   ```bash
   cat eval/results/latency_benchmark_*.json | jq .summary
   ```

3. **Compare with targets** (see table above)

4. **Identify bottlenecks** (high P95, high error rate)

5. **Optimize and re-test**
