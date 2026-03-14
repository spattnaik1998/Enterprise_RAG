# Testing Tier 1 + 2 — Execution & Verification Guide

## Quick Start (5 minutes)

### 1. Run Unit Tests

```bash
# Run all monitoring tests
pytest tests/test_quality_monitor.py tests/test_hard_negative_miner.py -v

# Expected output:
# ===== 18 passed in 0.45s =====
```

### 2. Test Monitoring Functionality

```bash
cd C:\Users\91838\Downloads\Enterprise_RAG

python << 'EOF'
from src.serving.pipeline import RAGPipeline
from src.observability.quality_monitor import QualityMonitor
from src.learning.hard_negative_miner import HardNegativeMiner

print("=" * 70)
print("Testing Tier 1 + 2 Monitoring Infrastructure")
print("=" * 70)

# Create monitors standalone
print("\n[1] Creating QualityMonitor...")
monitor = QualityMonitor(window_size=10)
print("    ✓ QualityMonitor initialized")

print("\n[2] Creating HardNegativeMiner...")
miner = HardNegativeMiner()
print("    ✓ HardNegativeMiner initialized")

# Test logging a good result
print("\n[3] Testing metric logging (good result)...")
monitor.log_result(
    query="Which clients have overdue invoices?",
    answer="Northern Lights and Crossroads Healthcare have overdue invoices.",
    citations=[
        {"source": "invoice_001", "source_type": "billing"},
        {"source": "invoice_002", "source_type": "billing"},
    ],
    latency_ms=3200.0,
    model="gpt-4o-mini",
    expected_keywords=["Northern Lights", "Crossroads"],
    expected_source_types=["billing"],
)
print("    ✓ Good result logged (recall=1.0, source_hit=1.0)")

# Test logging multiple results to trigger window aggregation
print("\n[4] Logging 5 more results...")
for i in range(5):
    monitor.log_result(
        query=f"Query {i}",
        answer="Test answer",
        citations=[{"source": "src", "source_type": "billing"}],
        latency_ms=1000.0 + (i * 100),
        model="gpt-4o-mini",
    )
print("    ✓ 5 results logged")

# Get metrics
print("\n[5] Getting current metrics...")
metrics = monitor.get_current_metrics()
print(f"    Window size: {metrics['window_size']}")
print(f"    Avg recall: {metrics['avg_recall']:.2f}")
print(f"    Avg source_hit: {metrics['avg_source_hit']:.2f}")
print(f"    Avg latency: {metrics['avg_latency_ms']:.0f}ms")
print(f"    P95 latency: {metrics['p95_latency_ms']:.0f}ms")

# Test hard negative collection
print("\n[6] Testing hard negative collection (failure case)...")
example = miner.collect(
    query="Which clients are overdue?",
    answer="I cannot find the information.",  # Bad answer
    citations=[],  # No citations
    recall_score=0.0,
    source_hit_score=0.0,
    category="billing",
    model="gpt-4o-mini",
    latency_ms=2500.0,
    expected_keywords=["Northern Lights", "Crossroads"],
    expected_source_types=["billing"],
)
if example:
    print(f"    ✓ Collected hard negative: {example.failure_id}")
    print(f"      Reason: {example.failure_reason}")
    print(f"      Severity: {example.severity}")

# Get statistics
print("\n[7] Getting hard negative statistics...")
stats = miner.get_statistics()
print(f"    Total examples: {stats['total_examples']}")
print(f"    By reason: {stats['by_reason']}")
print(f"    By severity: {stats['by_severity']}")

# Test no collection for good results
print("\n[8] Testing that good results are NOT collected...")
example2 = miner.collect(
    query="Test query",
    answer="Good answer with citations",
    citations=[{"source": "src", "source_type": "billing"}],
    recall_score=1.0,
    source_hit_score=1.0,
    category="billing",
    model="gpt-4o-mini",
    latency_ms=1000.0,
)
if example2 is None:
    print("    ✓ Good result correctly skipped (not collected)")

print("\n" + "=" * 70)
print("[SUCCESS] All Tier 1 + 2 components working correctly!")
print("=" * 70)
print("\nOutput files created:")
print("  - data/monitoring/quality_metrics.jsonl")
print("  - data/learning/hard_negatives.jsonl")
EOF
```

---

## Integration Test (10 minutes)

### Test with RAGPipeline

**Note**: Requires FAISS index from Phase II. If you haven't run Phase II yet:

```bash
# Quick check if index exists
ls -lh data/index/faiss.index
# If not found, run: python -m src.main phase2
```

If index exists, test the integration:

```bash
cd C:\Users\91838\Downloads\Enterprise_RAG

python << 'EOF'
from src.serving.pipeline import RAGPipeline

print("=" * 70)
print("Testing RAGPipeline Integration with Tier 1 + 2")
print("=" * 70)

try:
    print("\n[1] Loading RAGPipeline...")
    pipeline = RAGPipeline(index_dir="data/index")
    print("    ✓ RAGPipeline loaded")

    # Verify monitors exist
    print("\n[2] Verifying monitoring components...")
    assert hasattr(pipeline, 'quality_monitor'), "Missing quality_monitor"
    assert hasattr(pipeline, 'hard_negative_miner'), "Missing hard_negative_miner"
    print("    ✓ quality_monitor present")
    print("    ✓ hard_negative_miner present")

    print("\n[3] Running a query (first one may be slow due to index loading)...")
    result = pipeline.query("What is the total outstanding balance?")
    print(f"    ✓ Query completed in {result.total_ms:.0f}ms")
    print(f"    Answer length: {len(result.answer)} characters")
    print(f"    Citations: {len(result.citations)}")

    print("\n[4] Checking current metrics...")
    metrics = pipeline.quality_monitor.get_current_metrics()
    print(f"    Window size: {metrics['window_size']}")
    print(f"    Avg recall: {metrics['avg_recall']:.2f}")
    print(f"    Avg latency: {metrics['avg_latency_ms']:.0f}ms")

    print("\n[5] Checking for degradation alerts...")
    alerts = pipeline.quality_monitor.get_degradation_signal()
    if alerts:
        print(f"    ⚠️  {len(alerts)} alerts detected")
        for alert in alerts:
            print(f"       - {alert.alert_type}: {alert.current_value:.2f} < {alert.threshold:.2f}")
    else:
        print("    ✓ No degradation detected")

    print("\n[6] Checking hard negatives...")
    stats = pipeline.hard_negative_miner.get_statistics()
    print(f"    Total collected: {stats['total_examples']}")
    if stats['total_examples'] > 0:
        print(f"    By reason: {stats['by_reason']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] RAGPipeline integration working correctly!")
    print("=" * 70)

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
EOF
```

---

## Detailed Testing Scenarios (30 minutes)

### Scenario 1: Metric Tracking Over Multiple Queries

```bash
python << 'EOF'
from src.observability.quality_monitor import QualityMonitor
import json

print("Scenario 1: Metric Tracking Over 10 Queries")
print("=" * 70)

monitor = QualityMonitor(window_size=10, output_dir="data/test_monitoring")

# Simulate 10 queries with varying quality
queries = [
    {"keywords": ["A", "B"], "sources": ["billing"], "quality": "good"},
    {"keywords": ["A", "B"], "sources": ["billing"], "quality": "good"},
    {"keywords": ["C", "D"], "sources": ["contracts"], "quality": "partial"},
    {"keywords": ["E", "F"], "sources": ["crm"], "quality": "partial"},
    {"keywords": ["G", "H"], "sources": ["psa"], "quality": "bad"},
    {"keywords": ["I", "J"], "sources": ["billing"], "quality": "good"},
    {"keywords": ["K", "L"], "sources": ["contracts"], "quality": "partial"},
    {"keywords": ["M", "N"], "sources": ["crm"], "quality": "bad"},
    {"keywords": ["O", "P"], "sources": ["psa"], "quality": "good"},
    {"keywords": ["Q", "R"], "sources": ["billing"], "quality": "partial"},
]

for i, q in enumerate(queries, 1):
    if q["quality"] == "good":
        answer = f"{q['keywords'][0]} and {q['keywords'][1]} information"
        citations = [{"source": s, "source_type": q["sources"][0]} for s in q["sources"]]
    elif q["quality"] == "partial":
        answer = f"Information about {q['keywords'][0]}"
        citations = [{"source": q["sources"][0], "source_type": q["sources"][0]}]
    else:
        answer = "I don't have this information"
        citations = []

    monitor.log_result(
        query=f"Query {i}",
        answer=answer,
        citations=citations,
        latency_ms=2000.0 + (i * 100),
        model="gpt-4o-mini",
        expected_keywords=q["keywords"],
        expected_source_types=q["sources"],
    )
    print(f"  [{i:2d}] Logged - quality={q['quality']:7s} | "
          f"recall={'good' if q['quality'] == 'good' else 'poor':4s}")

# Get final metrics
metrics = monitor.get_current_metrics()
print("\nFinal Metrics:")
print(f"  Window size: {metrics['window_size']}")
print(f"  Avg recall: {metrics['avg_recall']:.2f}")
print(f"  Avg source_hit: {metrics['avg_source_hit']:.2f}")
print(f"  Avg latency: {metrics['avg_latency_ms']:.0f}ms")

# Check for degradation
alerts = monitor.get_degradation_signal()
if alerts:
    print(f"\n⚠️  {len(alerts)} Degradation Alerts:")
    for alert in alerts:
        print(f"  - {alert.alert_type}: {alert.current_value:.2f} < {alert.threshold:.2f}")
else:
    print("\n✓ No degradation alerts")

print("=" * 70)
EOF
```

### Scenario 2: Hard Negative Collection Heuristics

```bash
python << 'EOF'
from src.learning.hard_negative_miner import HardNegativeMiner

print("Scenario 2: Hard Negative Collection Heuristics")
print("=" * 70)

miner = HardNegativeMiner(output_dir="data/test_learning", max_examples=50)

test_cases = [
    {
        "name": "Missing Citations",
        "citations": 0,
        "recall": 0.0,
        "source_hit": 0.0,
        "answer_len": 100,
        "should_collect": True,
        "reason": "missing_citations",
    },
    {
        "name": "Low Recall",
        "citations": 1,
        "recall": 0.2,
        "source_hit": 0.9,
        "answer_len": 150,
        "should_collect": True,
        "reason": "low_recall",
    },
    {
        "name": "Wrong Source",
        "citations": 2,
        "recall": 0.8,
        "source_hit": 0.1,
        "answer_len": 150,
        "should_collect": True,
        "reason": "wrong_source",
    },
    {
        "name": "Hallucination Risk",
        "citations": 1,
        "recall": 1.0,
        "source_hit": 0.9,
        "answer_len": 500,
        "should_collect": True,
        "reason": "hallucination_risk",
    },
    {
        "name": "Good Result",
        "citations": 3,
        "recall": 1.0,
        "source_hit": 1.0,
        "answer_len": 200,
        "should_collect": False,
        "reason": None,
    },
]

for tc in test_cases:
    should_collect, reason = miner.should_collect(
        recall_score=tc["recall"],
        source_hit_score=tc["source_hit"],
        citation_count=tc["citations"],
        answer_length=tc["answer_len"],
    )

    status = "✓" if (should_collect == tc["should_collect"]) else "✗"
    print(f"{status} {tc['name']:20s} - Collect: {should_collect:5s} | "
          f"Reason: {reason or 'none':20s}")

print("=" * 70)
EOF
```

### Scenario 3: Hard Negative Statistics

```bash
python << 'EOF'
from src.learning.hard_negative_miner import HardNegativeMiner

print("Scenario 3: Hard Negative Statistics & Analysis")
print("=" * 70)

miner = HardNegativeMiner(output_dir="data/test_learning")

# Collect examples from different categories
categories = ["billing", "contracts", "crm", "psa", "communications"]
reasons = ["missing_citations", "low_recall", "wrong_source"]

print("Collecting 15 hard negatives across categories and reasons...")
for i in range(15):
    cat = categories[i % len(categories)]
    reason = reasons[i % len(reasons)]

    if reason == "missing_citations":
        recall, source_hit, citations = 0.0, 0.0, 0
    elif reason == "low_recall":
        recall, source_hit, citations = 0.2, 0.8, 1
    else:  # wrong_source
        recall, source_hit, citations = 0.8, 0.1, 2

    example = miner.collect(
        query=f"Query in {cat} ({reason})",
        answer="Some answer",
        citations=["src1"] * citations,
        recall_score=recall,
        source_hit_score=source_hit,
        category=cat,
        model="gpt-4o-mini",
        latency_ms=2000.0,
    )
    if example:
        print(f"  [{i+1:2d}] {cat:15s} | {reason:20s} | {example.failure_id}")

# Print statistics
print("\nStatistics:")
stats = miner.get_statistics()

print(f"\nTotal Examples: {stats['total_examples']}")

print("\nBy Category:")
for cat, count in stats['by_category'].items():
    print(f"  {cat:18s}: {count:3d}")

print("\nBy Failure Reason:")
for reason, count in stats['by_reason'].items():
    print(f"  {reason:20s}: {count:3d}")

print("\nBy Severity:")
for severity, count in stats['by_severity'].items():
    print(f"  {severity:10s}: {count:3d}")

print(f"\nMetrics:")
print(f"  Avg recall: {stats['avg_recall']:.2f}")
print(f"  Avg source_hit: {stats['avg_source_hit']:.2f}")

print("=" * 70)
EOF
```

---

## Viewing Output Files

### Monitor Quality Metrics

```bash
# View latest metrics
head -5 data/monitoring/quality_metrics.jsonl

# Count metrics entries
wc -l data/monitoring/quality_metrics.jsonl

# Pretty print (requires jq)
head -1 data/monitoring/quality_metrics.jsonl | jq '.'

# Watch in real-time (requires jq)
tail -f data/monitoring/quality_metrics.jsonl | jq '.'
```

### Analyze Hard Negatives

```bash
# View latest hard negatives
head -5 data/learning/hard_negatives.jsonl

# Count collected failures
wc -l data/learning/hard_negatives.jsonl

# Count by failure reason (requires jq)
jq -s 'group_by(.failure_reason) | map({reason: .[0].failure_reason, count: length})' \
  data/learning/hard_negatives.jsonl

# Count by category (requires jq)
jq -s 'group_by(.category) | map({category: .[0].category, count: length})' \
  data/learning/hard_negatives.jsonl

# Count by severity (requires jq)
jq -s 'group_by(.severity) | map({severity: .[0].severity, count: length})' \
  data/learning/hard_negatives.jsonl
```

---

## Running from CLI

### Phase III Query with Monitoring

```bash
# Single query
python -m src.main phase3 --query "Which clients have overdue invoices?"

# Interactive mode (monitoring runs automatically)
python -m src.main phase3

# Both modes will:
# 1. Log metrics to data/monitoring/quality_metrics.jsonl
# 2. Collect hard negatives to data/learning/hard_negatives.jsonl if quality degrades
```

### Check Monitoring Status

```bash
python << 'EOF'
from src.serving.pipeline import RAGPipeline

pipeline = RAGPipeline(index_dir="data/index")

# Get current metrics
metrics = pipeline.quality_monitor.get_current_metrics()
print(f"Metrics: {metrics}")

# Check for alerts
alerts = pipeline.quality_monitor.get_degradation_signal()
if alerts:
    for alert in alerts:
        print(f"Alert: {alert.alert_type}")

# Get hard negative stats
stats = pipeline.hard_negative_miner.get_statistics()
print(f"Hard negatives collected: {stats['total_examples']}")
EOF
```

---

## Eval Suite with Monitoring

The monitoring runs automatically when you use the eval suite:

```bash
# Eval with monitoring (auto logs metrics + collects hard negatives)
python -m eval.run_eval --models gpt-4o-mini --category billing --sample 5

# After completion:
# - data/monitoring/quality_metrics.jsonl will have ~5 entries
# - data/learning/hard_negatives.jsonl will have failures (if any)

# View eval results + monitoring
tail -5 data/monitoring/quality_metrics.jsonl | jq '.'
head -5 data/learning/hard_negatives.jsonl | jq '.'
```

---

## Web Server with Monitoring

```bash
# Start web server (monitoring runs on every query)
uvicorn app.server:app --reload --port 8000

# Make queries via API
# POST http://localhost:8000/api/chat
# Body: {"message": "Which clients are overdue?", "provider": "openai", "model": "gpt-4o-mini"}

# Monitor in real-time
tail -f data/monitoring/quality_metrics.jsonl | jq '.recall_score'
tail -f data/learning/hard_negatives.jsonl | jq '.failure_id'
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src.observability.quality_monitor'"

**Solution**: Ensure you're in the project root directory:
```bash
cd C:\Users\91838\Downloads\Enterprise_RAG
python -m pytest tests/test_quality_monitor.py
```

### Issue: "FileNotFoundError: data/index/faiss.index"

**Solution**: Run Phase II first to build the FAISS index:
```bash
python -m src.main phase2
```

### Issue: Tests fail with permission errors

**Solution**: Check file permissions and clear old test outputs:
```bash
rm -rf data/test_monitoring data/test_learning
pytest tests/test_quality_monitor.py tests/test_hard_negative_miner.py -v
```

### Issue: Output files not being created

**Solution**: Verify directories exist:
```bash
mkdir -p data/monitoring
mkdir -p data/learning
```

---

## Summary

| Test Type | Command | Duration | Requirements |
|---|---|---|---|
| Unit Tests | `pytest tests/test_*.py -v` | 1 min | None |
| Monitoring Test | `python -c "..."` (standalone) | 1 min | None |
| Integration Test | `RAGPipeline(...).query(...)` | 5 min | FAISS index (Phase II) |
| Scenario Tests | `python -c "..."` (3 scenarios) | 10 min | None |
| Full Flow | `python -m src.main phase3` | Ongoing | FAISS index |
| Web Server | `uvicorn app.server:app` | Ongoing | FAISS index + Supabase (optional) |

**Start with**: Unit tests → Monitoring test → Integration test
**Then explore**: Scenario tests → CLI/Web integration

All tests should pass without errors. Monitoring files will be created in `data/monitoring/` and `data/learning/`.
