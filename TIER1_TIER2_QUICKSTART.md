# Tier 1 + 2 Quick Start

## What Was Implemented

Two-tier passive monitoring infrastructure that runs automatically with every query.

### Tier 1: QualityMonitor
Tracks pipeline quality metrics and detects degradation.

```python
from src.serving.pipeline import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.query("Which clients are overdue?")

# Automatic: logs metrics to data/monitoring/quality_metrics.jsonl

# Manual access:
metrics = pipeline.quality_monitor.get_current_metrics()
print(metrics)
# {
#   'window_size': 87,
#   'avg_recall': 0.82,           # Keyword recall
#   'avg_source_hit': 0.78,       # Source type hit
#   'avg_citation_count': 2.5,
#   'avg_latency_ms': 3821.0,
#   'p95_latency_ms': 5234.0,
#   'models_used': ['gpt-4o-mini']
# }

# Detect degradation
alerts = pipeline.quality_monitor.get_degradation_signal()
if alerts:
    print(f"⚠️  {len(alerts)} alerts detected")
    for alert in alerts:
        print(f"  - {alert.alert_type}: {alert.current_value:.2f} < {alert.threshold:.2f}")
```

### Tier 2: HardNegativeMiner
Intelligently collects failed queries for DSPy retraining.

```python
# Automatic: collects to data/learning/hard_negatives.jsonl when quality degrades

# Manual access:
stats = pipeline.hard_negative_miner.get_statistics()
print(stats)
# {
#   'total_examples': 23,
#   'by_category': {'billing': 12, 'contracts': 8, 'crm': 3},
#   'by_reason': {'missing_citations': 15, 'low_recall': 8},
#   'by_severity': {'high': 20, 'medium': 3},
#   'avg_recall': 0.12,  # Average of collected failures
#   'avg_source_hit': 0.08,
# }

# Load for future DSPy retraining
all_negatives = pipeline.hard_negative_miner.load_examples_for_training()
billing_negatives = pipeline.hard_negative_miner.load_examples_for_training(category="billing")
```

## Key Thresholds

QualityMonitor alerts if averages fall below:
- Keyword Recall: 0.75
- Source Type Hit: 0.80
- Latency (p95): 5000 ms

## Output Files

```
data/monitoring/
└── quality_metrics.jsonl         # Sliding window (100 results max)
    {"timestamp": "...", "recall_score": 0.85, ...}

data/learning/
├── hard_negatives.jsonl          # Collected failures (auto-rotates at 500)
│   {"failure_id": "hard_neg_000001", "reason": "missing_citations", ...}
└── hard_negatives_20260314_103045.jsonl  # Archived when rotated
```

## Cost Model

| Tier | Component | Cost | Frequency |
|---|---|---|---|
| 1 | QualityMonitor | $0.00 | Every query |
| 2 | HardNegativeMiner | $0.00 | On degradation |
| 3 | AutoRetrainer (future) | $0.05-0.35 | Weekly check |

## What Happens Automatically

```
1. User submits query
   ↓
2. RAG pipeline runs (retrieve → rerank → generate)
   ↓
3. QualityMonitor.log_result() — records metrics
   ↓
4. QualityMonitor.get_degradation_signal() — checks thresholds
   ↓
5. If degradation detected:
   └→ HardNegativeMiner.collect() — saves failed query
   └→ Logs warning in console
   ↓
6. Return result to user (all monitoring is non-blocking)
```

## Files Created

```
src/observability/quality_monitor.py      # QualityMonitor class
src/learning/hard_negative_miner.py       # HardNegativeMiner class
src/learning/__init__.py                  # Package exports
tests/test_quality_monitor.py             # 6 unit tests
tests/test_hard_negative_miner.py         # 12 unit tests
TIER1_TIER2_MONITORING.md                 # Full documentation
```

## Integration Points

**RAGPipeline.__init__():**
```python
self.quality_monitor = QualityMonitor()
self.hard_negative_miner = HardNegativeMiner()
```

**RAGPipeline.query():**
```python
# After generating result:
self.quality_monitor.log_result(...)
alerts = self.quality_monitor.get_degradation_signal()
if alerts:
    self.hard_negative_miner.collect(...)
```

## Safety

✅ Non-blocking (all exceptions caught)
✅ Zero latency impact (<1ms overhead)
✅ Backward compatible (no changes to existing API)
✅ Passive only (no deployment required)
✅ Can run indefinitely (auto-rotating logs)

## Testing

All 18 tests passing:
```bash
pytest tests/test_quality_monitor.py tests/test_hard_negative_miner.py -v
# 18 passed
```

## Next: Tier 3 (Future)

AutoRetrainer will:
1. Run weekly check
2. If failure_rate > 5% OR avg_recall < 0.70:
   - Load collected hard_negatives.jsonl
   - Run DSPy BootstrapFewShot retraining
   - Deploy compiled program
   - Archive hard_negatives.jsonl

## View Live Data

```bash
# Watch quality metrics
tail -f data/monitoring/quality_metrics.jsonl | jq '.'

# Count collected failures
wc -l data/learning/hard_negatives.jsonl

# Analyze by category
jq -s 'group_by(.category) | map({cat: .[0].category, count: length})' \
  data/learning/hard_negatives.jsonl
```

## Git Commit

```
5caf05c feat: Implement Tier 1 (QualityMonitor) and Tier 2 (HardNegativeMiner)
```

View full commit:
```bash
git show 5caf05c
```

## No Deployment Needed

This infrastructure is fully backward-compatible and ready to run in:
- ✅ CLI mode (`python -m src.main phase3`)
- ✅ Web server (`uvicorn app.server:app`)
- ✅ Eval suite (`python -m eval.run_eval`)
- ✅ DSPy trainer (`python -m dspy_module.trainer`)

All existing code continues to work unchanged.
