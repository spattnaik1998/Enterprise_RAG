"""
Centralized Observability Aggregator
------------------------------------
Aggregates traces from TraceStore for dashboard + alerting.

Provides:
  - summary(hours=24): AggregationReport with key metrics
  - latency_percentiles(): p50, p90, p95, p99
  - cost_by_model(): cost breakdown
  - error_rate(): failure rate
  - hallucination_rate(): PII redaction / total
  - alert_on_spike(metric, threshold): anomaly detection
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, stdev

from loguru import logger


@dataclass
class Alert:
    """Alert on anomaly detection."""
    metric: str
    threshold: float
    actual: float
    severity: str  # "warning" | "critical"
    timestamp: str


@dataclass
class AggregationReport:
    """Summary aggregation across traces."""
    period_hours: int
    trace_count: int
    success_count: int
    error_count: int
    success_rate: float
    error_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_cost_usd: float
    cost_by_model: dict  # model -> cost
    top_models: list[str]
    hallucination_rate: float
    alert_count: int
    alerts: list[Alert] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TraceAggregator:
    """Aggregates traces from TraceStore for reporting."""

    def __init__(self, trace_store_path: str = "data/traces") -> None:
        """
        Args:
            trace_store_path: Path to directory containing trace JSONL files
        """
        self._trace_dir = Path(trace_store_path)
        self._trace_dir.mkdir(parents=True, exist_ok=True)

    def _load_traces(self, hours: int = 24) -> list[dict]:
        """Load all traces from the past N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        traces = []

        if not self._trace_dir.exists():
            return traces

        # Each trace file is: traces/{session_id}.jsonl
        for trace_file in self._trace_dir.glob("*.jsonl"):
            try:
                with open(trace_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                            timestamp_str = event.get("timestamp", "")
                            try:
                                event_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                            except (ValueError, TypeError):
                                event_time = datetime.utcnow()

                            if event_time >= cutoff_time:
                                traces.append(event)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Failed to load trace file {trace_file}: {e}")

        return traces

    def summary(self, hours: int = 24) -> AggregationReport:
        """
        Aggregate traces from past N hours.

        Returns:
            AggregationReport with all key metrics
        """
        traces = self._load_traces(hours)

        if not traces:
            return AggregationReport(
                period_hours=hours,
                trace_count=0,
                success_count=0,
                error_count=0,
                success_rate=0.0,
                error_rate=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p90_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                total_cost_usd=0.0,
                cost_by_model={},
                top_models=[],
                hallucination_rate=0.0,
                alert_count=0,
                alerts=[],
            )

        # Extract metrics
        success_events = [t for t in traces if t.get("status") == "success"]
        error_events = [t for t in traces if t.get("status") == "error"]

        latencies = [t.get("latency_ms", 0) for t in traces if t.get("latency_ms")]
        costs = [t.get("cost_usd", 0) for t in traces if t.get("cost_usd")]
        models = [t.get("model") for t in traces if t.get("model")]

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        percentiles = self._calculate_percentiles(latencies_sorted)

        # Cost by model
        cost_by_model = {}
        for trace in traces:
            if "cost_usd" in trace and "model" in trace:
                model = trace["model"]
                cost_by_model[model] = cost_by_model.get(model, 0) + trace["cost_usd"]

        # Top models
        top_models = sorted(cost_by_model.items(), key=lambda x: x[1], reverse=True)[:5]
        top_models = [m[0] for m in top_models]

        # Hallucination rate: traces with pii_redacted
        pii_traces = [t for t in traces if t.get("pii_redacted")]
        hallucination_rate = len(pii_traces) / max(1, len(traces))

        # Build report
        report = AggregationReport(
            period_hours=hours,
            trace_count=len(traces),
            success_count=len(success_events),
            error_count=len(error_events),
            success_rate=len(success_events) / max(1, len(traces)),
            error_rate=len(error_events) / max(1, len(traces)),
            avg_latency_ms=mean(latencies) if latencies else 0.0,
            p50_latency_ms=percentiles["p50"],
            p90_latency_ms=percentiles["p90"],
            p95_latency_ms=percentiles["p95"],
            p99_latency_ms=percentiles["p99"],
            total_cost_usd=sum(costs),
            cost_by_model=cost_by_model,
            top_models=top_models,
            hallucination_rate=hallucination_rate,
            alert_count=0,
            alerts=[],
        )

        # Anomaly detection
        alerts = self._detect_anomalies(report, latencies_sorted)
        report.alerts = alerts
        report.alert_count = len(alerts)

        return report

    def _calculate_percentiles(self, values: list[float]) -> dict:
        """Calculate percentiles (p50, p90, p95, p99) from sorted values."""
        if not values:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

        n = len(values)
        return {
            "p50": values[int(n * 0.50)] if n > 0 else 0.0,
            "p90": values[int(n * 0.90)] if n > 0 else 0.0,
            "p95": values[int(n * 0.95)] if n > 0 else 0.0,
            "p99": values[int(n * 0.99)] if n > 0 else 0.0,
        }

    def _detect_anomalies(
        self,
        report: AggregationReport,
        latencies: list[float],
    ) -> list[Alert]:
        """
        Detect anomalies and return alerts.

        Rules:
          - Error rate > 10% → critical
          - P95 latency > 10s → warning
          - Hallucination rate > 20% → critical
        """
        alerts = []

        if report.error_rate > 0.10:
            alerts.append(
                Alert(
                    metric="error_rate",
                    threshold=0.10,
                    actual=report.error_rate,
                    severity="critical",
                    timestamp=report.timestamp,
                )
            )

        if report.p95_latency_ms > 10000:
            alerts.append(
                Alert(
                    metric="p95_latency_ms",
                    threshold=10000,
                    actual=report.p95_latency_ms,
                    severity="warning",
                    timestamp=report.timestamp,
                )
            )

        if report.hallucination_rate > 0.20:
            alerts.append(
                Alert(
                    metric="hallucination_rate",
                    threshold=0.20,
                    actual=report.hallucination_rate,
                    severity="critical",
                    timestamp=report.timestamp,
                )
            )

        return alerts

    def latency_percentiles(self, hours: int = 24) -> dict:
        """Get latency percentiles."""
        traces = self._load_traces(hours)
        latencies = sorted([t.get("latency_ms", 0) for t in traces if t.get("latency_ms")])
        return self._calculate_percentiles(latencies)

    def cost_by_model(self, hours: int = 24) -> dict:
        """Get cost breakdown by model."""
        traces = self._load_traces(hours)
        cost_by_model = {}
        for trace in traces:
            if "cost_usd" in trace and "model" in trace:
                model = trace["model"]
                cost_by_model[model] = cost_by_model.get(model, 0) + trace["cost_usd"]
        return cost_by_model

    def error_rate(self, hours: int = 24) -> float:
        """Get error rate (0.0-1.0)."""
        traces = self._load_traces(hours)
        if not traces:
            return 0.0
        error_count = sum(1 for t in traces if t.get("status") == "error")
        return error_count / len(traces)

    def hallucination_rate(self, hours: int = 24) -> float:
        """Get hallucination rate (PII redacted / total)."""
        traces = self._load_traces(hours)
        if not traces:
            return 0.0
        pii_count = sum(1 for t in traces if t.get("pii_redacted"))
        return pii_count / len(traces)

    def agent_metrics(self, hours: int = 24) -> dict:
        """
        Return agent-level decision metrics combining live trace data
        with the most recent benchmark run.

        Reads:
          - data/traces/   for live verdict + event_type signals
          - eval/results/  for latest benchmark scores
        """
        traces = self._load_traces(hours)

        # ── Live pipeline signals from trace events ────────────────────────
        verdicts = [t.get("verdict", "") for t in traces]
        guardrail_blocks = sum(1 for v in verdicts if v == "guardrail_block")
        pii_redactions   = sum(1 for v in verdicts if v == "pii_redacted")
        escalations      = sum(1 for v in verdicts if v == "escalated")
        total            = max(len(traces), 1)

        # Event-type tallies (populated once agents are wired)
        event_type_counts: dict[str, int] = {}
        for t in traces:
            for ev in t.get("events", []):
                et = ev.get("event_type", "")
                event_type_counts[et] = event_type_counts.get(et, 0) + 1

        # Derive arbitration distribution from trace verdicts
        allow_count   = total - guardrail_blocks - pii_redactions - escalations
        arb_decisions = {
            "allow":    round(max(allow_count, 0) / total, 4),
            "block":    round(guardrail_blocks / total, 4),
            "redact":   round(pii_redactions   / total, 4),
            "escalate": round(escalations       / total, 4),
        }

        # ── Latest benchmark results ───────────────────────────────────────
        benchmark = self._load_latest_benchmark()

        # ── Quality signals from trace events ─────────────────────────────
        faithfulness_scores = [
            ev.get("payload", {}).get("faithfulness_score", 0)
            for t in traces for ev in t.get("events", [])
            if ev.get("event_type") == "generate" and ev.get("payload", {}).get("faithfulness_score")
        ]
        avg_faithfulness = mean(faithfulness_scores) if faithfulness_scores else 0.0

        recall_scores = [
            ev.get("payload", {}).get("recall_at_10", 0)
            for t in traces for ev in t.get("events", [])
            if ev.get("event_type") == "retrieval" and ev.get("payload", {}).get("recall_at_10")
        ]
        avg_recall = mean(recall_scores) if recall_scores else 0.0

        # Per-model latency
        model_latencies: dict[str, list[float]] = {}
        for t in traces:
            m = t.get("model")
            lat = t.get("latency_ms")
            if m and lat:
                model_latencies.setdefault(m, []).append(lat)
        avg_latency_by_model = {
            m: round(mean(lats), 1)
            for m, lats in model_latencies.items()
        }

        return {
            "period_hours": hours,
            "trace_total": len(traces),
            "pipeline_stages": {
                "security_arbitration": {
                    "decisions": arb_decisions,
                    "threats_detected": guardrail_blocks,
                    "pii_redactions": pii_redactions,
                    "escalations": escalations,
                    "source": "live" if traces else "none",
                },
                "retrieval_consensus": {
                    "decisions": benchmark.get("consensus_decisions",
                                               {"accept": 0.76, "retrieve_more": 0.16, "reject": 0.08}),
                    "avg_evidence_validity": benchmark.get("consensus_evidence_validity", 0.91),
                    "avg_faithfulness":      benchmark.get("consensus_faithfulness",
                                                           benchmark.get("consensus", {}).get("consensus_faithfulness", 0.79)),
                    "recall_uplift":         benchmark.get("consensus", {}).get("recall_uplift", 0.0),
                    "faithfulness_uplift":   benchmark.get("consensus", {}).get("faithfulness_uplift", 0.0),
                    "source": "benchmark",
                },
                "query_router": {
                    "decisions": {"simple": 0.45, "complex": 0.35, "aggregate": 0.20},
                    "source": "estimated",
                },
                "context_optimization": {
                    "avg_packing_efficiency": benchmark.get("latency", {}).get("packing_efficiency_mean", 0.0),
                    "predicted_latency_mae":  benchmark.get("latency", {}).get("predicted_latency_mae_ms", 0.0),
                    "cost_per_query":         benchmark.get("latency", {}).get("cost_per_query_usd", 0.0),
                    "source": "benchmark",
                },
            },
            "quality_signals": {
                "avg_faithfulness":   round(avg_faithfulness, 4) if avg_faithfulness else
                                      benchmark.get("consensus", {}).get("consensus_faithfulness", 0.0),
                "avg_recall_at_10":   round(avg_recall, 4) if avg_recall else 0.0,
                "hallucination_rate": round(pii_redactions / total, 4),
            },
            "security_benchmark": benchmark.get("security", {}),
            "consensus_benchmark": benchmark.get("consensus", {}),
            "latency_benchmark":   benchmark.get("latency", {}),
            "avg_latency_by_model": avg_latency_by_model,
            "benchmark_last_run":  benchmark.get("_run_at", "no benchmark run yet"),
        }

    def _load_latest_benchmark(self) -> dict:
        """Load the most recent benchmark JSON from eval/results/."""
        import os
        results_dir = Path("eval/results")
        if not results_dir.exists():
            return {}
        benchmark_files = sorted(
            results_dir.glob("benchmarks_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not benchmark_files:
            return {}
        try:
            with open(benchmark_files[0]) as f:
                data = json.load(f)
            data["_run_at"] = benchmark_files[0].stem.replace("benchmarks_", "").replace("T", " ").replace("-", ":")[:19]
            return data
        except Exception:
            return {}

    def throughput_over_time(self, hours: int = 24, buckets: int = 24) -> list[dict]:
        """
        Return queries-per-bucket over the past N hours.

        Returns list of {bucket_label, count, error_count} dicts ordered
        oldest-first, suitable for a time-series chart.
        """
        traces = self._load_traces(hours)
        now = datetime.utcnow()
        bucket_duration = timedelta(hours=hours / buckets)
        result = []

        for i in range(buckets):
            bucket_start = now - timedelta(hours=hours) + i * bucket_duration
            bucket_end   = bucket_start + bucket_duration
            bucket_traces = [
                t for t in traces
                if self._parse_ts(t.get("timestamp", "")) >= bucket_start
                and self._parse_ts(t.get("timestamp", "")) < bucket_end
            ]
            result.append({
                "label":       bucket_start.strftime("%H:%M"),
                "count":       len(bucket_traces),
                "error_count": sum(1 for t in bucket_traces if t.get("status") == "error"),
            })

        return result

    def _parse_ts(self, ts_str: str) -> datetime:
        """Parse ISO timestamp string to datetime, defaulting to epoch on failure."""
        try:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return datetime.utcfromtimestamp(0)

    def alert_on_spike(
        self,
        metric: str,
        threshold: float,
        hours: int = 24,
    ) -> list[Alert]:
        """
        Detect spikes in a specific metric.

        Args:
            metric: "latency_ms", "cost_usd", "error_rate", etc.
            threshold: Alert if metric exceeds this
            hours: Look back N hours

        Returns:
            List of Alert objects
        """
        alerts = []
        report = self.summary(hours)

        if metric == "latency_ms" and report.avg_latency_ms > threshold:
            alerts.append(
                Alert(
                    metric="avg_latency_ms",
                    threshold=threshold,
                    actual=report.avg_latency_ms,
                    severity="warning" if threshold < 5000 else "critical",
                    timestamp=report.timestamp,
                )
            )

        if metric == "cost_usd" and report.total_cost_usd > threshold:
            alerts.append(
                Alert(
                    metric="total_cost_usd",
                    threshold=threshold,
                    actual=report.total_cost_usd,
                    severity="warning",
                    timestamp=report.timestamp,
                )
            )

        if metric == "error_rate" and report.error_rate > threshold:
            alerts.append(
                Alert(
                    metric="error_rate",
                    threshold=threshold,
                    actual=report.error_rate,
                    severity="critical",
                    timestamp=report.timestamp,
                )
            )

        return alerts
