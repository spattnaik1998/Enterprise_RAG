"""
FailureBiasedSampler -- decides whether an AgentTrace should be persisted.

Rules are loaded from config/trace_rules.yaml. Each rule has a `condition`
(a simple Python expression evaluated against trace attributes) and a `reason`
label. The first matching rule wins.

A 1% baseline random sample is always applied as the final fallback so we
capture some successful traces for replay regression testing.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from loguru import logger

_DEFAULT_RULES = [
    {"condition": "verdict == 'error'", "reason": "pipeline_error"},
    {"condition": "verdict == 'guardrail_block'", "reason": "security_event"},
    {"condition": "verdict == 'pii_redacted'", "reason": "pii_access"},
    {"condition": "'pii_redact' in event_types", "reason": "pii_access"},
    {"condition": "'guardrail_block' in event_types", "reason": "security_event"},
    {"condition": "total_cost_usd > 0.05", "reason": "high_cost"},
    {"condition": "verdict == 'escalated'", "reason": "council_escalation"},
]
_BASELINE_SAMPLE_RATE = 0.01

_rules_cache: list[dict] | None = None


def _load_rules() -> list[dict]:
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    rules_path = Path("config/trace_rules.yaml")
    if rules_path.exists():
        try:
            import yaml
            with open(rules_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            loaded = data.get("capture_rules", [])
            _rules_cache = loaded if loaded else _DEFAULT_RULES
            return _rules_cache
        except Exception as exc:
            logger.warning(f"[Sampler] Failed to load trace_rules.yaml: {exc} -- using defaults")
    _rules_cache = _DEFAULT_RULES
    return _rules_cache


class FailureBiasedSampler:
    """
    Evaluates capture rules against an AgentTrace and returns a reason string
    if the trace should be persisted, or None if it should be dropped.
    """

    def should_capture(self, trace) -> Optional[str]:
        """
        Args:
            trace: AgentTrace instance.
        Returns:
            A reason string if the trace should be captured, else None.
        """
        namespace = {
            "__builtins__": {},
            "verdict": trace.verdict,
            "event_types": trace.event_types,
            "total_cost_usd": trace.total_cost_usd,
            "user_role": trace.user_role,
            "model": trace.model,
        }

        for rule in _load_rules():
            condition = rule.get("condition", "False")
            reason = rule.get("reason", "rule_match")
            try:
                if eval(condition, {"__builtins__": {}}, namespace):  # noqa: S307
                    return reason
            except Exception as exc:
                logger.debug(f"[Sampler] Rule eval error ({condition!r}): {exc}")

        # 1% baseline sample
        if random.random() < _BASELINE_SAMPLE_RATE:
            return "baseline_sample"

        return None
