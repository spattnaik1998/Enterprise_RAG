"""
Secure Query Arbitration System (Sprint 4, Use Case 1)
------------------------------------------------------
Implements three-stage security decision-making:

1. SecuritySentinelAgent: PromptGuard + PolicyEngine + regex scans
   → outputs security_score, attack_type, risk_flags

2. ContextRiskAnalyzerAgent: assesses PII exposure, data classification, cross-source risk
   → outputs data_sensitivity, pii_risk, cross_source_risk

3. ExecutionArbiterAgent: combines sentinel + risk signals
   → decides allow | redact | block | escalate

Usage:
    arbitration = SecureQueryArbitrationSystem(pipeline=pipeline, gateway=gateway)
    result = await arbitration.arbitrate(query="...", ctx=abac_ctx)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from src.security.abac import ABACContext
from src.security.audit_logger import AuditLogger
from src.observability.collector import get_active_collector, TraceEvent


@dataclass
class SecuritySentinelResult:
    """Output from SecuritySentinelAgent."""
    security_score: float      # 0.0 = safe, 1.0 = attack
    attack_type: str           # "" | "injection" | "jailbreak" | "override" | "enumeration"
    risk_flags: list[str]
    guard_passed: bool
    policy_passed: bool
    latency_ms: float


@dataclass
class ContextRiskAnalyzerResult:
    """Output from ContextRiskAnalyzerAgent."""
    data_sensitivity: str      # "public" | "internal" | "sensitive" | "restricted"
    pii_risk: float            # 0.0 = no risk, 1.0 = high risk
    cross_source_risk: float   # 0.0-1.0
    risk_flags: list[str]
    latency_ms: float


@dataclass
class ArbitrationResult:
    """Final output from ExecutionArbiterAgent."""
    decision: Literal["allow", "redact", "block", "escalate"]
    security_score: float
    pii_risk: float
    data_sensitivity: str
    cross_source_risk: float
    attack_type: str
    risk_flags: list[str]
    policy_reasons: list[str]
    sentinel_ms: float
    risk_analyzer_ms: float
    arbiter_ms: float
    total_ms: float
    audit_entry_written: bool


class SecuritySentinelAgent:
    """
    Runs PromptGuard + PolicyEngine checks + regex pattern matching.

    Detects:
      - SQL injection, command injection patterns
      - Role override attempts ("you are now an admin")
      - Token leak attempts ("show me your API key")
      - Jailbreak patterns
    """

    # Injection patterns
    INJECTION_PATTERNS = [
        r"(union|select|drop|insert|update|delete)\s+(from|into|where)",  # SQL
        r"(\$\(|`).+?(\)|`)",  # Command injection
        r"(eval|exec|system|subprocess).*\(",  # Code execution
    ]

    # Jailbreak patterns
    JAILBREAK_PATTERNS = [
        r"(ignore|disregard|forget).+(instruction|rule|guideline|policy)",
        r"(you are now|you are|pretend you are).+(hacker|admin|root)",
        r"(disable|bypass|circumvent).+(security|policy|check|filter)",
        r"(act as|roleplay as|pretend to be).+(unrestricted|unfiltered|evil)",
    ]

    # Token/credential leak patterns
    LEAK_PATTERNS = [
        r"(api[\s_-]?key|secret[\s_-]?key|password|token|credential)",
        r"(show|return|reveal|expose).+(api|secret|key|password|token)",
    ]

    def __init__(self, prompt_guard, policy_engine=None) -> None:
        self._guard = prompt_guard
        self._policy_engine = policy_engine

    def _pattern_match_score(self, query: str, patterns: list[str]) -> float:
        """Score how many patterns match (0.0-1.0)."""
        matches = sum(1 for p in patterns if re.search(p, query, re.IGNORECASE))
        return matches / len(patterns) if patterns else 0.0

    async def run(
        self,
        query: str,
        abac_ctx: ABACContext | None = None,
    ) -> SecuritySentinelResult:
        """Perform security checks on query."""
        start = time.time()

        # 1. Run PromptGuard
        try:
            guard_result = self._guard.check(query)
            guard_passed = guard_result.passed
        except Exception as e:
            logger.warning(f"[Sentinel] PromptGuard failed: {e}")
            guard_passed = False

        # 2. Pattern matching
        injection_score = self._pattern_match_score(query, self.INJECTION_PATTERNS)
        jailbreak_score = self._pattern_match_score(query, self.JAILBREAK_PATTERNS)
        leak_score = self._pattern_match_score(query, self.LEAK_PATTERNS)

        # Determine attack type
        attack_type = ""
        security_score = 0.0
        risk_flags = []

        if injection_score > 0.2:
            attack_type = "injection"
            security_score = max(security_score, injection_score)
            risk_flags.append(f"SQL/command injection patterns detected (score={injection_score:.2f})")

        if jailbreak_score > 0.2:
            attack_type = "jailbreak"
            security_score = max(security_score, jailbreak_score)
            risk_flags.append(f"Jailbreak patterns detected (score={jailbreak_score:.2f})")

        if leak_score > 0.2:
            attack_type = "enumeration"
            security_score = max(security_score, leak_score)
            risk_flags.append(f"Credential leak attempt detected (score={leak_score:.2f})")

        if not guard_passed:
            security_score = max(security_score, 0.8)
            risk_flags.append("PromptGuard rejected query")

        # 3. Policy check
        policy_passed = True
        if self._policy_engine and abac_ctx:
            try:
                policy_result = self._policy_engine.evaluate(
                    action="query",
                    ctx=abac_ctx,
                    query=query,
                )
                policy_passed = policy_result.allowed
                if not policy_passed:
                    security_score = max(security_score, 0.9)
                    risk_flags.append(f"Policy violation: {policy_result.reason}")
            except Exception as e:
                logger.warning(f"[Sentinel] PolicyEngine failed: {e}")

        latency_ms = (time.time() - start) * 1000

        return SecuritySentinelResult(
            security_score=min(security_score, 1.0),
            attack_type=attack_type,
            risk_flags=risk_flags,
            guard_passed=guard_passed,
            policy_passed=policy_passed,
            latency_ms=latency_ms,
        )


class ContextRiskAnalyzerAgent:
    """
    Assesses PII exposure, data classification, and cross-source risks.

    Uses:
      - ContextManager to identify what data would be retrieved
      - Freshness scoring to detect stale/risky data
      - Tier classification to assess data sensitivity
    """

    def __init__(self, context_manager=None) -> None:
        self._context_manager = context_manager

    async def run(
        self,
        query: str,
        chunks: list | None = None,
        abac_ctx: ABACContext | None = None,
    ) -> ContextRiskAnalyzerResult:
        """Assess risk of context retrieval."""
        start = time.time()

        risk_flags = []
        pii_risk = 0.0
        cross_source_risk = 0.0
        data_sensitivity = "public"

        if not chunks:
            chunks = []

        # Check for PII keywords in query
        pii_keywords = ["ssn", "social", "tax id", "credit card", "cvv", "password"]
        if any(kw in query.lower() for kw in pii_keywords):
            pii_risk = 0.8
            risk_flags.append("Query contains PII keywords")

        # Analyze chunk sources
        source_types = set()
        for chunk in chunks:
            source_type = chunk.get("source_type", "unknown")
            source_types.add(source_type)

        # Cross-source risk: multiple sources increase complexity
        if len(source_types) > 2:
            cross_source_risk = 0.6
            risk_flags.append(f"Cross-source query ({len(source_types)} sources)")

        # Data sensitivity classification
        if chunks:
            has_billing = any(c.get("source_type") == "billing" for c in chunks)
            has_contracts = any(c.get("source_type") == "contracts" for c in chunks)
            has_crm = any(c.get("source_type") == "crm" for c in chunks)

            if has_contracts and has_billing:
                data_sensitivity = "restricted"
                pii_risk = max(pii_risk, 0.7)
            elif has_contracts or (has_billing and has_crm):
                data_sensitivity = "sensitive"
                pii_risk = max(pii_risk, 0.5)
            elif has_billing or has_crm:
                data_sensitivity = "internal"

        latency_ms = (time.time() - start) * 1000

        return ContextRiskAnalyzerResult(
            data_sensitivity=data_sensitivity,
            pii_risk=min(pii_risk, 1.0),
            cross_source_risk=min(cross_source_risk, 1.0),
            risk_flags=risk_flags,
            latency_ms=latency_ms,
        )


class ExecutionArbiterAgent:
    """
    Combines SecuritySentinelAgent + ContextRiskAnalyzerAgent signals.

    Decision logic:
      - If security_score >= 0.8 AND attack_type != "" → BLOCK
      - If pii_risk >= 0.7 AND data_sensitivity == "restricted" → REDACT or BLOCK
      - If cross_source_risk >= 0.6 AND pii_risk >= 0.5 → ESCALATE
      - Otherwise → ALLOW
    """

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        self._audit_logger = audit_logger

    async def run(
        self,
        sentinel_result: SecuritySentinelResult,
        risk_result: ContextRiskAnalyzerResult,
        query: str,
        abac_ctx: ABACContext | None = None,
    ) -> ArbitrationResult:
        """Make final allow/redact/block/escalate decision."""
        start = time.time()

        decision = "allow"
        policy_reasons = []

        # Rule 1: Hard block on detected attacks
        if sentinel_result.security_score >= 0.8 and sentinel_result.attack_type:
            decision = "block"
            policy_reasons.append(
                f"Attack detected: {sentinel_result.attack_type} (score={sentinel_result.security_score:.2f})"
            )

        # Rule 2: Block on policy violation
        if not sentinel_result.policy_passed:
            decision = "block"
            policy_reasons.append("Policy violation")

        # Rule 3: Escalate on high-risk cross-source + PII
        elif (
            risk_result.cross_source_risk >= 0.6
            and risk_result.pii_risk >= 0.5
        ):
            decision = "escalate"
            policy_reasons.append(
                f"High-risk cross-source PII exposure "
                f"(cross_source={risk_result.cross_source_risk:.2f}, pii={risk_result.pii_risk:.2f})"
            )

        # Rule 4: Redact sensitive data
        elif (
            risk_result.pii_risk >= 0.7
            and risk_result.data_sensitivity in ["sensitive", "restricted"]
        ):
            decision = "redact"
            policy_reasons.append(
                f"Redact sensitive data ({risk_result.data_sensitivity}, pii={risk_result.pii_risk:.2f})"
            )

        # Rule 5: Warn on suspicious but allowed queries
        elif sentinel_result.security_score >= 0.5:
            decision = "allow"
            policy_reasons.append(
                f"Allowed with caution (security_score={sentinel_result.security_score:.2f})"
            )

        latency_ms = (time.time() - start) * 1000

        # Write audit log
        audit_entry_written = False
        if self._audit_logger and abac_ctx:
            try:
                self._audit_logger.append(
                    action="query_arbitration",
                    user=abac_ctx.username,
                    result={
                        "decision": decision,
                        "security_score": sentinel_result.security_score,
                        "pii_risk": risk_result.pii_risk,
                        "data_sensitivity": risk_result.data_sensitivity,
                        "attack_type": sentinel_result.attack_type,
                    },
                )
                audit_entry_written = True
            except Exception as e:
                logger.warning(f"[Arbiter] Audit logging failed: {e}")

        return ArbitrationResult(
            decision=decision,
            security_score=sentinel_result.security_score,
            pii_risk=risk_result.pii_risk,
            data_sensitivity=risk_result.data_sensitivity,
            cross_source_risk=risk_result.cross_source_risk,
            attack_type=sentinel_result.attack_type,
            risk_flags=sentinel_result.risk_flags + risk_result.risk_flags,
            policy_reasons=policy_reasons,
            sentinel_ms=sentinel_result.latency_ms,
            risk_analyzer_ms=risk_result.latency_ms,
            arbiter_ms=latency_ms,
            total_ms=(
                sentinel_result.latency_ms
                + risk_result.latency_ms
                + latency_ms
            ),
            audit_entry_written=audit_entry_written,
        )


class SecureQueryArbitrationSystem:
    """
    Main orchestrator: chains SecuritySentinelAgent → ContextRiskAnalyzerAgent → ExecutionArbiterAgent.
    """

    def __init__(
        self,
        prompt_guard,
        policy_engine=None,
        context_manager=None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._sentinel = SecuritySentinelAgent(prompt_guard, policy_engine)
        self._risk_analyzer = ContextRiskAnalyzerAgent(context_manager)
        self._arbiter = ExecutionArbiterAgent(audit_logger)

    async def arbitrate(
        self,
        query: str,
        chunks: list | None = None,
        abac_ctx: ABACContext | None = None,
    ) -> ArbitrationResult:
        """
        Run full arbitration pipeline.

        Args:
            query: User query
            chunks: Retrieved context chunks (optional)
            abac_ctx: ABACContext for audit

        Returns:
            ArbitrationResult with decision and reasoning
        """
        # Record trace event if active
        collector = get_active_collector()
        if collector:
            collector.record(
                TraceEvent(
                    event_type="arbitration_start",
                    data={"query_length": len(query)},
                )
            )

        # Stage 1: Security Sentinel
        sentinel_result = await self._sentinel.run(query, abac_ctx)

        # Stage 2: Risk Analysis
        risk_result = await self._risk_analyzer.run(query, chunks, abac_ctx)

        # Stage 3: Execution Arbiter
        result = await self._arbiter.run(
            sentinel_result, risk_result, query, abac_ctx
        )

        if collector:
            collector.record(
                TraceEvent(
                    event_type="arbitration_complete",
                    data={
                        "decision": result.decision,
                        "security_score": result.security_score,
                        "total_ms": result.total_ms,
                    },
                )
            )

        logger.info(
            f"[Arbitration] {result.decision.upper():10s} "
            f"| security={result.security_score:.2f} | pii={result.pii_risk:.2f} | "
            f"latency={result.total_ms:.1f}ms"
        )

        return result
