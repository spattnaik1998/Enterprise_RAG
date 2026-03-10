"""
Guardrails
-----------
Two protection layers applied around every RAG query:

1. PromptGuard  -- Detects prompt injection attempts in user input before
                   the query reaches the retriever or LLM.  Uses a curated
                   set of regex patterns covering common injection signatures
                   (jailbreaks, role overrides, instruction ignoring).
                   Supplemental patterns are loaded from
                   config/injection_patterns.yaml at startup.

2. PIIFilter    -- Redacts personally identifiable information from LLM-
                   generated answers before they are returned to the caller.
                   Targets email addresses, phone numbers, SSNs, credit-card
                   numbers, and IP addresses using non-overlapping regex subs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Prompt Injection Patterns (hardcoded baseline — always active)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you|i)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+\w+", re.I),
    re.compile(r"act\s+as\s+(a|an|the)\s+\w+", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bDAN\s+mode\b", re.I),
    re.compile(r"system\s+prompt\s*:", re.I),
    re.compile(r"<\s*/?system\s*>", re.I),
    re.compile(r"\[\s*INST\s*\]", re.I),
    re.compile(r"new\s+instructions?\s*:", re.I),
    re.compile(r"override\s+(your\s+)?(instructions?|rules?|guidelines?)", re.I),
]

# Supplemental patterns loaded from YAML (merged at module load)
_SUPPLEMENTAL_PATTERNS: list[re.Pattern[str]] = []
_SUPPLEMENTAL_LOADED = False


def _load_supplemental_patterns() -> list[re.Pattern[str]]:
    """Load extra injection patterns from config/injection_patterns.yaml."""
    global _SUPPLEMENTAL_LOADED
    if _SUPPLEMENTAL_LOADED:
        return _SUPPLEMENTAL_PATTERNS

    yaml_path = Path("config/injection_patterns.yaml")
    if not yaml_path.exists():
        _SUPPLEMENTAL_LOADED = True
        return _SUPPLEMENTAL_PATTERNS

    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        loaded = 0
        for entry in data.get("patterns", []):
            pattern_str = entry.get("pattern", "")
            if not pattern_str:
                continue
            try:
                compiled = re.compile(pattern_str, re.I)
                _SUPPLEMENTAL_PATTERNS.append(compiled)
                loaded += 1
            except re.error as exc:
                logger.warning(
                    f"[PromptGuard] Skipping bad pattern "
                    f"id={entry.get('id')!r}: {exc}"
                )
        logger.info(
            f"[PromptGuard] Loaded {loaded} supplemental injection patterns "
            f"from {yaml_path}"
        )
    except Exception as exc:
        logger.warning(f"[PromptGuard] Failed to load supplemental patterns: {exc}")

    _SUPPLEMENTAL_LOADED = True
    return _SUPPLEMENTAL_PATTERNS


# ---------------------------------------------------------------------------
# PII Redaction Patterns  (label, compiled pattern)
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "EMAIL",
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ),
    (
        "PHONE",
        re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    ),
    (
        "SSN",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "CC",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ),
    (
        "IP",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ),
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Result of running PromptGuard on a user query."""
    passed: bool
    blocked_reason: str = ""
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PromptGuard
# ---------------------------------------------------------------------------

class PromptGuard:
    """
    Scans user queries for prompt injection signatures.

    Usage:
        guard = PromptGuard()
        result = guard.check(query)
        if not result.passed:
            return result.blocked_reason
    """

    def check(self, query: str) -> GuardrailResult:
        """
        Return GuardrailResult.passed=True if the query is safe,
        False if an injection pattern is detected.

        Checks both the hardcoded baseline patterns and supplemental
        patterns loaded from config/injection_patterns.yaml.
        """
        flags: list[str] = []
        all_patterns = _INJECTION_PATTERNS + _load_supplemental_patterns()
        for pattern in all_patterns:
            if pattern.search(query):
                flags.append(pattern.pattern)

        if flags:
            logger.warning(
                f"[PromptGuard] Injection detected | query={query[:80]!r} | "
                f"patterns={len(flags)}"
            )
            return GuardrailResult(
                passed=False,
                blocked_reason=(
                    "Your query contains patterns that look like prompt injection. "
                    "Please rephrase your question about TechVault operations or AI/ML research."
                ),
                flags=flags,
            )

        return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# PIIFilter
# ---------------------------------------------------------------------------

class PIIFilter:
    """
    Redacts PII from LLM-generated text before returning it to the caller.

    Patterns covered: email, phone, SSN, credit card, IP address.

    Usage:
        pii = PIIFilter()
        clean_text, redacted_types = pii.redact(raw_answer)
    """

    def redact(self, text: str) -> tuple[str, list[str]]:
        """
        Replace PII occurrences with [REDACTED_<TYPE>] placeholders.

        Returns:
            (redacted_text, list_of_redacted_type_labels)
        """
        redacted_types: list[str] = []
        for label, pattern in _PII_PATTERNS:
            new_text, count = pattern.subn(f"[REDACTED_{label}]", text)
            if count > 0:
                text = new_text
                redacted_types.append(label)

        if redacted_types:
            logger.info(f"[PIIFilter] Redacted types: {redacted_types}")

        return text, redacted_types
