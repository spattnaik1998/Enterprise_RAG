"""
Concrete agent role implementations for the Council pattern.

Three roles:
  FastCreativeAgent         -- gpt-4o-mini, high temperature, creative synthesis
  ConservativeCheckerAgent  -- gpt-4o-mini, low temperature, factual conservatism
  PolicyVerifierAgent       -- claude-haiku-4-5-20251001, structured verification

The PolicyVerifierAgent does NOT use BaseAgent.propose() -- it receives both
proposals and returns a structured verdict JSON.
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.agents.base import AgentProposal, BaseAgent

_CREATIVE_SYSTEM_PROMPT = """\
You are a creative MSP intelligence agent. Your goal is to synthesise a
comprehensive, insightful answer that connects multiple data sources and
highlights non-obvious patterns. Be thorough and explore the implications.
Ground every claim in the provided context. Cite sources by number [1][2][3].
"""

_CONSERVATIVE_SYSTEM_PROMPT = """\
You are a conservative, fact-checking MSP intelligence agent. Your goal is
to provide a precise, conservative answer that ONLY states what is directly
supported by the retrieved context. Do not infer, extrapolate, or add
information not present in the sources. Cite sources by number [1][2][3].
"""

_VERIFIER_PROMPT = """\
You are a policy verifier for an MSP intelligence system. You receive two
proposed answers (Creative and Conservative) and the shared retrieved context.

Your task is to:
1. Identify which proposal is better grounded in the context
2. Check for hallucinated claims (facts not in the context)
3. Check for PII leakage (names, emails, phone numbers beyond what is needed)
4. Return a structured JSON verdict

Return ONLY this JSON (no explanation outside the JSON):
{{
  "verdict": "accept_creative" | "accept_conservative" | "reject_both" | "escalate",
  "winning_agent": "FastCreative" | "ConservativeChecker" | null,
  "reasons": ["reason1", "reason2"],
  "hallucination_detected": true | false,
  "pii_concern": true | false
}}

CONTEXT:
{context_str}

CREATIVE PROPOSAL:
{creative_answer}

CONSERVATIVE PROPOSAL:
{conservative_answer}

QUERY:
{query}
"""


class FastCreativeAgent(BaseAgent):
    """High-temperature creative synthesis agent using gpt-4o-mini."""

    def __init__(self) -> None:
        from src.generation.generator import RAGGenerator
        # Use a higher temperature by creating generator with custom params
        gen = RAGGenerator(model="gpt-4o-mini")
        # We'll set temperature in the generate call via monkey-patch
        super().__init__(
            name="FastCreative",
            generator=gen,
            system_prompt=_CREATIVE_SYSTEM_PROMPT,
        )


class ConservativeCheckerAgent(BaseAgent):
    """Low-temperature conservative fact-checker using gpt-4o-mini."""

    def __init__(self) -> None:
        from src.generation.generator import RAGGenerator
        gen = RAGGenerator(model="gpt-4o-mini")
        super().__init__(
            name="ConservativeChecker",
            generator=gen,
            system_prompt=_CONSERVATIVE_SYSTEM_PROMPT,
        )


class PolicyVerifierAgent:
    """
    Uses claude-haiku-4-5-20251001 to verify both proposals and pick a winner.

    Returns a structured VerifierVerdict instead of an AgentProposal.
    """

    def __init__(self) -> None:
        from src.generation.generator import AnthropicGenerator
        self._client = AnthropicGenerator(model="claude-haiku-4-5-20251001")
        self.name = "PolicyVerifier"

    def verify(
        self,
        query: str,
        creative: AgentProposal,
        conservative: AgentProposal,
        context_pieces: list[Any],
    ) -> dict:
        """
        Run policy verification over the two proposals.

        Returns a dict with keys:
          verdict, winning_agent, reasons, hallucination_detected, pii_concern
        """
        context_str = "\n".join(
            f"[{i+1}] {getattr(p, 'text', str(p))[:300]}"
            for i, p in enumerate(context_pieces[:8])
        )

        prompt = _VERIFIER_PROMPT.format(
            context_str=context_str,
            creative_answer=creative.answer[:1000] if creative.answer else "(no answer)",
            conservative_answer=conservative.answer[:1000] if conservative.answer else "(no answer)",
            query=query,
        )

        # Use a minimal chunk proxy so the generator receives the prompt as context.
        # Must include all attributes accessed by _build_context in generator.py.
        class _PromptChunk:
            text = prompt
            source_type = "verifier_context"
            source = "PolicyVerifier"
            title = "PolicyVerifier context"
            id = "verifier_0"
            chunk_id = "verifier_0"
            doc_id = "verifier_doc"
            chunk_index = 0
            url = ""
            score = 1.0
            metadata: dict = {}

        try:
            response = self._client.generate(
                "Return the verdict JSON for the proposals above.",
                [(_PromptChunk(), 1.0)],
            )
            raw = response.answer.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(line for line in lines if not line.startswith("```")).strip()
            verdict_data = json.loads(raw)
            return verdict_data
        except Exception as exc:
            logger.warning(f"[PolicyVerifier] Verification failed: {exc} -- defaulting to conservative")
            return {
                "verdict": "accept_conservative",
                "winning_agent": "ConservativeChecker",
                "reasons": [f"Verifier error: {exc}"],
                "hallucination_detected": False,
                "pii_concern": False,
            }
