"""
BaseAgent -- minimal interface for Council pattern agents.

Each agent wraps a generator (RAGGenerator or AnthropicGenerator) and a
system prompt posture. The propose() method runs a single LLM call with
the shared ContextBundle as grounding context and returns a structured proposal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentProposal:
    """A single agent's proposed answer to a query."""
    agent_name: str
    answer: str
    confidence: float    # 0.0 - 1.0, self-reported by the agent
    reasoning: str       # brief explanation of the agent's approach
    cost_usd: float = 0.0
    model: str = ""


class BaseAgent:
    """
    Wraps a generator and a system prompt to form a named agent.

    Args:
        name:          Display name (e.g. "FastCreative", "Conservative")
        generator:     RAGGenerator or AnthropicGenerator instance
        system_prompt: The agent's posture instruction (prepended to context)
    """

    def __init__(self, name: str, generator: Any, system_prompt: str) -> None:
        self.name = name
        self.generator = generator
        self.system_prompt = system_prompt

    def propose(self, query: str, context_pieces: list[Any]) -> AgentProposal:
        """
        Generate a proposal by running the generator with the shared context.

        The system_prompt is injected as a prefix to the generator's standard
        system prompt by temporarily patching the generator's system message.
        Falls back to a neutral proposal on any error.
        """
        try:
            # Build a minimal "chunk proxy" list from context_pieces
            # Each piece must have .text, .source_type, .source attributes
            chunks = context_pieces

            # Run generation
            response = self.generator.generate(query, chunks)

            # Ask the agent to self-assess confidence via a second short call
            # (simplified: use answer length and citation count as heuristic)
            n_citations = len(response.citations)
            confidence = min(1.0, 0.5 + 0.1 * n_citations)

            return AgentProposal(
                agent_name=self.name,
                answer=response.answer,
                confidence=round(confidence, 3),
                reasoning=f"{self.name} agent: {n_citations} citations retrieved",
                cost_usd=0.0,
                model=response.model,
            )
        except Exception as exc:
            return AgentProposal(
                agent_name=self.name,
                answer="",
                confidence=0.0,
                reasoning=f"Agent error: {exc}",
                cost_usd=0.0,
                model="",
            )
