"""
Context schemas — ContextPiece and ContextBundle.

These are the public types returned by ContextManager.get_context().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ContextPiece:
    """
    A single context item selected for the LLM context window.

    All fields are populated by ContextManager; callers receive these
    read-only via ContextBundle.
    """

    # Identity
    id: str
    """Unique chunk ID from the underlying Chunk dataclass."""

    chunk_index: int
    """Original position in the retrieval result list (0-based)."""

    # Content
    text: str
    """Chunk text passed to the LLM."""

    tokens: int
    """Approximate token count (tiktoken cl100k_base)."""

    # Provenance
    source_type: str
    """Source system: billing | psa | crm | contracts | communications"""

    source: str
    """Document source name (e.g. invoice ID, paper title)."""

    # Scoring
    relevance_score: float
    """RRF relevance score from HybridRetriever (0–1 range)."""

    freshness_score: float
    """Time-decay score: 1.0 = today, 0.5 = no date, decays over 90 days."""

    combined_score: float
    """relevance_score * freshness_score — used for final ranking."""

    # Context engineering
    tier: Literal["working", "ephemeral", "longterm"]
    """
    working   -- operationally immediate (billing, psa, communications)
    ephemeral -- query-relevant (crm, contracts)
    longterm  -- reserved for future background knowledge sources
    """

    context_position: int = 0
    """Final position in the context window after "lost in the middle" reorder."""

    included: bool = True
    """False if this piece was retrieved but dropped due to token budget."""


@dataclass
class ContextBundle:
    """
    The complete output of ContextManager.get_context().

    `pieces` is the ordered list of context items to pass to the generator.
    Pieces are already reordered for "lost in the middle" mitigation:
    working-tier items are placed first and last; longterm items in the middle.
    """

    pieces: list[ContextPiece]
    """Selected and reordered context pieces (budget-constrained)."""

    all_pieces: list[ContextPiece]
    """All scored pieces before budget truncation (for UI progressive disclosure)."""

    total_tokens: int
    """Sum of tokens across included pieces."""

    budget_tokens: int
    """The token budget this bundle was built against."""

    truncated: bool
    """True if pieces were dropped to stay within budget."""

    dropped_count: int
    """Number of candidate pieces dropped due to budget."""

    strategy_used: str
    """'budget_constrained' or 'full' (when all pieces fit within budget)."""

    fast_path: bool = False
    """True when the budget-constrained fast path was used."""

    def to_api_dict(self) -> list[dict]:
        """Serialise for the /api/chat context_pieces response field."""
        return [
            {
                "id":              p.id,
                "tier":            p.tier,
                "source_type":     p.source_type,
                "relevance_score": round(p.relevance_score, 4),
                "freshness_score": round(p.freshness_score, 4),
                "tokens":          p.tokens,
                "included":        p.included,
                "context_position": p.context_position,
            }
            for p in self.all_pieces
        ]
