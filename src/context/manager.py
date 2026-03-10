"""
ContextManager
---------------
Budget-aware context assembly for the RAG pipeline.

Takes a list of reranked Chunks, scores them by freshness + tier, packs
them greedily within a token budget, then reorders for "lost in the middle"
mitigation before returning a ContextBundle.

API:
    manager = ContextManager()
    bundle  = manager.get_context(query, chunks, budget_tokens=3000)
    # bundle.pieces is the final ordered list for the generator
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.context.schemas import ContextBundle, ContextPiece
from src.context.freshness import FreshnessScorer
from src.context.tiers import TierClassifier


# ---------------------------------------------------------------------------
# Token counting (tiktoken cl100k_base — same encoder as OpenAI embeddings)
# ---------------------------------------------------------------------------

_ENCODER = None


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken, with a fallback to word-based estimate."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = False  # sentinel: tiktoken unavailable
    if _ENCODER is False:
        return max(1, len(text.split()) * 4 // 3)
    return len(_ENCODER.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

_DEFAULT_BUDGET    = 3000
_FAST_PATH_BUDGET  = 1024


class ContextManager:
    """
    Assembles a cost-optimised ContextBundle from reranked chunks.

    Usage:
        manager = ContextManager()
        bundle  = manager.get_context(query, reranked_chunks, budget_tokens=3000)
        # Pass bundle.pieces to the generator
    """

    def __init__(self) -> None:
        self._freshness = FreshnessScorer()
        self._tiers     = TierClassifier()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_context(
        self,
        query: str,
        chunks: list,
        budget_tokens: int = _DEFAULT_BUDGET,
        fast_path: bool = False,
    ) -> ContextBundle:
        """
        Build a ContextBundle from reranked chunks.

        Args:
            query:         The user query (reserved for future query-aware scoring).
            chunks:        Ordered list of Chunk objects from LLMReranker.
            budget_tokens: Maximum tokens to include. Set to 1024 for fast path.
            fast_path:     If True, uses FAST_PATH_BUDGET (1024) regardless of
                           budget_tokens param.

        Returns:
            ContextBundle with pieces packed and reordered.
        """
        if fast_path:
            budget_tokens = _FAST_PATH_BUDGET

        if not chunks:
            return ContextBundle(
                pieces=[], all_pieces=[], total_tokens=0,
                budget_tokens=budget_tokens, truncated=False,
                dropped_count=0, strategy_used="empty", fast_path=fast_path,
            )

        # 1. Convert chunks to ContextPiece objects
        all_pieces = self._to_pieces(chunks)

        # 2. Sort by (tier_priority DESC, combined_score DESC)
        all_pieces.sort(
            key=lambda p: (self._tiers.priority(p.tier), p.combined_score),
            reverse=True,
        )

        # 3. Greedy token-budget packing
        selected: list[ContextPiece] = []
        total_tokens = 0
        for piece in all_pieces:
            if total_tokens + piece.tokens <= budget_tokens:
                selected.append(piece)
                total_tokens += piece.tokens
            else:
                piece.included = False

        dropped = len(all_pieces) - len(selected)
        truncated = dropped > 0
        strategy = "budget_constrained" if truncated else "full"

        # 4. Reorder for "lost in the middle" mitigation
        reordered = self._tiers.reorder_for_lim(selected)

        logger.debug(
            f"[ContextManager] "
            f"chunks={len(chunks)} selected={len(selected)} "
            f"dropped={dropped} tokens={total_tokens}/{budget_tokens} "
            f"strategy={strategy} fast_path={fast_path}"
        )

        return ContextBundle(
            pieces=reordered,
            all_pieces=all_pieces,
            total_tokens=total_tokens,
            budget_tokens=budget_tokens,
            truncated=truncated,
            dropped_count=dropped,
            strategy_used=strategy,
            fast_path=fast_path,
        )

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _to_pieces(self, chunks: list) -> list[ContextPiece]:
        pieces: list[ContextPiece] = []
        for i, chunk in enumerate(chunks):
            text     = getattr(chunk, "text", "") or ""
            tokens   = _count_tokens(text)
            src_type = getattr(chunk, "source_type", "")
            source   = getattr(chunk, "source", "")
            cid      = getattr(chunk, "id", f"chunk_{i}")

            # Scores
            relevance = float(getattr(chunk, "score", 0.0) or 0.0)
            freshness = self._freshness.score(chunk)
            combined  = round(relevance * (0.7 + 0.3 * freshness), 6)
            # Weight: 70% relevance, 30% freshness boost

            tier = self._tiers.classify(src_type)

            pieces.append(
                ContextPiece(
                    id=str(cid),
                    chunk_index=i,
                    text=text,
                    tokens=tokens,
                    source_type=src_type,
                    source=source,
                    relevance_score=relevance,
                    freshness_score=freshness,
                    combined_score=combined,
                    tier=tier,
                    included=True,
                )
            )
        return pieces
