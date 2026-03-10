"""
Tier Classifier
----------------
Maps a source_type string to one of three context tiers:

  working   -- operationally immediate data (billing, psa, communications)
               These are placed at the FIRST and LAST positions in the context
               window to mitigate "lost in the middle" degradation.

  ephemeral -- query-relevant but not time-critical (crm, contracts)
               Placed just inside the working-tier bookends.

  longterm  -- background knowledge (arxiv, wikipedia, rss)
               Placed in the middle where recency matters least.

Tier priority (for sort order — higher = ranked earlier):
  working   -> 3
  ephemeral -> 2
  longterm  -> 1
"""
from __future__ import annotations

from typing import Literal


TierType = Literal["working", "ephemeral", "longterm"]

_TIER_MAP: dict[str, TierType] = {
    # Working tier — operationally immediate MSP data
    "billing":        "working",
    "psa":            "working",
    "communications": "working",

    # Ephemeral tier — query-relevant but not urgent
    "crm":       "ephemeral",
    "contracts": "ephemeral",

    # Longterm tier — background knowledge
    "arxiv":     "longterm",
    "wikipedia": "longterm",
    "rss":       "longterm",
}

_TIER_PRIORITY: dict[TierType, int] = {
    "working":   3,
    "ephemeral": 2,
    "longterm":  1,
}


class TierClassifier:
    """
    Classifies a source_type into a context tier.

    Unknown source types default to 'ephemeral' (mid-priority).
    """

    def classify(self, source_type: str) -> TierType:
        """Return the tier for the given source_type string."""
        return _TIER_MAP.get(source_type.lower().strip(), "ephemeral")

    def priority(self, tier: TierType) -> int:
        """Return the numeric sort priority (higher = ranked earlier)."""
        return _TIER_PRIORITY.get(tier, 2)

    def reorder_for_lim(self, pieces: list) -> list:
        """
        Reorder context pieces to mitigate "lost in the middle" degradation.

        Strategy:
          - Place the highest-priority (working-tier) pieces at positions
            0..(K-1) and -(K)..-1  (first K and last K slots).
          - Fill the middle with ephemeral and longterm pieces.

        K = number of working-tier pieces, capped at ceil(len(pieces)/3).
        """
        if not pieces:
            return pieces

        working   = [p for p in pieces if p.tier == "working"]
        middle    = [p for p in pieces if p.tier in ("ephemeral", "longterm")]

        # Split working pieces: first half at top, second half at bottom
        mid_w  = len(working) // 2
        top    = working[:mid_w + len(working) % 2]   # ceiling half
        bottom = working[mid_w + len(working) % 2:]

        reordered = top + middle + bottom

        # Update context_position
        for i, piece in enumerate(reordered):
            piece.context_position = i

        return reordered
