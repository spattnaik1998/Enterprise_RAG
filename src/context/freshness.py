"""
Freshness Scorer
-----------------
Assigns a freshness score to a Chunk based on its metadata date field.

Score = max(0, 1 - days_since_creation / DECAY_DAYS)
  - 1.0  = created today
  - 0.5  = DECAY_DAYS/2 ago  (~45 days)
  - 0.0  = DECAY_DAYS+ ago   (~90 days or older)
  - 0.5  = no date field (neutral, not penalised)

The 90-day decay window ensures recent MSP billing events and PSA tickets
rank higher than six-month-old RSS articles.
"""
from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger


_DECAY_DAYS = 90.0      # score reaches 0 at this age
_NO_DATE_SCORE = 0.5    # neutral score for chunks without a date field

# Candidate metadata field names (checked in order)
_DATE_FIELDS = (
    "date",
    "created_at",
    "invoice_date",
    "ticket_date",
    "published",
    "updated",
    "expiry_date",
    "effective_date",
)


class FreshnessScorer:
    """
    Scores chunks by how recently they were created.

    Looks for a date value in chunk.metadata using a list of candidate field
    names. Falls back to the neutral score (0.5) if no recognisable date is
    found.
    """

    def score(self, chunk) -> float:
        """
        Return a freshness score in [0, 1].

        Args:
            chunk: A Chunk object (src/chunking/schemas.py) or any object
                   with a .metadata dict attribute.
        """
        metadata = getattr(chunk, "metadata", {}) or {}

        date_str = self._extract_date_str(metadata)
        if not date_str:
            return _NO_DATE_SCORE

        parsed = self._parse_date(date_str)
        if parsed is None:
            return _NO_DATE_SCORE

        now = datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        days_since = (now - parsed).total_seconds() / 86400.0
        score = max(0.0, 1.0 - days_since / _DECAY_DAYS)
        return round(score, 4)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_date_str(metadata: dict) -> str | None:
        for field in _DATE_FIELDS:
            val = metadata.get(field)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return None

    @staticmethod
    def _parse_date(date_str: str) -> datetime | None:
        """Try a series of date format parsers. Returns None if none match."""
        # ISO 8601 / datetime string
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
            "%Y-%m",
            "%B %Y",   # e.g. "January 2025"
            "%b %Y",   # e.g. "Jan 2025"
            "%d/%m/%Y",
            "%m/%d/%Y",
        ):
            try:
                return datetime.strptime(date_str[:len(fmt)+5], fmt)
            except ValueError:
                pass

        # Try dateutil if installed
        try:
            from dateutil import parser as dateutil_parser
            return dateutil_parser.parse(date_str, fuzzy=True)
        except Exception:
            pass

        logger.debug(f"[FreshnessScorer] Could not parse date: {date_str!r}")
        return None
