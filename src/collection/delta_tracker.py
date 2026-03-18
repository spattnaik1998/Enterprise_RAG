"""
Delta Tracker
-------------
Tracks last collection checkpoint per source for incremental collection.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger


class DeltaTracker:
    """Manages delta collection checkpoints per source."""

    def __init__(self, checkpoint_file: str | Path = "data/delta_checkpoints.json"):
        """
        Initialize delta tracker.

        Args:
            checkpoint_file: Path to checkpoint JSON file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoints = self._load_checkpoints()

    def get_checkpoint(self, source: str) -> Optional[str]:
        """
        Get last checkpoint for a source.

        Args:
            source: Source name (e.g., "billing", "psa", "crm", "contracts", "comms")

        Returns:
            ISO-formatted datetime string, or None if no checkpoint exists
        """
        return self.checkpoints.get(source)

    def set_checkpoint(self, source: str, timestamp: Optional[str] = None) -> None:
        """
        Set checkpoint for a source.

        Args:
            source: Source name
            timestamp: ISO-formatted datetime (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        self.checkpoints[source] = timestamp
        self._save_checkpoints()
        logger.info(f"[DeltaTracker] Updated checkpoint for {source}: {timestamp}")

    def should_collect(self, source: str, force: bool = False) -> bool:
        """
        Check if source should be collected.

        Args:
            source: Source name
            force: If True, always return True (for --full mode)

        Returns:
            True if checkpoint missing or force=True
        """
        if force:
            return True
        return source not in self.checkpoints

    def _load_checkpoints(self) -> dict:
        """Load checkpoints from file."""
        if not self.checkpoint_file.exists():
            return {}

        try:
            with open(self.checkpoint_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[DeltaTracker] Failed to load checkpoints: {e}")
            return {}

    def _save_checkpoints(self) -> None:
        """Save checkpoints to file."""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.checkpoints, f, indent=2)
        except Exception as e:
            logger.warning(f"[DeltaTracker] Failed to save checkpoints: {e}")

    def reset(self, source: Optional[str] = None) -> None:
        """
        Reset checkpoint(s).

        Args:
            source: Source to reset, or None to reset all
        """
        if source:
            if source in self.checkpoints:
                del self.checkpoints[source]
                self._save_checkpoints()
                logger.info(f"[DeltaTracker] Reset checkpoint for {source}")
        else:
            self.checkpoints.clear()
            self._save_checkpoints()
            logger.info("[DeltaTracker] Reset all checkpoints")
