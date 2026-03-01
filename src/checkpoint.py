"""
Phase I -> Phase II Human Checkpoint
-------------------------------------
Saves all validated and rejected documents to disk and writes a
machine-readable checkpoint state file (data/checkpoint_phase1.json).

The pipeline deliberately STOPS here - no Phase II code runs automatically.
A human must inspect the data and then invoke Phase II explicitly.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from src.schemas import CollectionStats, RejectedDocument, ValidatedDocument
from src.utils.helpers import ensure_dirs, save_json


class PhaseICheckpoint:
    """Persists Phase I output and writes the handoff state."""

    def __init__(self, config: dict) -> None:
        storage = config.get("storage", {})
        self.validated_dir = Path(storage.get("validated_dir", "data/validated"))
        self.rejected_dir = Path(storage.get("rejected_dir", "data/rejected"))

    async def save(
        self,
        validated: list[ValidatedDocument],
        rejected: list[RejectedDocument],
        stats: CollectionStats,
    ) -> None:
        """Persist documents and write checkpoint state."""
        ensure_dirs(self.validated_dir, self.rejected_dir)

        self._persist(validated, self.validated_dir)
        self._persist(rejected, self.rejected_dir)

        state = {
            "phase": "I",
            "status": "awaiting_human_review",
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": stats.run_id,
            "validated_count": len(validated),
            "rejected_count": len(rejected),
            "validated_dir": str(self.validated_dir),
            "rejected_dir": str(self.rejected_dir),
            "next_phase": "II - Chunking & Embedding",
            "next_command": "python -m src.main phase2",
            "instructions": (
                "1. Review validated documents in data/validated/.\n"
                "2. Check data/validation_report.json for quality metrics.\n"
                "3. Optionally move any rejected docs you want to include back to validated/.\n"
                "4. When satisfied, run: python -m src.main phase2"
            ),
        }
        save_json(state, "data/checkpoint_phase1.json")
        logger.info(
            f"Checkpoint written - {len(validated)} validated, "
            f"{len(rejected)} rejected.  Next: Phase II."
        )

    @staticmethod
    def _persist(docs: list, directory: Path) -> None:
        for doc in docs:
            safe_id = doc.id.replace("/", "_").replace(":", "_").replace(" ", "_")
            path = directory / f"{doc.source_type.value}_{safe_id}.json"
            save_json(doc.model_dump(mode="json"), path)
        logger.debug(f"Saved {len(docs)} documents -> {directory}/")
