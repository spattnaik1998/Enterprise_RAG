"""
Chunk schema - the atomic unit that gets embedded and indexed.

A Chunk traces back to its parent ValidatedDocument so every
retrieval result carries full provenance for citations.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    A single embeddable text window produced from a ValidatedDocument.

    Traceability fields (doc_id, source, source_type) are inherited from
    the parent document so the serving layer can cite the origin of every
    retrieved passage.
    """

    # Identity
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str                          # Parent ValidatedDocument.id
    chunk_index: int                     # Position within the document
    chunk_strategy: str                  # "keep_whole" | "sentence_window" | "fixed_overlap"

    # Content
    text: str                            # The actual text to embed
    token_count: int = 0                 # Populated by the chunker

    # Provenance (copied from parent doc for zero-join retrieval)
    source: str                          # e.g. "billing:quickbooks:INV-2025-1042"
    source_type: str                     # e.g. "billing"
    title: str                           # Parent document title
    url: Optional[str] = None

    # Metadata pass-through (key business fields for metadata filtering)
    metadata: dict[str, Any] = Field(default_factory=dict)
