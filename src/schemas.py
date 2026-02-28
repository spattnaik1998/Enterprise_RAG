"""
Core Pydantic schemas for the Enterprise RAG Pipeline.

All stages share these models to guarantee type safety and traceability
end-to-end from raw collection through indexing and serving.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field


# --- Enumerations ------------------------------------------------------------

class SourceType(str, Enum):
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"
    RSS = "rss"
    LOCAL = "local"
    WEB = "web"
    MCP = "mcp"
    # Enterprise MSP source systems
    BILLING = "billing"          # QuickBooks-style invoice/AR system
    PSA = "psa"                  # ConnectWise-style professional services automation
    CRM = "crm"                  # HubSpot-style client relationship management
    COMMUNICATIONS = "communications"  # Exchange/email reminder log
    CONTRACTS = "contracts"      # Contract management / document repository


class DocumentStatus(str, Enum):
    RAW = "raw"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"


# --- Core Document Models -----------------------------------------------------

class RawDocument(BaseModel):
    """
    A document as collected from a source - no quality guarantees yet.

    Traceability fields (source, url, collected_at, metadata) are mandatory
    so every downstream chunk can be traced back to its origin.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str                         # e.g. "arxiv:RAG", "wikipedia:FAISS"
    source_type: SourceType
    title: str
    content: str                        # Raw, uncleaned full text
    url: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def checksum(self) -> str:
        """SHA-256 of content - used for exact-duplicate detection."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @computed_field
    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.content)


class ValidatedDocument(BaseModel):
    """
    A document that passed all validation checks.

    Inherits all RawDocument fields but adds quality metadata
    produced by the validator.
    """

    # RawDocument fields (duplicated to avoid computed_field inheritance issues)
    id: str
    source: str
    source_type: SourceType
    title: str
    content: str
    url: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    collected_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Validation metadata
    status: DocumentStatus = DocumentStatus.VALIDATED
    quality_score: float = 0.0
    validation_notes: list[str] = Field(default_factory=list)
    validated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @computed_field
    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.content)

    @classmethod
    def from_raw(
        cls,
        raw: RawDocument,
        quality_score: float,
        notes: list[str],
    ) -> "ValidatedDocument":
        return cls(
            id=raw.id,
            source=raw.source,
            source_type=raw.source_type,
            title=raw.title,
            content=raw.content,
            url=raw.url,
            authors=raw.authors,
            published_at=raw.published_at,
            collected_at=raw.collected_at,
            metadata=raw.metadata,
            quality_score=quality_score,
            validation_notes=notes,
        )


class RejectedDocument(BaseModel):
    """A document that failed one or more validation checks."""

    id: str
    source: str
    source_type: SourceType
    title: str
    content: str
    url: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    collected_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    status: DocumentStatus = DocumentStatus.REJECTED
    rejection_reasons: list[str] = Field(default_factory=list)
    rejected_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @computed_field
    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @classmethod
    def from_raw(cls, raw: RawDocument, reasons: list[str]) -> "RejectedDocument":
        return cls(
            id=raw.id,
            source=raw.source,
            source_type=raw.source_type,
            title=raw.title,
            content=raw.content,
            url=raw.url,
            authors=raw.authors,
            published_at=raw.published_at,
            collected_at=raw.collected_at,
            metadata=raw.metadata,
            rejection_reasons=reasons,
        )


# --- Pipeline Metadata --------------------------------------------------------

class ValidationResult(BaseModel):
    """Result of validating a single document - not persisted, used internally."""

    document_id: str
    passed: bool
    quality_score: float
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CollectionStats(BaseModel):
    """Aggregated statistics from a collection run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_collected: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    total_validated: int = 0
    total_rejected: int = 0
    errors: list[str] = Field(default_factory=list)
