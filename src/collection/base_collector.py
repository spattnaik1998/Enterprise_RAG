"""Abstract base class for all data collectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from loguru import logger

from src.schemas import RawDocument, SourceType


class BaseCollector(ABC):
    """
    All collectors inherit from this class.

    Guarantees a uniform async-generator interface so the
    CollectionPipeline can treat every source identically.
    """

    source_type: SourceType

    def __init__(self, config: dict) -> None:
        self.config = config
        self.collected_count: int = 0
        self.error_count: int = 0

    @abstractmethod
    async def collect(self) -> AsyncIterator[RawDocument]:
        """Yield RawDocument instances from the source."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the source is reachable and responsive."""
        ...

    async def safe_collect(self) -> AsyncIterator[RawDocument]:
        """collect() wrapped with per-document error isolation."""
        try:
            async for doc in self.collect():
                self.collected_count += 1
                yield doc
        except Exception as exc:
            self.error_count += 1
            logger.error(f"[{self.__class__.__name__}] Unhandled collection error: {exc}")
