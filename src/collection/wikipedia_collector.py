"""
Wikipedia Collector
-------------------
Fetches full article text from English Wikipedia using the wikipedia-api library.
Runs synchronous calls in a thread pool executor.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import wikipediaapi
from loguru import logger

from src.collection.base_collector import BaseCollector
from src.schemas import RawDocument, SourceType
from src.utils.helpers import clean_text


# Sections to skip - low-value for RAG retrieval
_SKIP_SECTIONS = frozenset({
    "references", "external links", "see also", "notes",
    "further reading", "bibliography", "citations",
})


class WikipediaCollector(BaseCollector):
    """Collects full article content from English Wikipedia."""

    source_type = SourceType.WIKIPEDIA

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.topics: list[str] = config.get("topics", [])
        self._wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="EnterpriseRAG/1.0 (portfolio; research-only)",
        )

    async def health_check(self) -> bool:
        try:
            loop = asyncio.get_event_loop()
            page = await loop.run_in_executor(
                None, lambda: self._wiki.page("Artificial intelligence")
            )
            return page.exists()
        except Exception as exc:
            logger.warning(f"[Wikipedia] Health check failed: {exc}")
            return False

    async def collect(self) -> AsyncIterator[RawDocument]:
        """Yield one RawDocument per Wikipedia topic."""
        for topic in self.topics:
            logger.info(f"[Wikipedia] Fetching: '{topic}'")
            try:
                doc = await self._fetch_article(topic)
                if doc:
                    yield doc
                await asyncio.sleep(0.5)
            except Exception as exc:
                logger.error(f"[Wikipedia] Error on topic '{topic}': {exc}")

    async def _fetch_article(self, topic: str) -> RawDocument | None:
        loop = asyncio.get_event_loop()

        def _sync() -> RawDocument | None:
            page = self._wiki.page(topic)
            if not page.exists():
                logger.warning(f"[Wikipedia] Page not found: '{topic}'")
                return None

            content_parts = [f"# {page.title}", "", page.summary]

            for section in page.sections:
                if section.title.lower() in _SKIP_SECTIONS:
                    continue
                if section.text.strip():
                    content_parts.append(f"\n## {section.title}\n\n{section.text}")

            full_content = "\n".join(content_parts)

            return RawDocument(
                source=f"wikipedia:{topic}",
                source_type=SourceType.WIKIPEDIA,
                title=page.title,
                content=clean_text(full_content),
                url=page.fullurl,
                metadata={
                    "page_id": page.pageid,
                    "language": "en",
                    "summary_chars": len(page.summary),
                    "total_chars": len(page.text),
                    "sections": [s.title for s in page.sections[:15]],
                    "categories": list(page.categories.keys())[:15],
                },
            )

        return await loop.run_in_executor(None, _sync)
