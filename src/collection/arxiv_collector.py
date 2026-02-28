"""
ArXiv Collector
---------------
Fetches research papers from arxiv.org using the official arxiv Python SDK.
Runs the synchronous SDK calls in a thread pool to remain async-friendly.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import arxiv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.collection.base_collector import BaseCollector
from src.schemas import RawDocument, SourceType
from src.utils.helpers import clean_text


class ArXivCollector(BaseCollector):
    """Collects research papers from ArXiv for configured search queries."""

    source_type = SourceType.ARXIV

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.queries: list[str] = config.get("queries", [])
        self.max_results: int = config.get("max_results", 20)
        self.categories: list[str] = config.get("categories", ["cs.AI"])
        self._client = arxiv.Client(
            page_size=min(self.max_results, 100),
            delay_seconds=3.0,
            num_retries=3,
        )

    async def health_check(self) -> bool:
        try:
            papers = await self._fetch_papers("test", max_results=1)
            return len(papers) > 0
        except Exception as exc:
            logger.warning(f"[ArXiv] Health check failed: {exc}")
            return False

    async def collect(self) -> AsyncIterator[RawDocument]:
        """Yield one RawDocument per ArXiv paper across all configured queries."""
        seen_ids: set[str] = set()

        for query in self.queries:
            logger.info(f"[ArXiv] Query: '{query}' (max {self.max_results})")
            try:
                papers = await self._fetch_papers(query, max_results=self.max_results)
                new_count = 0

                for paper in papers:
                    if paper.entry_id in seen_ids:
                        continue
                    seen_ids.add(paper.entry_id)
                    new_count += 1

                    content = self._build_content(paper)
                    yield RawDocument(
                        source=f"arxiv:{query[:40]}",
                        source_type=SourceType.ARXIV,
                        title=paper.title.strip(),
                        content=clean_text(content),
                        url=paper.entry_id,
                        authors=[str(a) for a in paper.authors[:8]],
                        published_at=paper.published,
                        metadata={
                            "arxiv_id": paper.entry_id,
                            "categories": paper.categories,
                            "primary_category": paper.primary_category,
                            "doi": paper.doi,
                            "journal_ref": paper.journal_ref,
                            "pdf_url": str(paper.pdf_url) if paper.pdf_url else None,
                            "query": query,
                            "updated": paper.updated.isoformat() if paper.updated else None,
                        },
                    )

                logger.info(f"[ArXiv] '{query}': {new_count} new papers")
                await asyncio.sleep(1.5)  # Respect rate limits

            except Exception as exc:
                logger.error(f"[ArXiv] Error on query '{query}': {exc}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch_papers(self, query: str, max_results: int) -> list:
        """Execute a synchronous ArXiv search in a thread-pool executor."""
        loop = asyncio.get_event_loop()

        def _sync() -> list:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            return list(self._client.results(search))

        return await loop.run_in_executor(None, _sync)

    @staticmethod
    def _build_content(paper) -> str:
        parts = [
            paper.title,
            "",
            f"Authors: {', '.join(str(a) for a in paper.authors[:8])}",
            f"Published: {paper.published.strftime('%Y-%m-%d') if paper.published else 'Unknown'}",
            f"Categories: {', '.join(paper.categories)}",
            "",
            "Abstract:",
            paper.summary,
        ]
        if paper.pdf_url:
            parts.append(f"\nPDF: {paper.pdf_url}")
        return "\n".join(parts)
