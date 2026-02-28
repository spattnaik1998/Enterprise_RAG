"""
RSS / Atom Feed Collector
-------------------------
Fetches and parses RSS/Atom feeds using feedparser.
Uses httpx for async HTTP and runs feedparser (synchronous) in a thread pool.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import AsyncIterator

import feedparser
import httpx
from loguru import logger

from src.collection.base_collector import BaseCollector
from src.schemas import RawDocument, SourceType
from src.utils.helpers import clean_text

_HEADERS = {"User-Agent": "EnterpriseRAG/1.0 (portfolio; research-only)"}


class RSSCollector(BaseCollector):
    """Collects articles from configured RSS/Atom feeds."""

    source_type = SourceType.RSS

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.feeds: list[dict] = config.get("feeds", [])

    async def health_check(self) -> bool:
        if not self.feeds:
            return False
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(self.feeds[0]["url"], headers=_HEADERS)
                return resp.status_code == 200
        except Exception as exc:
            logger.warning(f"[RSS] Health check failed: {exc}")
            return False

    async def collect(self) -> AsyncIterator[RawDocument]:
        """Yield one RawDocument per feed entry across all configured feeds."""
        for feed_cfg in self.feeds:
            name = feed_cfg.get("name", "Unknown")
            url = feed_cfg["url"]
            logger.info(f"[RSS] Fetching feed: '{name}' - {url}")
            try:
                async for doc in self._fetch_feed(name, url):
                    yield doc
                await asyncio.sleep(1.0)
            except Exception as exc:
                logger.error(f"[RSS] Error on feed '{name}': {exc}")

    async def _fetch_feed(self, name: str, url: str) -> AsyncIterator[RawDocument]:
        """Download and parse a single RSS feed."""
        try:
            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=True
            ) as client:
                resp = await client.get(url, headers=_HEADERS)
                resp.raise_for_status()
                raw_xml = resp.text
        except httpx.HTTPError as exc:
            logger.error(f"[RSS] HTTP error for '{name}': {exc}")
            return

        loop = asyncio.get_event_loop()
        feed = await loop.run_in_executor(None, lambda: feedparser.parse(raw_xml))

        if feed.bozo and not feed.entries:
            logger.warning(f"[RSS] Malformed feed '{name}': {feed.bozo_exception}")
            return

        feed_title = feed.feed.get("title", name)
        logger.info(f"[RSS] '{feed_title}': {len(feed.entries)} entries found")

        for entry in feed.entries:
            title = entry.get("title", "Untitled").strip()

            # Collect content from various feed formats
            content_parts = [title]
            if summary := entry.get("summary", ""):
                content_parts.append(summary)
            elif content_list := entry.get("content", []):
                content_parts.append(content_list[0].get("value", ""))

            content = clean_text("\n\n".join(filter(None, content_parts)))
            if len(content) < 80:
                continue  # Skip stubs

            published_at = self._parse_date(entry.get("published_parsed"))

            yield RawDocument(
                source=f"rss:{name}",
                source_type=SourceType.RSS,
                title=title,
                content=content,
                url=entry.get("link"),
                authors=[a.get("name", "") for a in entry.get("authors", [])],
                published_at=published_at,
                metadata={
                    "feed_name": name,
                    "feed_url": url,
                    "feed_title": feed_title,
                    "entry_id": entry.get("id", ""),
                    "tags": [t.get("term", "") for t in entry.get("tags", [])[:10]],
                },
            )

    @staticmethod
    def _parse_date(parsed_time) -> datetime | None:
        if parsed_time is None:
            return None
        try:
            return datetime.fromtimestamp(time.mktime(parsed_time), tz=timezone.utc)
        except Exception:
            return None
