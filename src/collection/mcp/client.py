"""
Enterprise RAG - MCP Collection Client
---------------------------------------
Thin async client that spawns the MCP server as a subprocess and
invokes its tools over stdio.

Usage (async context manager):
    async with MCPCollectorClient() as client:
        papers = await client.search_arxiv("retrieval augmented generation")
        article = await client.fetch_wikipedia("FAISS")
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPCollectorClient:
    """
    Async context-manager wrapper around the Enterprise RAG MCP server.

    The server process is spawned on __aenter__ and torn down on __aexit__.
    """

    def __init__(self, server_module: str = "src.collection.mcp.server") -> None:
        self._server_module = server_module
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._session_ctx = None

    async def __aenter__(self) -> "MCPCollectorClient":
        await self._connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self._disconnect()

    # --- Connection -----------------------------------------------------------

    async def _connect(self) -> None:
        params = StdioServerParameters(
            command="python",
            args=["-m", self._server_module],
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()

        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

        tools = await self._session.list_tools()
        tool_names = [t.name for t in tools.tools]
        logger.info(f"[MCP Client] Connected | tools: {tool_names}")

    async def _disconnect(self) -> None:
        if self._session_ctx:
            await self._session_ctx.__aexit__(None, None, None)
        if self._stdio_ctx:
            await self._stdio_ctx.__aexit__(None, None, None)

    # --- Tool Wrappers --------------------------------------------------------

    async def search_arxiv(self, query: str, max_results: int = 10) -> list[dict]:
        """Search ArXiv and return a list of paper dicts."""
        result = await self._session.call_tool(
            "search_arxiv", {"query": query, "max_results": max_results}
        )
        return json.loads(result.content[0].text)

    async def fetch_wikipedia(self, topic: str) -> dict:
        """Fetch a Wikipedia article and return its content dict."""
        result = await self._session.call_tool("fetch_wikipedia", {"topic": topic})
        return json.loads(result.content[0].text)

    async def fetch_rss_feed(self, url: str, max_items: int = 20) -> dict:
        """Fetch and parse an RSS/Atom feed."""
        result = await self._session.call_tool(
            "fetch_rss_feed", {"url": url, "max_items": max_items}
        )
        return json.loads(result.content[0].text)

    async def fetch_webpage(self, url: str) -> dict:
        """Fetch raw text from a webpage."""
        result = await self._session.call_tool("fetch_webpage", {"url": url})
        return json.loads(result.content[0].text)

    async def list_sources(self) -> dict:
        """List all sources the server can collect from."""
        result = await self._session.call_tool("list_available_sources", {})
        return json.loads(result.content[0].text)
