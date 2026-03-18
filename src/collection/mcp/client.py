"""
Enterprise RAG - MCP Collection Client
---------------------------------------
Thin async client that spawns the MCP server as a subprocess and
invokes its enterprise MSP tools over stdio.

Usage (async context manager):
    async with MCPCollectorClient() as client:
        # Use enterprise tools (billing, PSA, CRM, comms, contracts)
        pass
"""
from __future__ import annotations

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

