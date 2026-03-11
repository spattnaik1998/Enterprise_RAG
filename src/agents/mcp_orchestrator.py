"""
MCP Orchestrator (Sprint 4, Cross-Cutting)
------------------------------------------
Dynamically composes MCP tools for complex enterprise queries.

Uses keyword heuristics (no LLM) to:
  1. Determine which tools to invoke
  2. Execute tools in parallel
  3. Synthesize results into coherent answer

Tool selection strategy:
  - Overdue/invoice/balance → billing_get_overdue_invoices
  - Ticket/psa/technician → psa_get_client_tickets
  - Profile/health/crm → crm_get_client_profile
  - Contract/sla/renewal → contracts_get_terms
  - Client 360/full picture/multi-source → get_client_360
  - Email/comms → comms_get_invoice_history

Usage:
    orchestrator = MCPOrchestrator(pipeline=pipeline)
    result = await orchestrator.compose(query="Which clients have overdue invoices?", ctx=abac_ctx)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from loguru import logger


@dataclass
class MCPCompositionResult:
    """Output from MCPOrchestrator."""
    answer: str
    tools_used: list[str]
    tool_results: dict  # tool_name -> result
    tool_latencies: dict  # tool_name -> latency_ms
    total_latency_ms: float
    cost_usd: float
    data_sources: list[str]


class MCPOrchestrator:
    """
    Dynamically composes MCP tools for complex enterprise queries.

    Note: This is a simplified implementation using keyword heuristics.
    A production system would use LLM-based planning.
    """

    # Tool selection keywords
    BILLING_KEYWORDS = ["overdue", "invoice", "balance", "payment", "receivable", "ar"]
    PSA_KEYWORDS = ["ticket", "psa", "technician", "resolved", "billable", "hours"]
    CRM_KEYWORDS = ["profile", "health", "account", "contact", "industry", "risk"]
    CONTRACT_KEYWORDS = ["contract", "sla", "renewal", "termination", "penalty", "term"]
    COMMS_KEYWORDS = ["email", "comms", "reminder", "invoice", "communication"]
    CLIENT_360_KEYWORDS = ["client", "full", "picture", "summary", "overview", "360"]

    def __init__(self, mcp_tools: dict | None = None) -> None:
        """
        Args:
            mcp_tools: Dict mapping tool names to callable functions.
                      If None, tools will be loaded lazily from src.collection.mcp.server
        """
        self._mcp_tools = mcp_tools or {}

    async def _load_tools_if_needed(self) -> None:
        """Lazy-load MCP tools if not already loaded."""
        if self._mcp_tools:
            return

        try:
            # Import tool server to get available tools
            # This is a placeholder; actual implementation would import from mcp.server
            logger.info("[MCPOrchestrator] Loading MCP tools (placeholder)")
            # In production, would do:
            # from src.collection.mcp.server import get_available_tools
            # self._mcp_tools = get_available_tools()
        except Exception as e:
            logger.warning(f"[MCPOrchestrator] Failed to load MCP tools: {e}")

    def _plan_tools(self, query: str) -> list[str]:
        """Determine which tools to invoke based on query keywords."""
        query_lower = query.lower()
        tools = []

        # Check for specific tool matches
        if any(kw in query_lower for kw in self.BILLING_KEYWORDS):
            tools.extend([
                "billing_get_overdue_invoices",
                "billing_get_aged_receivables",
            ])

        if any(kw in query_lower for kw in self.PSA_KEYWORDS):
            tools.extend([
                "psa_get_client_tickets",
                "psa_get_unbilled_work",
            ])

        if any(kw in query_lower for kw in self.CRM_KEYWORDS):
            tools.extend([
                "crm_get_client_profile",
                "crm_get_at_risk_accounts",
            ])

        if any(kw in query_lower for kw in self.CONTRACT_KEYWORDS):
            tools.append("contracts_get_terms")

        if any(kw in query_lower for kw in self.COMMS_KEYWORDS):
            tools.append("comms_get_invoice_history")

        # Client 360 for comprehensive queries
        if any(kw in query_lower for kw in self.CLIENT_360_KEYWORDS):
            tools = ["get_client_360"]

        # De-duplicate
        tools = list(set(tools))

        logger.info(f"[MCPOrchestrator] Planned tools: {tools}")
        return tools

    async def _execute_tools(
        self,
        tools: list[str],
        query: str,
        ctx=None,
    ) -> dict:
        """
        Execute selected tools in parallel.

        Returns:
            Dict mapping tool_name -> result
        """
        results = {}
        latencies = {}

        # Create tasks for each tool
        tasks = []
        for tool_name in tools:
            if tool_name in self._mcp_tools:
                task = self._execute_single_tool(
                    tool_name,
                    query,
                    ctx,
                )
                tasks.append((tool_name, task))
            else:
                logger.warning(f"[MCPOrchestrator] Tool not found: {tool_name}")

        # Execute all tasks in parallel
        if tasks:
            for tool_name, task in tasks:
                try:
                    start = time.time()
                    result = await asyncio.wait_for(task, timeout=10.0)
                    latencies[tool_name] = (time.time() - start) * 1000
                    results[tool_name] = result
                except asyncio.TimeoutError:
                    logger.warning(f"[MCPOrchestrator] Tool {tool_name} timed out")
                    results[tool_name] = {"error": "timeout"}
                except Exception as e:
                    logger.warning(f"[MCPOrchestrator] Tool {tool_name} failed: {e}")
                    results[tool_name] = {"error": str(e)}

        return results, latencies

    async def _execute_single_tool(self, tool_name: str, query: str, ctx) -> dict:
        """Execute a single tool (placeholder)."""
        tool_fn = self._mcp_tools.get(tool_name)
        if not tool_fn:
            return {"error": f"Tool {tool_name} not available"}

        # Call tool (may be async or sync)
        try:
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(query=query, ctx=ctx)
            else:
                result = tool_fn(query=query, ctx=ctx)
            return result
        except Exception as e:
            logger.warning(f"[MCPOrchestrator] Error executing {tool_name}: {e}")
            return {"error": str(e)}

    def _synthesize(self, tool_results: dict, query: str) -> str:
        """
        Synthesize tool results into a coherent answer.

        For now, this is a simple concatenation. In production, would use LLM.
        """
        if not tool_results:
            return "No tools were executed."

        answer_parts = []

        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and "error" in result:
                continue

            # Format result based on tool type
            if "get_client_360" in tool_name:
                answer_parts.append(f"**Client 360 View:**\n{str(result)}")
            elif "billing" in tool_name:
                answer_parts.append(f"**Billing Data:**\n{str(result)}")
            elif "psa" in tool_name:
                answer_parts.append(f"**Service Tickets:**\n{str(result)}")
            elif "crm" in tool_name:
                answer_parts.append(f"**CRM Profile:**\n{str(result)}")
            elif "contract" in tool_name:
                answer_parts.append(f"**Contract Terms:**\n{str(result)}")
            else:
                answer_parts.append(str(result))

        if answer_parts:
            return "\n\n".join(answer_parts)
        else:
            return "No results found from tools."

    async def compose(
        self,
        query: str,
        ctx=None,
    ) -> MCPCompositionResult:
        """
        Dynamically compose MCP tools to answer a query.

        Args:
            query: User query
            ctx: ABACContext for security/audit (optional)

        Returns:
            MCPCompositionResult with answer + tool metadata
        """
        start = time.time()

        # Load tools if needed
        await self._load_tools_if_needed()

        # Step 1: Plan tools
        tools = self._plan_tools(query)

        if not tools:
            # No tools matched; return helpful message
            return MCPCompositionResult(
                answer="No MCP tools matched your query. Try asking about: "
                       "invoices, contracts, service tickets, CRM profiles, or "
                       "cross-source client information.",
                tools_used=[],
                tool_results={},
                tool_latencies={},
                total_latency_ms=(time.time() - start) * 1000,
                cost_usd=0.0,
                data_sources=[],
            )

        # Step 2: Execute tools in parallel
        tool_results, tool_latencies = await self._execute_tools(tools, query, ctx)

        # Step 3: Synthesize results
        answer = self._synthesize(tool_results, query)

        # Extract data sources from results
        data_sources = list(tool_results.keys())

        # Estimate cost (simple: $0.001 per tool call)
        cost_usd = len(tool_results) * 0.001

        total_latency = (time.time() - start) * 1000

        return MCPCompositionResult(
            answer=answer,
            tools_used=tools,
            tool_results=tool_results,
            tool_latencies=tool_latencies,
            total_latency_ms=total_latency,
            cost_usd=cost_usd,
            data_sources=data_sources,
        )
