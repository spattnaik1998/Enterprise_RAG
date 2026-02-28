"""
Enterprise RAG - MCP Collection Server
---------------------------------------
A Model Context Protocol server that exposes data-collection tools.
Any MCP-compatible client (Claude Desktop, custom clients) can connect
to this server and invoke tools to pull from ArXiv, Wikipedia, RSS, the web,
or the five enterprise MSP back-office systems.

Original sources: ArXiv, Wikipedia, RSS, Web
Enterprise tools: Billing, PSA, CRM, Communications, Contracts, Client 360

Run standalone:
    python -m src.collection.mcp.server
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import arxiv
import feedparser
import httpx
import wikipediaapi
from loguru import logger
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool
import mcp.server.stdio

# --- Shared SDK Clients -------------------------------------------------------

_WIKI = wikipediaapi.Wikipedia(
    language="en",
    user_agent="EnterpriseRAG-MCP/1.0",
)
_ARXIV = arxiv.Client(page_size=25, delay_seconds=3.0, num_retries=3)
_HEADERS = {"User-Agent": "EnterpriseRAG-MCP/1.0"}

# --- Enterprise data paths ----------------------------------------------------

_DATA_DIR = Path("data/enterprise")


def _load_enterprise(filename: str) -> list:
    """Load a JSON file from data/enterprise/ and return its list contents."""
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Enterprise data file not found: {path}. "
            "Run: python scripts/generate_enterprise_data.py"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else raw.get("records", [])


# --- MCP Server ---------------------------------------------------------------

server = Server("enterprise-rag-collector")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # -- Original research tools ------------------------------------------
        Tool(
            name="search_arxiv",
            description=(
                "Search ArXiv for research papers by keyword query. "
                "Returns paper titles, abstracts, authors, publication dates, "
                "categories, and entry URLs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'retrieval augmented generation')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum papers to return (1-50, default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="fetch_wikipedia",
            description=(
                "Fetch a Wikipedia article by its exact title. "
                "Returns the article summary, section headings, and full text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Exact Wikipedia article title (e.g. 'FAISS')",
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="fetch_rss_feed",
            description=(
                "Download and parse an RSS or Atom feed URL. "
                "Returns article titles, summaries, links, and publication dates."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL of the RSS/Atom feed",
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum items to return (default 20)",
                        "default": 20,
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="fetch_webpage",
            description=(
                "Fetch raw text content from a given webpage URL. "
                "Returns the response body (first 10,000 characters)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full HTTP/HTTPS URL of the page to fetch",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="list_available_sources",
            description="List all data source types supported by this MCP server.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # -- Enterprise Billing Tools (QuickBooks) ----------------------------
        Tool(
            name="billing_get_overdue_invoices",
            description=(
                "Return all invoices that are overdue by at least N days. "
                "Useful for AR aging reports and collection workflow triggers."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "days_threshold": {
                        "type": "integer",
                        "description": "Minimum days past due to include (default 60)",
                        "default": 60,
                    },
                    "include_critical": {
                        "type": "boolean",
                        "description": "Include OVERDUE_CRITICAL (90+ days) records (default true)",
                        "default": True,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="billing_get_aged_receivables",
            description=(
                "Return an accounts receivable aging summary bucketed by "
                "0-30, 31-60, 61-90, and 90+ days. Includes totals per bucket "
                "and per client."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="billing_get_client_statement",
            description=(
                "Return all invoices for a specific client with full "
                "payment history and current AR balance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Client ID (e.g. 'CLIENT-001')",
                    },
                },
                "required": ["client_id"],
            },
        ),
        Tool(
            name="billing_get_invoice_details",
            description="Return the full record for a single invoice by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "Invoice ID (e.g. 'INV-2025-0001')",
                    },
                },
                "required": ["invoice_id"],
            },
        ),
        # -- Enterprise PSA Tools (ConnectWise) -------------------------------
        Tool(
            name="psa_get_client_tickets",
            description=(
                "Return service tickets for a specific client. "
                "Optionally filter by status (OPEN, CLOSED, IN_PROGRESS, or 'all')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Client ID (e.g. 'CLIENT-001')",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by status: 'OPEN', 'CLOSED', 'IN_PROGRESS', or 'all' (default 'all')",
                        "default": "all",
                    },
                },
                "required": ["client_id"],
            },
        ),
        Tool(
            name="psa_get_unbilled_work",
            description=(
                "Return all service tickets with billable hours that have "
                "not yet been invoiced, grouped by client."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Filter to a single client ID (optional)",
                    },
                },
                "required": [],
            },
        ),
        # -- Enterprise CRM Tools (HubSpot) -----------------------------------
        Tool(
            name="crm_get_client_profile",
            description=(
                "Return the full CRM profile for a client: account health, "
                "NPS, financial summary, upsell opportunities, and key contacts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Client ID (e.g. 'CLIENT-001')",
                    },
                },
                "required": ["client_id"],
            },
        ),
        Tool(
            name="crm_get_at_risk_accounts",
            description=(
                "Return all client accounts flagged as at-risk: "
                "health score below 50, churn risk HIGH or CRITICAL, "
                "or NPS score below 7 (Detractor)."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        # -- Enterprise Communications Tools (Exchange) -----------------------
        Tool(
            name="comms_get_invoice_history",
            description=(
                "Return the full communication history (reminders sent, "
                "client responses) for a specific invoice."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "Invoice ID (e.g. 'INV-2025-0001')",
                    },
                },
                "required": ["invoice_id"],
            },
        ),
        # -- Enterprise Contracts Tools (SharePoint) --------------------------
        Tool(
            name="contracts_get_terms",
            description=(
                "Return the service agreement for a specific client: "
                "SLA commitments, payment terms, late fee clauses, "
                "and services included."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Client ID (e.g. 'CLIENT-001')",
                    },
                },
                "required": ["client_id"],
            },
        ),
        # -- Cross-Source Aggregation (Enterprise RAG showpiece) ---------------
        Tool(
            name="get_client_360",
            description=(
                "Aggregate a complete 360-degree view of a client by "
                "joining data from ALL five enterprise source systems: "
                "Billing (QuickBooks), PSA (ConnectWise), CRM (HubSpot), "
                "Communications (Exchange), and Contracts (SharePoint). "
                "This is the cornerstone enterprise RAG tool - a single "
                "query that would require five separate system logins "
                "without this pipeline."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Client ID (e.g. 'CLIENT-001')",
                    },
                },
                "required": ["client_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        match name:
            # Research tools
            case "search_arxiv":
                return await _search_arxiv(arguments)
            case "fetch_wikipedia":
                return await _fetch_wikipedia(arguments)
            case "fetch_rss_feed":
                return await _fetch_rss_feed(arguments)
            case "fetch_webpage":
                return await _fetch_webpage(arguments)
            case "list_available_sources":
                return await _list_sources()
            # Billing tools
            case "billing_get_overdue_invoices":
                return await _billing_overdue(arguments)
            case "billing_get_aged_receivables":
                return await _billing_aged_ar()
            case "billing_get_client_statement":
                return await _billing_client_statement(arguments)
            case "billing_get_invoice_details":
                return await _billing_invoice_details(arguments)
            # PSA tools
            case "psa_get_client_tickets":
                return await _psa_client_tickets(arguments)
            case "psa_get_unbilled_work":
                return await _psa_unbilled_work(arguments)
            # CRM tools
            case "crm_get_client_profile":
                return await _crm_client_profile(arguments)
            case "crm_get_at_risk_accounts":
                return await _crm_at_risk()
            # Communications tools
            case "comms_get_invoice_history":
                return await _comms_invoice_history(arguments)
            # Contracts tools
            case "contracts_get_terms":
                return await _contracts_terms(arguments)
            # Cross-source
            case "get_client_360":
                return await _client_360(arguments)
            case _:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as exc:
        logger.error(f"[MCP] Tool error [{name}]: {exc}")
        return [TextContent(type="text", text=f"Error in {name}: {exc}")]


# --- Original Tool Implementations --------------------------------------------

async def _search_arxiv(args: dict) -> list[TextContent]:
    query = args["query"]
    max_results = min(int(args.get("max_results", 10)), 50)
    loop = asyncio.get_event_loop()

    def _sync():
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        return list(_ARXIV.results(search))

    papers = await loop.run_in_executor(None, _sync)
    results = [
        {
            "arxiv_id": p.entry_id,
            "title": p.title,
            "abstract": p.summary[:600] + ("..." if len(p.summary) > 600 else ""),
            "authors": [str(a) for a in p.authors[:6]],
            "published": p.published.isoformat() if p.published else None,
            "categories": p.categories,
            "url": p.entry_id,
            "pdf_url": str(p.pdf_url) if p.pdf_url else None,
        }
        for p in papers
    ]
    return [TextContent(type="text", text=json.dumps(results, indent=2, default=str))]


async def _fetch_wikipedia(args: dict) -> list[TextContent]:
    topic = args["topic"]
    loop = asyncio.get_event_loop()

    def _sync():
        page = _WIKI.page(topic)
        if not page.exists():
            return {"error": f"Wikipedia page not found: '{topic}'"}
        return {
            "title": page.title,
            "url": page.fullurl,
            "summary": page.summary[:1500],
            "char_count": len(page.text),
            "sections": [s.title for s in page.sections[:12]],
            "content_preview": page.text[:6000] + ("..." if len(page.text) > 6000 else ""),
        }

    result = await loop.run_in_executor(None, _sync)
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _fetch_rss_feed(args: dict) -> list[TextContent]:
    url = args["url"]
    max_items = min(int(args.get("max_items", 20)), 100)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()
        raw = resp.text

    loop = asyncio.get_event_loop()
    feed = await loop.run_in_executor(None, lambda: feedparser.parse(raw))

    def _parse_date(t):
        if t is None:
            return None
        try:
            return datetime.fromtimestamp(time.mktime(t), tz=timezone.utc).isoformat()
        except Exception:
            return None

    items = [
        {
            "title": e.get("title", ""),
            "summary": e.get("summary", "")[:400],
            "url": e.get("link", ""),
            "published": _parse_date(e.get("published_parsed")),
        }
        for e in feed.entries[:max_items]
    ]
    result = {
        "feed_title": feed.feed.get("title", url),
        "feed_url": url,
        "total_entries": len(feed.entries),
        "returned": len(items),
        "items": items,
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _fetch_webpage(args: dict) -> list[TextContent]:
    url = args["url"]
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()
        content = resp.text[:10_000]
    result = {"url": url, "content_length": len(resp.text), "content": content}
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _list_sources() -> list[TextContent]:
    sources = {
        "server": "enterprise-rag-collector",
        "version": "2.0.0",
        "available_sources": [
            {"name": "arxiv",          "tool": "search_arxiv",                 "description": "Research papers from ArXiv.org"},
            {"name": "wikipedia",      "tool": "fetch_wikipedia",              "description": "English Wikipedia articles"},
            {"name": "rss",            "tool": "fetch_rss_feed",               "description": "RSS/Atom news and blog feeds"},
            {"name": "web",            "tool": "fetch_webpage",                "description": "Generic HTTP web pages"},
            {"name": "billing",        "tool": "billing_get_overdue_invoices", "description": "QuickBooks-style invoice AR system"},
            {"name": "psa",            "tool": "psa_get_client_tickets",       "description": "ConnectWise-style PSA service tickets"},
            {"name": "crm",            "tool": "crm_get_client_profile",       "description": "HubSpot-style CRM client profiles"},
            {"name": "communications", "tool": "comms_get_invoice_history",    "description": "Exchange Online email/reminder log"},
            {"name": "contracts",      "tool": "contracts_get_terms",          "description": "SharePoint service agreement repository"},
            {"name": "cross_source",   "tool": "get_client_360",               "description": "Multi-system 360 client view (enterprise RAG)"},
        ],
    }
    return [TextContent(type="text", text=json.dumps(sources, indent=2))]


# --- Enterprise Billing Tool Implementations ----------------------------------

async def _billing_overdue(args: dict) -> list[TextContent]:
    threshold = int(args.get("days_threshold", 60))
    include_critical = bool(args.get("include_critical", True))
    invoices = _load_enterprise("invoices.json")

    target_statuses = {"OVERDUE", "PAST_DUE"}
    if include_critical:
        target_statuses.add("OVERDUE_CRITICAL")

    overdue = [
        inv for inv in invoices
        if inv["status"] in target_statuses
        and inv["days_outstanding"] >= threshold
    ]
    overdue.sort(key=lambda x: x["days_outstanding"], reverse=True)

    summary = {
        "query": {
            "days_threshold": threshold,
            "include_critical": include_critical,
        },
        "total_count": len(overdue),
        "total_balance_due": round(sum(i["balance_due"] for i in overdue), 2),
        "invoices": [
            {
                "invoice_id":      i["invoice_id"],
                "client_name":     i["client_name"],
                "client_industry": i.get("client_industry", ""),
                "status":          i["status"],
                "total_amount":    i["total_amount"],
                "balance_due":     i["balance_due"],
                "days_outstanding": i["days_outstanding"],
                "due_date":        i["due_date"],
                "billing_contact": i.get("billing_contact", ""),
                "account_manager": i.get("account_manager", ""),
            }
            for i in overdue
        ],
    }
    return [TextContent(type="text", text=json.dumps(summary, indent=2))]


async def _billing_aged_ar() -> list[TextContent]:
    invoices = _load_enterprise("invoices.json")
    open_invoices = [
        i for i in invoices if i["status"] not in ("PAID",)
    ]

    buckets = {
        "current_0_30":   {"count": 0, "balance": 0.0, "invoices": []},
        "past_due_31_60": {"count": 0, "balance": 0.0, "invoices": []},
        "overdue_61_90":  {"count": 0, "balance": 0.0, "invoices": []},
        "critical_90_plus": {"count": 0, "balance": 0.0, "invoices": []},
    }

    for inv in open_invoices:
        days = inv["days_outstanding"]
        bal  = inv["balance_due"]
        entry = {
            "invoice_id":   inv["invoice_id"],
            "client_name":  inv["client_name"],
            "balance_due":  bal,
            "days_outstanding": days,
        }
        if days <= 30:
            b = "current_0_30"
        elif days <= 60:
            b = "past_due_31_60"
        elif days <= 90:
            b = "overdue_61_90"
        else:
            b = "critical_90_plus"
        buckets[b]["count"] += 1
        buckets[b]["balance"] = round(buckets[b]["balance"] + bal, 2)
        buckets[b]["invoices"].append(entry)

    # Sort each bucket by days descending
    for b in buckets.values():
        b["invoices"].sort(key=lambda x: x["days_outstanding"], reverse=True)

    total_ar = round(sum(i["balance_due"] for i in open_invoices), 2)
    result = {
        "report_date": "2026-02-28",
        "total_open_ar": total_ar,
        "aging_buckets": buckets,
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _billing_client_statement(args: dict) -> list[TextContent]:
    client_id = args["client_id"]
    invoices = _load_enterprise("invoices.json")
    client_inv = [i for i in invoices if i["client_id"] == client_id]
    client_inv.sort(key=lambda x: x["issue_date"])

    if not client_inv:
        return [TextContent(type="text", text=json.dumps({"error": f"No invoices found for client_id: {client_id}"}))]

    total_billed = sum(i["total_amount"] for i in client_inv)
    total_paid   = sum(i["total_amount"] - i["balance_due"] for i in client_inv)
    open_balance = sum(i["balance_due"] for i in client_inv)

    result = {
        "client_id":    client_id,
        "client_name":  client_inv[0]["client_name"],
        "statement_date": "2026-02-28",
        "summary": {
            "total_invoices":  len(client_inv),
            "total_billed":    round(total_billed, 2),
            "total_paid":      round(total_paid, 2),
            "open_balance":    round(open_balance, 2),
        },
        "invoices": client_inv,
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _billing_invoice_details(args: dict) -> list[TextContent]:
    invoice_id = args["invoice_id"]
    invoices = _load_enterprise("invoices.json")
    match = next((i for i in invoices if i["invoice_id"] == invoice_id), None)
    if not match:
        return [TextContent(type="text", text=json.dumps({"error": f"Invoice not found: {invoice_id}"}))]
    return [TextContent(type="text", text=json.dumps(match, indent=2, default=str))]


# --- Enterprise PSA Tool Implementations --------------------------------------

async def _psa_client_tickets(args: dict) -> list[TextContent]:
    client_id = args["client_id"]
    status_filter = args.get("status", "all").upper()
    tickets = _load_enterprise("psa_tickets.json")
    client_tickets = [t for t in tickets if t["client_id"] == client_id]

    if status_filter != "ALL":
        client_tickets = [t for t in client_tickets if t["status"] == status_filter]

    client_tickets.sort(key=lambda x: x.get("created_date", ""), reverse=True)

    result = {
        "client_id":    client_id,
        "status_filter": status_filter,
        "total_tickets": len(client_tickets),
        "tickets": client_tickets,
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _psa_unbilled_work(args: dict) -> list[TextContent]:
    client_id = args.get("client_id")
    tickets = _load_enterprise("psa_tickets.json")

    # Tickets that have billable hours and are not yet invoiced
    unbilled = [
        t for t in tickets
        if t.get("billable_hours", 0) > 0 and not t.get("invoiced", False)
    ]

    if client_id:
        unbilled = [t for t in unbilled if t["client_id"] == client_id]

    # Group by client
    by_client: dict[str, dict] = {}
    for t in unbilled:
        cid = t["client_id"]
        if cid not in by_client:
            by_client[cid] = {
                "client_id":   cid,
                "client_name": t["client_name"],
                "ticket_count": 0,
                "total_billable_hours": 0.0,
                "total_billable_amount": 0.0,
                "tickets": [],
            }
        by_client[cid]["ticket_count"] += 1
        by_client[cid]["total_billable_hours"] = round(
            by_client[cid]["total_billable_hours"] + t.get("billable_hours", 0), 2
        )
        by_client[cid]["total_billable_amount"] = round(
            by_client[cid]["total_billable_amount"] + t.get("billable_amount", 0), 2
        )
        by_client[cid]["tickets"].append({
            "ticket_id":      t["ticket_id"],
            "summary":        t["summary"],
            "billable_hours": t.get("billable_hours", 0),
            "billable_amount": t.get("billable_amount", 0),
        })

    result = {
        "total_unbilled_tickets": len(unbilled),
        "total_unbilled_amount":  round(sum(t.get("billable_amount", 0) for t in unbilled), 2),
        "by_client": list(by_client.values()),
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# --- Enterprise CRM Tool Implementations --------------------------------------

async def _crm_client_profile(args: dict) -> list[TextContent]:
    client_id = args["client_id"]
    profiles = _load_enterprise("crm_profiles.json")
    match = next((p for p in profiles if p["client_id"] == client_id), None)
    if not match:
        return [TextContent(type="text", text=json.dumps({"error": f"CRM profile not found: {client_id}"}))]
    return [TextContent(type="text", text=json.dumps(match, indent=2, default=str))]


async def _crm_at_risk() -> list[TextContent]:
    profiles = _load_enterprise("crm_profiles.json")
    at_risk = [
        p for p in profiles
        if (
            p.get("health_score", 100) < 50
            or p.get("churn_risk") in ("HIGH", "CRITICAL")
            or p.get("nps_score", 10) < 7
        )
    ]
    at_risk.sort(key=lambda x: x.get("health_score", 100))

    result = {
        "report_date":    "2026-02-28",
        "total_at_risk":  len(at_risk),
        "accounts": [
            {
                "client_id":    p["client_id"],
                "client_name":  p["client_name"],
                "industry":     p.get("industry", ""),
                "health_score": p.get("health_score", 0),
                "health_trend": p.get("health_trend", ""),
                "churn_risk":   p.get("churn_risk", ""),
                "nps_score":    p.get("nps_score", 0),
                "open_balance": p.get("open_balance", 0),
                "account_manager": p.get("account_manager", ""),
            }
            for p in at_risk
        ],
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# --- Enterprise Communications Tool Implementations ---------------------------

async def _comms_invoice_history(args: dict) -> list[TextContent]:
    invoice_id = args["invoice_id"]
    comms = _load_enterprise("communications.json")
    history = sorted(
        [c for c in comms if c["invoice_id"] == invoice_id],
        key=lambda x: x.get("sent_date", ""),
    )

    if not history:
        return [TextContent(type="text", text=json.dumps(
            {"invoice_id": invoice_id, "notice_count": 0, "message": "No communications found for this invoice."}
        ))]

    result = {
        "invoice_id":    invoice_id,
        "client_name":   history[0]["client_name"],
        "notice_count":  len(history),
        "last_sent":     history[-1]["sent_date"],
        "last_response": history[-1].get("client_response", "NO_RESPONSE"),
        "history":       history,
    }
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# --- Enterprise Contracts Tool Implementations --------------------------------

async def _contracts_terms(args: dict) -> list[TextContent]:
    client_id = args["client_id"]
    contracts = _load_enterprise("contracts.json")
    match = next((c for c in contracts if c["client_id"] == client_id), None)
    if not match:
        return [TextContent(type="text", text=json.dumps({"error": f"Contract not found for client: {client_id}"}))]
    return [TextContent(type="text", text=json.dumps(match, indent=2, default=str))]


# --- Cross-Source Client 360 (Enterprise RAG showpiece) ----------------------

async def _client_360(args: dict) -> list[TextContent]:
    """
    Aggregate a complete 360-degree client view by joining all 5 source systems.
    This is the core enterprise RAG demonstration: a single tool call replaces
    five separate system logins and manual data reconciliation.
    """
    client_id = args["client_id"]

    # Load all source systems concurrently
    loop = asyncio.get_event_loop()
    invoices  = await loop.run_in_executor(None, _load_enterprise, "invoices.json")
    tickets   = await loop.run_in_executor(None, _load_enterprise, "psa_tickets.json")
    profiles  = await loop.run_in_executor(None, _load_enterprise, "crm_profiles.json")
    comms     = await loop.run_in_executor(None, _load_enterprise, "communications.json")
    contracts = await loop.run_in_executor(None, _load_enterprise, "contracts.json")

    # Filter per source
    client_invoices  = [i for i in invoices  if i["client_id"] == client_id]
    client_tickets   = [t for t in tickets   if t["client_id"] == client_id]
    client_profile   = next((p for p in profiles  if p["client_id"] == client_id), {})
    client_comms     = [c for c in comms     if c["client_id"] == client_id]
    client_contract  = next((c for c in contracts if c["client_id"] == client_id), {})

    if not client_profile and not client_invoices:
        return [TextContent(type="text", text=json.dumps(
            {"error": f"No data found for client_id: {client_id}"}
        ))]

    # Compute billing summary
    open_invoices    = [i for i in client_invoices if i["status"] != "PAID"]
    overdue_invoices = [
        i for i in client_invoices
        if i["status"] in ("OVERDUE", "OVERDUE_CRITICAL", "PAST_DUE")
    ]
    total_ar         = round(sum(i["balance_due"] for i in client_invoices), 2)
    overdue_balance  = round(sum(i["balance_due"] for i in overdue_invoices), 2)

    # Compute PSA summary
    open_tickets     = [t for t in client_tickets if t["status"] == "OPEN"]
    sla_breaches     = [t for t in client_tickets if not t.get("sla_met", True)]
    unbilled_hours   = sum(t.get("billable_hours", 0) for t in client_tickets if not t.get("invoiced", False))

    # Communication summary
    notice_count     = len(client_comms)
    last_response    = client_comms[-1].get("client_response", "N/A") if client_comms else "N/A"

    view = {
        "client_360_view": {
            "client_id":   client_id,
            "generated_at": "2026-02-28T00:00:00",
            "data_sources": ["QuickBooks Enterprise", "ConnectWise Manage",
                             "HubSpot CRM", "Exchange Online", "SharePoint"],
        },
        "identity": {
            "client_name":  client_profile.get("client_name", "Unknown"),
            "industry":     client_profile.get("industry", ""),
            "service_tier": client_profile.get("service_tier", ""),
            "client_since": client_profile.get("client_since", ""),
            "account_manager": client_profile.get("account_manager", ""),
        },
        "account_health": {
            "health_score":    client_profile.get("health_score", 0),
            "health_trend":    client_profile.get("health_trend", ""),
            "churn_risk":      client_profile.get("churn_risk", ""),
            "nps_score":       client_profile.get("nps_score", 0),
            "payment_reliability": client_profile.get("payment_reliability", ""),
        },
        "billing_summary": {
            "total_invoices":     len(client_invoices),
            "open_invoices":      len(open_invoices),
            "overdue_invoices":   len(overdue_invoices),
            "total_ar_balance":   total_ar,
            "overdue_balance":    overdue_balance,
            "overdue_details":    [
                {
                    "invoice_id":    i["invoice_id"],
                    "amount":        i["total_amount"],
                    "balance":       i["balance_due"],
                    "days_outstanding": i["days_outstanding"],
                    "status":        i["status"],
                    "due_date":      i["due_date"],
                }
                for i in sorted(overdue_invoices, key=lambda x: x["days_outstanding"], reverse=True)
            ],
        },
        "service_delivery": {
            "total_tickets":      len(client_tickets),
            "open_tickets":       len(open_tickets),
            "sla_breaches":       len(sla_breaches),
            "unbilled_hours":     round(unbilled_hours, 2),
        },
        "collection_history": {
            "total_reminders_sent": notice_count,
            "last_reminder_response": last_response,
            "latest_notices": [
                {
                    "comm_id":    c["comm_id"],
                    "sent_date":  c["sent_date"],
                    "notice_seq": c["reminder_sequence"],
                    "response":   c.get("client_response", "NO_RESPONSE"),
                }
                for c in sorted(client_comms, key=lambda x: x.get("sent_date", ""), reverse=True)[:5]
            ],
        },
        "contract_summary": {
            "contract_id":    client_contract.get("contract_id", ""),
            "contract_type":  client_contract.get("contract_type", ""),
            "status":         client_contract.get("status", ""),
            "mrr":            client_contract.get("monthly_recurring_revenue", 0),
            "end_date":       client_contract.get("end_date", ""),
            "payment_terms":  client_contract.get("payment_terms", ""),
            "late_fee_clause": client_contract.get("late_fee_clause", ""),
            "sla_terms":      client_contract.get("sla_terms", {}),
        },
        "upsell_pipeline": client_profile.get("upsell_opportunities", []),
    }

    return [TextContent(type="text", text=json.dumps(view, indent=2, default=str))]


# --- Entry Point --------------------------------------------------------------

async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="enterprise-rag-collector",
                server_version="2.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
