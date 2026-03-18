"""
Red Key Sandbox Enterprise RAG - Web API Server
------------------------------------------
FastAPI server that wraps the Phase III RAGPipeline and serves the chat UI
plus the Client Portal with JWT-based authentication.

Public routes (no auth required):
  GET  /                          -> serve app/static/landing.html (dual login)
  GET  /rag                       -> serve app/static/index.html        (RAG chat -- MSP)
  GET  /agent                     -> serve app/static/agent_chat.html  (Council chat -- MSP)
  GET  /forecast                  -> serve app/static/forecast.html
  GET  /logs                      -> serve app/static/logs.html
  GET  /msp                       -> serve app/static/msp_portal.html
  GET  /client                    -> serve app/static/client_portal.html
  GET  /engineer                  -> serve app/static/engineer_portal.html

Auth endpoints (no token needed):
  POST /api/auth/login            -> {token, role, client_id, client_name, username}

MSP-only endpoints (Bearer token, role=msp):
  GET  /api/health                -> pipeline status + model info
  POST /api/chat                  -> full RAG query
  GET  /api/clients               -> client list (forecasting)
  GET  /api/forecast/{client_id}  -> TimeFM revenue forecast
  GET  /api/logs                  -> paginated chat history
  GET  /api/logs/stats            -> chat statistics
  GET  /api/msp/tickets           -> all service tickets (filterable)
  GET  /api/msp/tickets/{id}      -> single ticket detail
  PATCH /api/msp/tickets/{id}     -> update ticket (status, assignee, notes)
  GET  /api/msp/ticket-stats      -> ticket counts by status
  GET  /api/msp/engineers         -> engineer profiles list
  GET  /api/msp/clients/credentials -> all client credentials (passwords hidden)

Client-only endpoints (Bearer token, role=client):
  GET  /api/portal/tickets        -> client's own tickets
  POST /api/portal/tickets        -> create a new service ticket

Engineer-only endpoints (Bearer token, role=engineer):
  GET  /api/engineer/tickets      -> engineer's assigned tickets
  GET  /api/engineer/tickets/{id} -> single ticket assigned to engineer
  PATCH /api/engineer/tickets/{id}-> update ticket (status, notes -- engineer only)
  GET  /api/engineer/stats        -> ticket counts by status for engineer

Run from the project root (Enterprise_RAG/):
    uvicorn app.server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Literal, Optional

# Windows cp1252 terminal fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env BEFORE any local imports so JWT_SECRET, SUPABASE_* etc. are
# in os.environ when app.auth / app.portal_db read them at module level.
from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, field_validator, model_validator
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.rate_limiter import limiter

from app.auth import (
    create_token,
    get_current_user,
    require_client,
    require_engineer,
    require_msp,
    verify_password,
)
from app.chat_logger import compute_stats, load_logs, log_interaction, update_feedback
from app.portal_db import (
    create_ticket,
    get_all_client_credentials,
    get_all_engineers,
    get_all_tickets,
    get_engineer_stats,
    get_engineer_ticket_by_id,
    get_engineer_tickets,
    get_ticket_by_id,
    get_ticket_stats,
    get_tickets_for_client,
    get_user_by_username,
    update_engineer_ticket,
    update_last_login,
    update_ticket,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
INDEX_DIR  = "data/index"

# Allowed model identifiers and their providers
_OPENAI_MODELS    = {"gpt-4o-mini", "gpt-4o"}
_ANTHROPIC_MODELS = {"claude-haiku-4-5-20251001", "claude-sonnet-4-6"}
_ALL_MODELS       = _OPENAI_MODELS | _ANTHROPIC_MODELS

# ---------------------------------------------------------------------------
# Pipeline singleton + forecaster singleton
# ---------------------------------------------------------------------------

_pipeline = None
_forecaster: Optional["InvoiceForecaster"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once at startup; clean up on shutdown."""
    global _pipeline
    try:
        from src.embedding.supabase_index import SupabaseIndex
        from src.serving.pipeline import RAGPipeline
        from src.utils.logger import setup_logger
        from src.proactive.alert_store import AlertStore
        from src.proactive.scheduler import ProactiveScheduler

        setup_logger()
        logger.info("[Server] Connecting to Supabase index...")
        supabase_index = SupabaseIndex()
        vector_count = supabase_index.ntotal
        logger.info(f"[Server] Supabase index ready | {vector_count:,} vectors")

        _pipeline = RAGPipeline(index=supabase_index)
        logger.info(
            f"[Server] Pipeline ready | "
            f"{vector_count:,} vectors | "
            f"default model={_pipeline.generator.model}"
        )

        # Initialize AlertStore with Supabase client
        alert_store = AlertStore(supabase_client=supabase_index._sb)
        app.state.alert_store = alert_store
        logger.info("[Server] AlertStore initialized")

        # Initialize and start ProactiveScheduler
        scheduler = ProactiveScheduler()
        await scheduler.start(alert_store)
        app.state.scheduler = scheduler

    except Exception as exc:
        logger.error(f"[Server] Startup failed: {exc}")
        raise
    yield
    # Shutdown scheduler
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.stop()
        logger.info("[Server] Scheduler stopped")

    _pipeline = None
    logger.info("[Server] Pipeline unloaded.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Red Key Sandbox Enterprise RAG API",
    description="Retrieval-Augmented Generation over MSP operations and AI research",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    session_id: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in _ALL_MODELS:
            raise ValueError(
                f"Unknown model '{v}'. "
                f"Allowed: {sorted(_ALL_MODELS)}"
            )
        return v

    @model_validator(mode="after")
    def validate_provider_model_match(self) -> "ChatRequest":
        if self.provider == "openai" and self.model in _ANTHROPIC_MODELS:
            raise ValueError(f"Model '{self.model}' is not an OpenAI model.")
        if self.provider == "anthropic" and self.model in _OPENAI_MODELS:
            raise ValueError(f"Model '{self.model}' is not an Anthropic model.")
        return self


class CitationModel(BaseModel):
    index: int
    title: str
    source: str
    source_type: str
    chunk_index: int
    relevance_score: float
    url: Optional[str] = None


class LatencyModel(BaseModel):
    retrieval: float
    rerank: float
    generation: float
    total: float


class TokenModel(BaseModel):
    prompt: int
    completion: int
    total: int


class ContextPieceModel(BaseModel):
    id: str
    tier: str
    source_type: str
    relevance_score: float
    freshness_score: float
    tokens: int
    included: bool
    context_position: int


class ContextBundleModel(BaseModel):
    total_tokens: int
    budget_tokens: int
    truncated: bool
    dropped_count: int
    strategy_used: str
    fast_path: bool
    pieces: list[ContextPieceModel]


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    blocked: bool
    blocked_reason: str
    latency_ms: LatencyModel
    tokens: TokenModel
    estimated_cost_usd: float
    pii_redacted: list[str]
    model: str
    provider: str
    context_bundle: Optional[ContextBundleModel] = None


# ---------------------------------------------------------------------------
# Proactive alert models
# ---------------------------------------------------------------------------

class AlertModel(BaseModel):
    """Alert response model."""
    id: str
    alert_type: str
    severity: str
    payload: dict
    acknowledged: bool
    created_at: str
    acknowledged_at: Optional[str] = None


class AlertListResponse(BaseModel):
    """Paginated alerts response."""
    alerts: list[AlertModel]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Portal auth models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    role: str
    username: str
    client_id: str | None = None
    client_name: str | None = None


# ---------------------------------------------------------------------------
# Portal ticket models
# ---------------------------------------------------------------------------

VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_CATEGORIES = {"network", "hardware", "software", "security", "email", "cloud", "backup", "other"}
VALID_STATUSES   = {"open", "in_progress", "waiting_client", "resolved", "closed"}


class CreateTicketRequest(BaseModel):
    title: str
    description: str
    priority: str = "medium"
    category: str = "other"

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        if v not in VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(VALID_PRIORITIES)}")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}")
        return v


class UpdateTicketRequest(BaseModel):
    status: str | None = None
    assigned_to: str | None = None
    engineer_notes: str | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        return v


class FeedbackRequest(BaseModel):
    rating: int
    feedback: str = ""

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("rating must be between 1 and 5")
        return v


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------

def _make_generator(provider: str, model: str):
    """Instantiate the correct generator class for the given provider/model."""
    if provider == "anthropic":
        from src.generation.generator import AnthropicGenerator
        return AnthropicGenerator(model=model)
    from src.generation.generator import RAGGenerator
    return RAGGenerator(model=model)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_landing():
    """Serve the dual-login landing page."""
    path = STATIC_DIR / "landing.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="landing.html not found")
    return FileResponse(str(path), media_type="text/html")


@app.get("/rag", include_in_schema=False)
async def serve_rag_ui():
    """Serve the RAG chat UI (MSP only -- guarded client-side)."""
    path = STATIC_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(path), media_type="text/html")


@app.get("/agent", include_in_schema=False)
async def serve_agent_chat():
    """Serve the Multi-Agent Council chat UI (MSP only -- guarded client-side)."""
    path = STATIC_DIR / "agent_chat.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="agent_chat.html not found")
    return FileResponse(str(path), media_type="text/html")


@app.get("/msp", include_in_schema=False)
async def serve_msp_portal():
    """Serve the MSP portal hub."""
    path = STATIC_DIR / "msp_portal.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="msp_portal.html not found")
    return FileResponse(str(path), media_type="text/html")


@app.get("/client", include_in_schema=False)
async def serve_client_portal():
    """Serve the client portal (ticket creation + status)."""
    path = STATIC_DIR / "client_portal.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="client_portal.html not found")
    return FileResponse(str(path), media_type="text/html")


@app.get("/portal", include_in_schema=False)
async def serve_portal_by_client(client_id: str = ""):
    """Serve the client portal — linked from Manage Clients 'View Portal' button."""
    path = STATIC_DIR / "client_portal.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="client_portal.html not found")
    return FileResponse(str(path), media_type="text/html")


@app.get("/engineer", include_in_schema=False)
async def serve_engineer_portal():
    """Serve the engineer portal (view and update assigned tickets)."""
    path = STATIC_DIR / "engineer_portal.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="engineer_portal.html not found")
    return FileResponse(str(path), media_type="text/html")


@app.get("/observability", include_in_schema=False)
async def serve_observability_dashboard():
    """Serve the Agent Observability Dashboard."""
    path = STATIC_DIR / "observability.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="observability.html not found")
    return FileResponse(str(path), media_type="text/html")


@app.get("/forecast", include_in_schema=False)
async def serve_forecast_ui():
    forecast_path = STATIC_DIR / "forecast.html"
    if not forecast_path.exists():
        raise HTTPException(status_code=404, detail="Forecast UI not found at app/static/forecast.html")
    return FileResponse(str(forecast_path), media_type="text/html")


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate a portal user (MSP admin or client).
    Returns a signed JWT on success.
    """
    user = get_user_by_username(request.username.strip().lower())
    if user is None or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_token(
        username=user["username"],
        role=user["role"],
        client_id=user.get("client_id"),
        client_name=user.get("client_name"),
    )
    update_last_login(user["username"])
    logger.info(f"[Auth] Login: username={user['username']} role={user['role']}")
    return LoginResponse(
        token=token,
        role=user["role"],
        username=user["username"],
        client_id=user.get("client_id"),
        client_name=user.get("client_name"),
    )


# ---------------------------------------------------------------------------
# Client portal ticket routes
# ---------------------------------------------------------------------------

@app.get("/api/portal/tickets")
async def portal_list_tickets(user: dict = Depends(require_client)):
    """Return all tickets belonging to the authenticated client."""
    client_id = user["client_id"]
    tickets = get_tickets_for_client(client_id)
    return {"tickets": tickets, "count": len(tickets)}


@app.post("/api/portal/tickets", status_code=201)
async def portal_create_ticket(
    request: CreateTicketRequest,
    user: dict = Depends(require_client),
):
    """Create a new service ticket for the authenticated client."""
    try:
        ticket = create_ticket(
            client_id=user["client_id"],
            client_name=user["client_name"],
            title=request.title.strip(),
            description=request.description.strip(),
            priority=request.priority,
            category=request.category,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    logger.info(
        f"[Portal] Ticket created | client={user['client_id']} | "
        f"ticket={ticket.get('ticket_number')}"
    )
    return ticket


# ---------------------------------------------------------------------------
# Engineer ticket management routes
# ---------------------------------------------------------------------------

@app.get("/api/engineer/tickets")
async def engineer_list_tickets(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(require_engineer),
):
    """Return all tickets assigned to this engineer (engineer only)."""
    engineer_name = user["sub"]
    tickets = get_engineer_tickets(engineer_name)

    # Client-side filtering by status if requested
    if status:
        tickets = [t for t in tickets if t.get("status") == status]

    # Paginate
    paginated = tickets[offset : offset + limit]
    return {"tickets": paginated, "count": len(tickets), "limit": limit, "offset": offset}


@app.get("/api/engineer/tickets/{ticket_id}")
async def engineer_get_ticket(ticket_id: str, user: dict = Depends(require_engineer)):
    """Return a single ticket assigned to this engineer."""
    engineer_name = user["sub"]
    ticket = get_engineer_ticket_by_id(engineer_name, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found or not assigned to you.")
    return ticket


@app.patch("/api/engineer/tickets/{ticket_id}")
async def engineer_update_ticket(
    ticket_id: str,
    request: UpdateTicketRequest,
    user: dict = Depends(require_engineer),
):
    """Update ticket status or notes (engineer can only update their own)."""
    engineer_name = user["sub"]
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")

    updated = update_engineer_ticket(engineer_name, ticket_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found or not assigned to you.")
    return updated


@app.get("/api/engineer/stats")
async def engineer_stats(user: dict = Depends(require_engineer)):
    """Return ticket stats for this engineer."""
    engineer_name = user["sub"]
    return get_engineer_stats(engineer_name)


# ---------------------------------------------------------------------------
# MSP ticket management routes
# ---------------------------------------------------------------------------

@app.get("/api/msp/tickets")
async def msp_list_tickets(
    status:    str | None = Query(default=None),
    client_id: str | None = Query(default=None),
    limit:     int        = Query(default=100, ge=1, le=500),
    offset:    int        = Query(default=0, ge=0),
    _user: dict = Depends(require_msp),
):
    """Return all service tickets with optional filters (MSP only)."""
    tickets = get_all_tickets(
        status_filter=status,
        client_id_filter=client_id,
        limit=limit,
        offset=offset,
    )
    return {"tickets": tickets, "count": len(tickets), "limit": limit, "offset": offset}


@app.get("/api/msp/tickets/{ticket_id}")
async def msp_get_ticket(ticket_id: str, _user: dict = Depends(require_msp)):
    """Return a single service ticket by ID (MSP only)."""
    ticket = get_ticket_by_id(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return ticket


@app.patch("/api/msp/tickets/{ticket_id}")
async def msp_update_ticket(
    ticket_id: str,
    request: UpdateTicketRequest,
    _user: dict = Depends(require_msp),
):
    """Update ticket status, assignee, or engineer notes (MSP only)."""
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")
    updated = update_ticket(ticket_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found or update failed.")
    return updated


@app.get("/api/msp/ticket-stats")
async def msp_ticket_stats(_user: dict = Depends(require_msp)):
    """Return ticket counts grouped by status (MSP dashboard)."""
    return get_ticket_stats()


@app.get("/api/msp/engineers")
async def msp_list_engineers(_user: dict = Depends(require_msp)):
    """Return all engineer profiles (MSP only)."""
    engineers = get_all_engineers()
    return {"engineers": engineers, "count": len(engineers)}


@app.get("/api/msp/clients/credentials")
async def msp_list_client_credentials(_user: dict = Depends(require_msp)):
    """Return all client portal credentials (MSP only). Passwords are hidden."""
    credentials = get_all_client_credentials()
    return {"clients": credentials, "count": len(credentials)}


@app.get("/api/health")
async def health(_user: dict = Depends(require_msp)):
    """Return pipeline status, index metadata, and available models."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return {
        "status": "ok",
        "vectors": _pipeline.index.ntotal,
        "default_model": _pipeline.generator.model,
        "reranking_enabled": _pipeline.enable_reranking,
        "rerank_top_k": _pipeline.rerank_top_k,
        "top_k": _pipeline.retriever.top_k,
        "available_models": {
            "openai": sorted(_OPENAI_MODELS),
            "anthropic": sorted(_ANTHROPIC_MODELS),
        },
    }


@app.get("/api/clients")
async def list_clients(_user: dict = Depends(require_msp)):
    """Return all client IDs and names from invoice data."""
    global _forecaster
    try:
        if _forecaster is None:
            from src.forecasting.invoice_forecaster import InvoiceForecaster
            _forecaster = InvoiceForecaster()
        return {"clients": _forecaster.get_clients()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"[API] /api/clients error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/forecast/{client_id}")
@limiter.limit("30/minute")
async def forecast_invoices(request: Request, client_id: str, horizon: int = 6, _user: dict = Depends(require_msp)):
    """
    Run TimeFM revenue forecast for a client.

    The TimeFM model (~800 MB) is downloaded from HuggingFace on the first
    call. Subsequent calls use the cached singleton. The blocking inference
    runs in a thread-pool executor to avoid stalling the async event loop.

    horizon is clamped server-side to [1, 12].
    """
    global _forecaster
    horizon = max(1, min(12, horizon))

    try:
        if _forecaster is None:
            from src.forecasting.invoice_forecaster import InvoiceForecaster
            _forecaster = InvoiceForecaster()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    logger.info(f"[API] Forecast | client_id={client_id} horizon={horizon}")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, partial(_forecaster.forecast, client_id, horizon)
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "TimeFM is not installed. "
                "Run: pip install -e \"C:/Users/91838/Downloads/TimeFM/timesfm[torch]\""
            ),
        )
    except Exception as exc:
        logger.error(f"[API] Forecast error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return result


@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit("60/minute")
async def chat(request: Request, body: ChatRequest, _user: dict = Depends(require_msp)):
    """
    Run the full RAG pipeline for a user question.

    The user selects which LLM to use via the `provider` + `model` fields.
    Retrieval and reranking always use the pipeline's defaults (gpt-4o-mini
    for reranking) — only the final generation step uses the chosen model.

    The blocking pipeline.query() call runs in a thread-pool executor to
    avoid stalling FastAPI's async event loop.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    # Warn early if the vector index has no data (Supabase not migrated)
    if _pipeline.index.ntotal == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "The knowledge base index is empty. "
                "Please migrate data to Supabase first: "
                "python scripts/migrate_to_supabase.py"
            ),
        )

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info(
        f"[API] Chat | provider={body.provider} model={body.model} | "
        f"query={message[:80]!r}"
    )

    generator = _make_generator(body.provider, body.model)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, partial(_pipeline.query, message, generator, session_id=body.session_id or "")
    )

    citations = [
        CitationModel(
            index=cit["index"],
            title=cit["title"],
            source=cit["source"],
            source_type=cit["source_type"],
            chunk_index=cit["chunk_index"],
            relevance_score=cit["relevance_score"],
            url=cit.get("url"),
        )
        for cit in result.citations
    ]

    # Build context bundle model if available
    ctx_bundle_model = None
    if result.context_bundle is not None:
        try:
            bundle = result.context_bundle
            ctx_bundle_model = ContextBundleModel(
                total_tokens=bundle.total_tokens,
                budget_tokens=bundle.budget_tokens,
                truncated=bundle.truncated,
                dropped_count=bundle.dropped_count,
                strategy_used=bundle.strategy_used,
                fast_path=bundle.fast_path,
                pieces=[
                    ContextPieceModel(
                        id=p.id,
                        tier=p.tier,
                        source_type=p.source_type,
                        relevance_score=round(p.relevance_score, 4),
                        freshness_score=round(p.freshness_score, 4),
                        tokens=p.tokens,
                        included=p.included,
                        context_position=p.context_position,
                    )
                    for p in bundle.all_pieces
                ],
            )
        except Exception as exc:
            logger.warning(f"[API] context bundle serialisation failed: {exc}")

    response = ChatResponse(
        answer=result.answer,
        citations=citations,
        blocked=result.blocked,
        blocked_reason=result.blocked_reason,
        latency_ms=LatencyModel(
            retrieval=round(result.retrieval_ms, 1),
            rerank=round(result.rerank_ms, 1),
            generation=round(result.generation_ms, 1),
            total=round(result.total_ms, 1),
        ),
        tokens=TokenModel(
            prompt=result.prompt_tokens,
            completion=result.completion_tokens,
            total=result.prompt_tokens + result.completion_tokens,
        ),
        estimated_cost_usd=round(result.estimated_cost_usd, 6),
        pii_redacted=result.pii_redacted,
        model=result.model,
        provider=body.provider,
        context_bundle=ctx_bundle_model,
    )

    # Persist interaction to JSONL log (non-blocking; errors must not affect the response)
    try:
        log_interaction(
            session_id=body.session_id,
            query=message,
            answer=result.answer,
            provider=body.provider,
            model=result.model,
            blocked=result.blocked,
            blocked_reason=result.blocked_reason,
            citations=result.citations,
            retrieval_ms=result.retrieval_ms,
            rerank_ms=result.rerank_ms,
            generation_ms=result.generation_ms,
            total_ms=result.total_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
            pii_redacted=result.pii_redacted,
        )
    except Exception as exc:
        logger.warning(f"[API] Chat log write failed (non-fatal): {exc}")

    return response


# ---------------------------------------------------------------------------
# Logs & monitoring routes
# ---------------------------------------------------------------------------

@app.get("/logs", include_in_schema=False)
async def serve_logs_ui():
    logs_path = STATIC_DIR / "logs.html"
    if not logs_path.exists():
        raise HTTPException(status_code=404, detail="Logs UI not found at app/static/logs.html")
    return FileResponse(str(logs_path), media_type="text/html")


@app.get("/api/logs")
async def get_logs(
    limit:  int = Query(default=50,  ge=1, le=500),
    offset: int = Query(default=0,   ge=0),
    _user: dict = Depends(require_msp),
):
    """Return paginated chat log records (newest first)."""
    records = load_logs(limit=limit, offset=offset)
    return {"records": records, "count": len(records), "limit": limit, "offset": offset}


@app.get("/api/logs/stats")
async def get_logs_stats(_user: dict = Depends(require_msp)):
    """Return aggregated statistics over the full chat history."""
    return compute_stats()


# ---------------------------------------------------------------------------
# Trace observability endpoints (Feature 4)
# ---------------------------------------------------------------------------

@app.get("/api/traces")
async def list_traces(
    verdict: Optional[str] = Query(default=None),
    capture_reason: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _user: dict = Depends(require_msp),
):
    """List recent agent traces from the failure-biased sampler store."""
    from src.observability.store import TraceStore
    store = TraceStore("data/traces")
    traces = store.query(verdict=verdict, capture_reason=capture_reason, limit=limit)
    return {"traces": traces, "count": len(traces)}


@app.get("/api/traces/summary")
async def get_traces_summary(
    hours: int = Query(default=24, ge=1, le=168),
    _user: dict = Depends(require_msp),
):
    """
    Return aggregated trace summary over past N hours.

    Includes:
      - Success rate, error rate
      - Latency percentiles (p50, p90, p95, p99)
      - Cost breakdown by model
      - Hallucination rate, alert count
    """
    from src.observability.aggregator import TraceAggregator

    aggregator = TraceAggregator(trace_store_path="data/traces")
    report = aggregator.summary(hours=hours)

    return {
        "period_hours": report.period_hours,
        "trace_count": report.trace_count,
        "success_rate": report.success_rate,
        "error_rate": report.error_rate,
        "latency_ms": {
            "avg": report.avg_latency_ms,
            "p50": report.p50_latency_ms,
            "p90": report.p90_latency_ms,
            "p95": report.p95_latency_ms,
            "p99": report.p99_latency_ms,
        },
        "cost_usd": {
            "total": report.total_cost_usd,
            "by_model": report.cost_by_model,
            "top_models": report.top_models,
        },
        "hallucination_rate": report.hallucination_rate,
        "alert_count": report.alert_count,
        "alerts": [
            {
                "metric": a.metric,
                "threshold": a.threshold,
                "actual": a.actual,
                "severity": a.severity,
            }
            for a in report.alerts
        ],
    }


@app.get("/api/agent-metrics")
async def get_agent_metrics(
    hours: int = Query(default=24, ge=1, le=168),
    _user: dict = Depends(require_msp),
):
    """
    Return agent-level decision metrics and benchmark results.

    Combines live trace signals with the most recent benchmark run to
    provide a complete picture of the multi-agent pipeline performance.
    """
    from src.observability.aggregator import TraceAggregator
    agg = TraceAggregator(trace_store_path="data/traces")
    metrics = agg.agent_metrics(hours=hours)
    throughput = agg.throughput_over_time(hours=hours, buckets=24)
    metrics["throughput"] = throughput
    return metrics


@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str, _user: dict = Depends(require_msp)):
    """Return a full trace case file by trace_id."""
    from src.observability.store import TraceStore
    store = TraceStore("data/traces")
    trace = store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found.")
    return trace


@app.post("/api/traces/{trace_id}/replay")
async def replay_trace(trace_id: str, _user: dict = Depends(require_msp)):
    """Re-run a historical trace through the current pipeline for diff comparison."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    from src.observability.replayer import TraceReplayer
    replayer = TraceReplayer(_pipeline)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, replayer.replay, trace_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found or has no query.")
    return result


# ---------------------------------------------------------------------------
# Council Orchestrator endpoint (Feature 6)
# ---------------------------------------------------------------------------

class CouncilRequest(BaseModel):
    message: str
    user_role: str = "msp"
    budget_tokens: int = 3000
    session_id: Optional[str] = None


class CouncilVerdictResponse(BaseModel):
    accepted_answer: str
    winning_agent: str
    dissent_summary: str
    escalated: bool
    policy_reasons: list[str]
    total_cost_usd: float
    trace_id: str
    hallucination_detected: bool
    pii_concern: bool
    latency_ms: float


@app.post("/api/council", response_model=CouncilVerdictResponse)
@limiter.limit("20/minute")
async def council_query(request: Request, body: CouncilRequest, _user: dict = Depends(require_msp)):
    """
    Run the 3-agent Council pattern for a high-stakes MSP query.

    FastCreative and ConservativeChecker agents run in parallel;
    PolicyVerifier selects the better-grounded answer or escalates.

    Latency: ~2,000 ms p95 (vs ~1,200 ms for single-agent mode).
    Use for contract renewals, overdue escalations, and billing disputes.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info(f"[API] Council | role={body.user_role} | query={message[:80]!r}")

    from src.agents.council import CouncilOrchestrator
    from src.observability.collector import TraceCollector
    from src.observability.schemas import TraceEvent

    council = CouncilOrchestrator(_pipeline)
    session_id = body.session_id or f"council_{int(time.time() * 1000)}"

    # Wrap with TraceCollector for observability
    with TraceCollector(
        session_id=session_id,
        query=message,
        model="council-3-agent",
        user_role=body.user_role,
    ) as tc:
        try:
            t0 = time.perf_counter()
            verdict = await council.run(
                query=message,
                budget_tokens=body.budget_tokens,
                session_id=session_id,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            # Record council verdict as a trace event
            tc.add_event(TraceEvent(
                event_type="council_verdict",
                payload={
                    "winning_agent": verdict.winning_agent,
                    "escalated": verdict.escalated,
                    "hallucination_detected": verdict.hallucination_detected,
                    "pii_concern": verdict.pii_concern,
                    "policy_reasons": verdict.policy_reasons[:3] if verdict.policy_reasons else [],
                },
                cost_usd=verdict.total_cost_usd,
            ))

            # Set final verdict based on council outcome
            if verdict.escalated:
                tc.set_verdict("escalated")
            elif verdict.hallucination_detected:
                tc.set_verdict("pii_redacted")
            else:
                tc.set_verdict("success")

            logger.info(
                f"[API] Council complete | session={session_id} | "
                f"agent={verdict.winning_agent} | latency={latency_ms:.0f}ms | "
                f"cost=${verdict.total_cost_usd:.6f}"
            )

        except Exception as exc:
            logger.error(f"[API] Council error: {exc}")
            tc.set_verdict("error")
            raise HTTPException(status_code=500, detail=str(exc))

    return CouncilVerdictResponse(**verdict.to_dict())


# ---------------------------------------------------------------------------
# Proactive Alerts API
# ---------------------------------------------------------------------------


@app.get("/api/alerts", response_model=AlertListResponse)
@limiter.limit("30/minute")
async def list_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    acknowledged: Optional[bool] = Query(None),
    _user: dict = Depends(require_msp),
):
    """
    List proactive alerts (AR risk, contract expiry, quality degradation).

    Query parameters:
      - limit: Max results (1-500, default 50)
      - offset: Pagination offset (default 0)
      - acknowledged: Filter by read status (None = all)

    Returns:
        Paginated alerts list
    """
    if not hasattr(app.state, "alert_store"):
        raise HTTPException(status_code=503, detail="Alert store not ready")

    try:
        store = app.state.alert_store
        alerts = await store.list_alerts(limit=limit, offset=offset, acknowledged=acknowledged)

        logger.info(f"[API] Listed {len(alerts)} alerts | role={_user.get('role')}")

        return AlertListResponse(
            alerts=[AlertModel(**alert) for alert in alerts],
            total=len(alerts),
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"[API] List alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/alerts/{alert_id}/acknowledge", response_model=dict)
@limiter.limit("30/minute")
async def acknowledge_alert(
    alert_id: str,
    acknowledged: bool = Query(True),
    _user: dict = Depends(require_msp),
):
    """
    Mark an alert as acknowledged (read) or unacknowledged.

    Path parameters:
      - alert_id: UUID of the alert

    Query parameters:
      - acknowledged: True = mark read, False = mark unread (default True)

    Returns:
        {success: bool, message: str}
    """
    if not hasattr(app.state, "alert_store"):
        raise HTTPException(status_code=503, detail="Alert store not ready")

    try:
        store = app.state.alert_store
        success = await store.acknowledge_alert(alert_id, acknowledged=acknowledged)

        if success:
            status = "acknowledged" if acknowledged else "unacknowledged"
            logger.info(f"[API] Alert {status}: {alert_id} | role={_user.get('role')}")
            return {"success": True, "message": f"Alert {status}"}
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Acknowledge alert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Skills API
# ---------------------------------------------------------------------------


class SkillMetadataModel(BaseModel):
    """Skill metadata."""
    name: str
    description: str
    version: str
    required_sources: list[str]


class SkillExecutionRequest(BaseModel):
    """Request to execute a skill."""
    query: Optional[str] = None
    metadata: dict = {}


class SkillExecutionResponse(BaseModel):
    """Response from skill execution."""
    success: bool
    skill_name: str
    data: Optional[dict] = None
    error: Optional[str] = None
    latency_ms: float


@app.get("/api/skills", response_model=list[SkillMetadataModel])
@limiter.limit("30/minute")
async def list_skills(
    request: Request,
    _user: dict = Depends(require_msp),
):
    """
    List available skills.

    Returns:
        List of skill metadata (name, description, version, required_sources)
    """
    try:
        from src.skills.registry import SkillRegistry

        registry = SkillRegistry()
        skills = registry.list_skills()

        logger.info(f"[API] Listed {len(skills)} skills | role={_user.get('role')}")

        return [SkillMetadataModel(**skill) for skill in skills]

    except Exception as e:
        logger.error(f"[API] List skills error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/skills/{skill_name}/execute", response_model=SkillExecutionResponse)
@limiter.limit("10/minute")
async def execute_skill(
    request: Request,
    skill_name: str,
    body: SkillExecutionRequest,
    _user: dict = Depends(require_msp),
):
    """
    Execute a skill.

    Path parameters:
      - skill_name: Name of the skill to execute

    Request body:
      - query: Optional query context (e.g., client name for escalation brief)
      - metadata: Optional metadata dict

    Returns:
        Skill execution result with data, error, and latency
    """
    try:
        from src.skills.base import SkillContext
        from src.skills.registry import SkillRegistry

        registry = SkillRegistry()
        skill = registry.get(skill_name)

        if not skill:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

        # Execute skill
        context = SkillContext(
            query=body.query or "",
            user_id=_user.get("sub"),
            user_role=_user.get("role", "msp"),
            metadata=body.metadata,
        )

        result = await skill.execute(context)

        logger.info(
            f"[API] Skill executed: {skill_name} | "
            f"success={result.success} | latency={result.latency_ms:.0f}ms | "
            f"role={_user.get('role')}"
        )

        return SkillExecutionResponse(
            success=result.success,
            skill_name=result.skill_name,
            data=result.data,
            error=result.error,
            latency_ms=result.latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Execute skill error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
