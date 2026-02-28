"""
TechVault Enterprise RAG - Web API Server
------------------------------------------
FastAPI server that wraps the Phase III RAGPipeline and serves the chat UI.

Endpoints:
  GET  /              -> serve app/static/index.html
  GET  /api/health    -> pipeline status, vector count, model name
  POST /api/chat      -> run full RAG query, return answer + citations

Run from the project root (Enterprise_RAG/):
    uvicorn app.server:app --reload --port 8000

The pipeline loads data/index/ relative to CWD, so the working directory
MUST be Enterprise_RAG/ when starting the server.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Windows cp1252 terminal fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"
INDEX_DIR = "data/index"

# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once at startup; clean up on shutdown."""
    global _pipeline
    try:
        from src.serving.pipeline import RAGPipeline
        from src.utils.logger import setup_logger

        setup_logger()
        logger.info("[Server] Loading RAG pipeline...")
        _pipeline = RAGPipeline(index_dir=INDEX_DIR)
        logger.info(
            f"[Server] Pipeline ready | "
            f"{_pipeline.index.faiss_index.ntotal:,} vectors | "
            f"model={_pipeline.generator.model}"
        )
    except FileNotFoundError as exc:
        logger.error(
            f"[Server] Index not found: {exc}\n"
            "Run Phase II first: python -m src.main phase2"
        )
        raise
    yield
    _pipeline = None
    logger.info("[Server] Pipeline unloaded.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TechVault Enterprise RAG API",
    description="Retrieval-Augmented Generation over MSP operations and AI research",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS / JS if ever split out)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the chat UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found at app/static/index.html")
    return FileResponse(str(index_path), media_type="text/html")


@app.get("/api/health")
async def health():
    """Return pipeline status and index metadata."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return {
        "status": "ok",
        "vectors": _pipeline.index.faiss_index.ntotal,
        "model": _pipeline.generator.model,
        "reranking_enabled": _pipeline.enable_reranking,
        "rerank_top_k": _pipeline.rerank_top_k,
        "top_k": _pipeline.retriever.top_k,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Run the full RAG pipeline for a user question.

    The pipeline.query() call is blocking (synchronous OpenAI SDK calls),
    so it runs in a thread-pool executor to avoid blocking the event loop.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    logger.info(f"[API] Chat request: {message[:80]!r}")

    # Run blocking pipeline in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _pipeline.query, message)

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

    prompt_tok = result.prompt_tokens
    comp_tok = result.completion_tokens

    return ChatResponse(
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
            prompt=prompt_tok,
            completion=comp_tok,
            total=prompt_tok + comp_tok,
        ),
        estimated_cost_usd=round(result.estimated_cost_usd, 6),
        pii_redacted=result.pii_redacted,
        model=result.model,
    )
