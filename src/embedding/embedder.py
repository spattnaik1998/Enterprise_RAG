"""
OpenAI Embedding Client with LangSmith instrumentation
---------------------------------------------------------
Wraps the OpenAI text-embedding-3-small API with:
  - Batching (up to 2048 texts per API call)
  - LangSmith run tracing for cost / latency observability
  - Retry logic via tenacity
  - Token usage logging
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# LangSmith client (no-op if credentials not configured)
try:
    from langsmith import Client as LangSmithClient
    from langsmith import traceable
    _LS_CLIENT = LangSmithClient() if os.getenv("LANGSMITH_API_KEY") else None
except ImportError:
    _LS_CLIENT = None
    def traceable(*args, **kwargs):           # type: ignore
        def decorator(fn):
            return fn
        return decorator


MODEL = "text-embedding-3-small"
DIMENSIONS = 1536          # text-embedding-3-small native dimensions
BATCH_SIZE = 512           # OpenAI allows up to 2048; 512 keeps requests < 1 MB


class Embedder:
    """
    Generates L2-normalised embeddings using text-embedding-3-small.

    Embeddings are normalised to unit length so cosine similarity ==
    inner product, which lets us use IndexFlatIP (FAISS inner product)
    as a cosine similarity index.
    """

    def __init__(self, model: str = MODEL, batch_size: int = BATCH_SIZE) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.total_tokens_used: int = 0
        self.total_api_calls: int = 0

    @traceable(name="embed_texts", run_type="embedding")
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings and return an (N, DIMENSIONS) float32 array.
        Texts are processed in batches to stay within API limits.
        """
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            embeddings, tokens = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
            self.total_tokens_used += tokens
            self.total_api_calls += 1

            logger.debug(
                f"[Embedder] Batch {i // self.batch_size + 1} | "
                f"{len(batch)} texts | {tokens} tokens | "
                f"Running total: {self.total_tokens_used} tokens"
            )

        matrix = np.array(all_embeddings, dtype=np.float32)
        # L2-normalise so cosine sim == inner product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid div-by-zero
        return (matrix / norms).astype(np.float32)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Call the OpenAI Embeddings API for a single batch."""
        # Replace empty strings with a space to avoid API errors
        safe_texts = [t if t.strip() else " " for t in texts]
        start = time.perf_counter()
        response = self._client.embeddings.create(model=self.model, input=safe_texts)
        elapsed = time.perf_counter() - start

        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        tokens_used = response.usage.total_tokens
        logger.debug(f"[Embedder] API call: {len(texts)} texts, {tokens_used} tokens, {elapsed:.2f}s")
        return embeddings, tokens_used

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1536,) float32 array."""
        return self.embed_texts([text])[0]

    def usage_summary(self) -> dict:
        return {
            "model": self.model,
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            # text-embedding-3-small: $0.020 per million tokens (as of Feb 2026)
            "estimated_cost_usd": round(self.total_tokens_used / 1_000_000 * 0.020, 6),
        }
