"""
Supabase Vector Index
----------------------
Drop-in replacement for FAISSIndex that uses:
  - pgvector (via match_chunks RPC)        for dense semantic search
  - PostgreSQL FTS (via search_chunks_fts) for sparse keyword search
  - Reciprocal Rank Fusion                 for hybrid score fusion

The public interface is identical to FAISSIndex.search_hybrid() so
HybridRetriever requires zero changes.
"""
from __future__ import annotations

import os
from typing import Optional

from loguru import logger

from src.chunking.schemas import Chunk

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def _make_client():
    from supabase import create_client
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env"
        )
    return create_client(_SUPABASE_URL, _SUPABASE_KEY)


class SupabaseIndex:
    """
    Hybrid vector index backed by Supabase / PostgreSQL.

    Dense  : pgvector cosine similarity via match_chunks() RPC
    Sparse : PostgreSQL full-text search via search_chunks_fts() RPC
    Fusion : Reciprocal Rank Fusion (same formula as FAISSIndex)
    """

    def __init__(self) -> None:
        self._sb = _make_client()
        self._ntotal: Optional[int] = None   # cached row count

    # ------------------------------------------------------------------
    # Public interface (mirrors FAISSIndex)
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        """Total number of indexed chunks (cached after first call)."""
        if self._ntotal is None:
            try:
                result = (
                    self._sb.table("rag_chunks")
                    .select("chunk_id", count="exact")
                    .limit(0)
                    .execute()
                )
                self._ntotal = result.count or 0
            except Exception as exc:
                logger.warning(f"[SupabaseIndex] ntotal query failed: {exc}")
                self._ntotal = 0
        return self._ntotal

    @property
    def is_built(self) -> bool:
        return self.ntotal > 0

    def search_hybrid(
        self,
        query_vec,                    # np.ndarray  (1536,)
        query_text: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[tuple[Chunk, float]]:
        """
        Hybrid retrieval: fuse pgvector dense + FTS sparse rankings via RRF.

        Args:
            query_vec:    L2-normalised query embedding (numpy array).
            query_text:   Raw query string for full-text search.
            top_k:        Number of fused results to return.
            dense_weight: RRF weight applied to semantic results.
            sparse_weight: RRF weight applied to keyword results.

        Returns:
            List of (Chunk, fused_rrf_score) sorted descending.
        """
        k = max(top_k * 5, 60)   # over-fetch before fusion

        dense_results  = self._search_dense(query_vec, k)
        sparse_results = self._search_sparse(query_text, k)

        # Reciprocal Rank Fusion (matches FAISSIndex.search_hybrid exactly)
        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0.0) + dense_weight / (rank + 60)
            )
            chunk_map[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(sparse_results):
            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0.0) + sparse_weight / (rank + 60)
            )
            chunk_map[chunk.chunk_id] = chunk

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = [(chunk_map[cid], score) for cid, score in fused]

        logger.info(
            f"[SupabaseIndex] Hybrid search | dense={len(dense_results)} "
            f"sparse={len(sparse_results)} fused={len(results)}"
        )
        return results

    # ------------------------------------------------------------------
    # Private search methods
    # ------------------------------------------------------------------

    def _search_dense(
        self, query_vec, k: int
    ) -> list[tuple[Chunk, float]]:
        """Call match_chunks() RPC for pgvector cosine similarity search."""
        embedding = query_vec.astype(float).tolist()
        try:
            result = self._sb.rpc(
                "match_chunks",
                {"query_embedding": embedding, "match_count": k},
            ).execute()
            rows = result.data or []
        except Exception as exc:
            logger.warning(f"[SupabaseIndex] Dense search RPC failed: {exc}")
            return []

        return [(self._row_to_chunk(r), float(r.get("similarity", 0.0))) for r in rows]

    def _search_sparse(
        self, query_text: str, k: int
    ) -> list[tuple[Chunk, float]]:
        """Call search_chunks_fts() RPC for PostgreSQL full-text search."""
        # websearch_to_tsquery rejects empty strings — guard here
        clean = query_text.strip()
        if not clean:
            return []

        try:
            result = self._sb.rpc(
                "search_chunks_fts",
                {"query_text": clean, "match_count": k},
            ).execute()
            rows = result.data or []
        except Exception as exc:
            logger.warning(f"[SupabaseIndex] Sparse search RPC failed: {exc}")
            return []

        return [(self._row_to_chunk(r), float(r.get("rank", 0.0))) for r in rows]

    # ------------------------------------------------------------------
    # Row -> Chunk
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row: dict) -> Chunk:
        return Chunk(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            chunk_index=int(row.get("chunk_index", 0)),
            chunk_strategy=row.get("chunk_strategy", ""),
            text=row.get("text", ""),
            token_count=int(row.get("token_count", 0)),
            source=row.get("source", ""),
            source_type=row.get("source_type", ""),
            title=row.get("title", ""),
            url=row.get("url"),
            metadata=row.get("metadata") or {},
        )
