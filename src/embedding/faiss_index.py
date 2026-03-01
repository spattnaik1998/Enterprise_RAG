"""
FAISS Vector Index
-------------------
Wraps faiss.IndexFlatIP (inner product == cosine similarity after L2
normalisation) for fast approximate nearest-neighbour search.

The index stores:
  - A FAISS IndexFlatIP for vector search
  - A parallel list of Chunk objects (same ordering as FAISS row IDs)
  - A BM25 keyword index (rank_bm25) for the hybrid retrieval path

Persistence:
  - FAISS index -> data/index/faiss.index
  - Chunk metadata -> data/index/chunks.json
  - BM25 corpus tokens -> data/index/bm25_corpus.json
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from src.chunking.schemas import Chunk


def _bm25_tokens(text: str) -> list[str]:
    """Normalise text for BM25: lowercase, strip punctuation, split on whitespace.

    Using plain .split() on raw text causes mismatches for possessives (trust's),
    trailing commas/colons, and ampersands.  This helper strips all non-alphanumeric
    characters first so 'Heritage Bank & Trust' and "trust's" both tokenise to
    ['heritage', 'bank', 'trust'] and match correctly.
    """
    normalised = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in normalised.split() if len(t) > 1]

INDEX_DIR = Path("data/index")
FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
BM25_CORPUS_PATH = INDEX_DIR / "bm25_corpus.json"


class FAISSIndex:
    """
    Dual-index: FAISS (dense) + BM25 (sparse) for hybrid retrieval.

    Add chunks via build_from_chunks(), then call save().
    Load a persisted index via FAISSIndex.load().
    """

    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions
        self.faiss_index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimensions)
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self._bm25_corpus_tokens: list[list[str]] = []

    # --- Build ----------------------------------------------------------------

    def build_from_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Populate both indexes from chunks and their pre-computed embeddings.

        Args:
            chunks: List of Chunk objects.
            embeddings: Float32 array of shape (len(chunks), dimensions).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        logger.info(f"[FAISSIndex] Building index from {len(chunks)} chunks...")

        # FAISS index (embeddings must be float32 and L2-normalised)
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.faiss_index.add(embeddings_f32)
        self.chunks = list(chunks)

        # BM25 keyword index -- include title so client/invoice names are always
        # searchable, and strip punctuation for clean token matching.
        self._bm25_corpus_tokens = [
            _bm25_tokens(f"{chunk.title} {chunk.text}") for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self._bm25_corpus_tokens)

        logger.info(
            f"[FAISSIndex] FAISS index size: {self.faiss_index.ntotal} vectors | "
            f"BM25 corpus: {len(self._bm25_corpus_tokens)} documents"
        )

    # --- Search ---------------------------------------------------------------

    def search_dense(
        self, query_vec: np.ndarray, top_k: int = 10
    ) -> list[tuple[Chunk, float]]:
        """
        Dense (semantic) search.

        Returns: List of (Chunk, cosine_score) sorted descending.
        """
        qv = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        scores, indices = self.faiss_index.search(qv, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        return results

    def search_sparse(
        self, query_text: str, top_k: int = 10
    ) -> list[tuple[Chunk, float]]:
        """
        Sparse (BM25 keyword) search.

        Returns: List of (Chunk, bm25_score) sorted descending.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_from_chunks() first.")
        tokens = _bm25_tokens(query_text)
        bm25_scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        return [
            (self.chunks[i], float(bm25_scores[i]))
            for i in top_indices
            if bm25_scores[i] > 0
        ]

    def search_hybrid(
        self,
        query_vec: np.ndarray,
        query_text: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[tuple[Chunk, float]]:
        """
        Hybrid retrieval: fuse dense (FAISS) and sparse (BM25) rankings
        using Reciprocal Rank Fusion (RRF) with configurable weights.

        RRF score = sum_of(weight / (rank + 60)) -- robust to score scale mismatch.

        Args:
            query_vec: L2-normalised query embedding.
            query_text: Raw query string for BM25.
            top_k: Number of results to return.
            dense_weight: Weight for semantic results (default 0.7).
            sparse_weight: Weight for keyword results (default 0.3).

        Returns:
            List of (Chunk, fused_score) sorted descending.
        """
        k = max(top_k * 5, 60)  # Retrieve 5x candidates before fusion

        dense_results = self.search_dense(query_vec, top_k=k)
        sparse_results = self.search_sparse(query_text, top_k=k)

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                dense_weight / (rank + 60)
            )
            chunk_map[chunk.chunk_id] = chunk

        for rank, (chunk, _) in enumerate(sparse_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                sparse_weight / (rank + 60)
            )
            chunk_map[chunk.chunk_id] = chunk

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(chunk_map[cid], score) for cid, score in fused]

    # --- Persistence ----------------------------------------------------------

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        """Persist FAISS index + chunk metadata + BM25 corpus to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss_index, str(index_dir / "faiss.index"))
        logger.info(f"[FAISSIndex] FAISS index saved -> {index_dir}/faiss.index")

        chunks_data = [c.model_dump(mode="json") for c in self.chunks]
        (index_dir / "chunks.json").write_text(
            json.dumps(chunks_data, indent=2, default=str), encoding="utf-8"
        )
        logger.info(f"[FAISSIndex] {len(chunks_data)} chunk records saved -> {index_dir}/chunks.json")

        (index_dir / "bm25_corpus.json").write_text(
            json.dumps(self._bm25_corpus_tokens), encoding="utf-8"
        )
        logger.info(f"[FAISSIndex] BM25 corpus saved -> {index_dir}/bm25_corpus.json")

        # Write index manifest
        manifest = {
            "total_vectors": self.faiss_index.ntotal,
            "dimensions": self.dimensions,
            "total_chunks": len(self.chunks),
            "source_types": list({c.source_type for c in self.chunks}),
        }
        (index_dir / "index_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "FAISSIndex":
        """Load a persisted index from disk."""
        instance = cls()
        instance.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))

        raw_chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        instance.chunks = [Chunk(**c) for c in raw_chunks]

        corpus_tokens = json.loads((index_dir / "bm25_corpus.json").read_text(encoding="utf-8"))
        instance._bm25_corpus_tokens = corpus_tokens
        instance.bm25 = BM25Okapi(corpus_tokens)

        logger.info(
            f"[FAISSIndex] Loaded: {instance.faiss_index.ntotal} vectors, "
            f"{len(instance.chunks)} chunks"
        )
        return instance

    @property
    def is_built(self) -> bool:
        return self.faiss_index.ntotal > 0
