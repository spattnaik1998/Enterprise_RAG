"""
Hybrid Retriever
-----------------
Embeds the user query and performs hybrid (dense + sparse) search
over the FAISS + BM25 dual index built in Phase II.

The retriever is stateless per query -- call retrieve() as many times
as you like from the same instance.
"""
from __future__ import annotations

import numpy as np
from langsmith import traceable
from loguru import logger

from src.chunking.schemas import Chunk
from src.embedding.embedder import Embedder
from src.embedding.faiss_index import FAISSIndex


class HybridRetriever:
    """
    Wraps FAISSIndex.search_hybrid() with automatic query embedding.

    Dense path  : FAISS IndexFlatIP (cosine similarity on L2-normalised vecs)
    Sparse path : BM25Okapi keyword match
    Fusion      : Reciprocal Rank Fusion with configurable weights
    """

    def __init__(
        self,
        index: FAISSIndex,
        embedder: Embedder,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    @traceable(name="retrieve", run_type="retriever")
    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """
        Embed the query and return top-k chunks via hybrid RRF search.

        Args:
            query: Raw user query string.

        Returns:
            List of (Chunk, fused_rrf_score) sorted by score descending.
        """
        logger.debug(f"[Retriever] Query: {query[:80]!r}")

        query_vec: np.ndarray = self.embedder.embed_query(query)

        results = self.index.search_hybrid(
            query_vec=query_vec,
            query_text=query,
            top_k=self.top_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )

        logger.info(
            f"[Retriever] Retrieved {len(results)} candidates "
            f"(top score: {results[0][1]:.4f})" if results else "[Retriever] No results"
        )
        return results
