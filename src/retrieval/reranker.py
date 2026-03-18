"""
LLM Reranker
-------------
Uses a single OpenAI call to score all retrieved chunks at once (listwise).
Sending all candidates in one prompt is faster and cheaper than calling
the API once per chunk.

Scoring rubric (0-10):
  9-10  Directly and fully answers the query
  6-8   Relevant, contains useful partial information
  3-5   Tangentially related
  0-2   Irrelevant or off-topic

Falls back to the original retrieval order if the LLM call fails.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

from langsmith import traceable
from loguru import logger
from openai import OpenAI

from src.chunking.schemas import Chunk

_RERANK_SYSTEM = (
    "You are a relevance scoring engine for an enterprise RAG system. "
    "Your only job is to output valid JSON -- no prose, no markdown fences."
)

_RERANK_USER = """\
Score each chunk for its relevance to the query on a scale of 0 to 10.

Rubric:
  9-10: Directly answers the query with specific facts/figures
  6-8 : Relevant, contains useful partial information
  3-5 : Tangentially related
  0-2 : Irrelevant or off-topic

Query: {query}

Chunks:
{chunks_block}

Return ONLY a JSON object with a "scores" key containing one entry per chunk, in order:
{{"scores": [{{"index": 1, "score": <0-10>}}, {{"index": 2, "score": <0-10>}}, ...]}}
"""


class LLMReranker:
    """
    Batch LLM reranker: one API call scores all candidates simultaneously.

    Args:
        model:         OpenAI model to use for scoring (default: gpt-4o-mini).
        rerank_top_k:  Number of chunks to keep after reranking (default: 5).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        rerank_top_k: int = 5,
    ) -> None:
        self.model = model
        self.rerank_top_k = rerank_top_k
        self._client = OpenAI()

    @traceable(name="rerank", run_type="chain")
    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """
        Score all candidates in a single LLM call, keep top rerank_top_k.

        Args:
            query:      The user's original question.
            candidates: List of (Chunk, retrieval_score) from the retriever.

        Returns:
            Reranked list of (Chunk, llm_score) -- top rerank_top_k entries,
            sorted by LLM relevance score descending.
            Falls back to original order on any API/parse failure.
        """
        if not candidates:
            return []

        # Build the numbered chunk block for the prompt
        chunks_block = "\n\n".join(
            f"[{i}] {chunk.text[:800]}"
            for i, (chunk, _) in enumerate(candidates, start=1)
        )

        scores: list[float] = self._score_batch(query, chunks_block, len(candidates))

        if not scores or len(scores) != len(candidates):
            logger.warning("[Reranker] Falling back to retrieval order (score mismatch)")
            return candidates[: self.rerank_top_k]

        # Pair chunks with their LLM scores
        scored = [
            (chunk, float(scores[i]))
            for i, (chunk, _) in enumerate(candidates)
        ]

        # Detect and demote anomalous scores (context poisoning defense)
        anomalies = self._detect_anomalies(scored)
        if anomalies:
            logger.warning(f"[Reranker] {len(anomalies)} scoring anomalies detected")

        # Re-sort after potential demotion
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.rerank_top_k]

        logger.info(
            f"[Reranker] {len(candidates)} -> {len(top)} chunks "
            f"| top score: {top[0][1]:.1f}"
        )
        for rank, (chunk, score) in enumerate(top, start=1):
            logger.debug(
                f"  #{rank} score={score:.1f} | {chunk.source_type} | "
                f"{chunk.title[:60]}"
            )
        return top

    def _score_batch(
        self, query: str, chunks_block: str, expected: int
    ) -> list[float]:
        """Call the LLM once to score all chunks. Returns [] on failure."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _RERANK_SYSTEM},
                    {
                        "role": "user",
                        "content": _RERANK_USER.format(
                            query=query,
                            chunks_block=chunks_block,
                        ),
                    },
                ],
                max_tokens=256,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "[]"
            # The model returns {"scores": [...]} or a direct array
            parsed = json.loads(raw)
            # Unwrap {"scores": [...]} envelope
            if isinstance(parsed, dict):
                for key in ("scores", "results", "chunks"):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
            if not isinstance(parsed, list):
                raise ValueError(f"Unexpected JSON shape: {type(parsed)}")

            # Extract scores in index order
            index_score_map: dict[int, float] = {
                int(item["index"]): float(item["score"]) for item in parsed
            }
            return [index_score_map.get(i, 0.0) for i in range(1, expected + 1)]

        except Exception as exc:
            logger.warning(f"[Reranker] Batch scoring failed: {exc}")
            return []

    def _detect_anomalies(
        self, scored: list[tuple["Chunk", float]]
    ) -> list[dict]:
        """
        Detect scoring anomalies that may indicate context poisoning.

        Groups chunks by source_type, computes mean/stddev per group,
        and flags chunks with z-score > 2.0. Demotes outliers with
        z-score > 2.5 by halving their score.

        Returns list of anomaly dicts for logging.
        """
        if len(scored) < 3:
            return []

        # Group scores by source_type
        groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for idx, (chunk, score) in enumerate(scored):
            src = getattr(chunk, "source_type", "unknown")
            groups[src].append((idx, score))

        anomalies = []
        all_scores = [s for _, s in scored]
        global_mean = sum(all_scores) / len(all_scores)
        global_var = sum((s - global_mean) ** 2 for s in all_scores) / len(all_scores)
        global_std = math.sqrt(global_var) if global_var > 0 else 1.0

        for src_type, entries in groups.items():
            if len(entries) < 2:
                continue
            scores_in_group = [s for _, s in entries]
            mean = sum(scores_in_group) / len(scores_in_group)
            var = sum((s - mean) ** 2 for s in scores_in_group) / len(scores_in_group)
            std = math.sqrt(var) if var > 0 else 1.0

            for idx, score in entries:
                z_score = abs(score - mean) / std if std > 0.01 else 0.0
                global_z = abs(score - global_mean) / global_std if global_std > 0.01 else 0.0

                if z_score > 2.0 or global_z > 2.0:
                    anomaly = {
                        "chunk_index": idx,
                        "source_type": src_type,
                        "score": score,
                        "group_mean": round(mean, 2),
                        "z_score": round(z_score, 2),
                        "global_z": round(global_z, 2),
                    }
                    anomalies.append(anomaly)

                    # Demote chunks with extreme z-scores
                    if z_score > 2.5 or global_z > 2.5:
                        old_score = scored[idx][1]
                        chunk_ref = scored[idx][0]
                        scored[idx] = (chunk_ref, old_score * 0.5)
                        anomaly["demoted"] = True
                        anomaly["new_score"] = round(old_score * 0.5, 2)
                        logger.warning(
                            f"[Reranker] Anomaly: {src_type} chunk #{idx} "
                            f"score={score:.1f} z={z_score:.1f} -> demoted to {old_score * 0.5:.1f}"
                        )
