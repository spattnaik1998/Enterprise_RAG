"""
RAG Generator
--------------
Formats reranked chunks into a grounded context window, calls the LLM,
and returns a structured RAGResponse with the answer, citations, and
token usage for cost tracking.

Model: gpt-4o-mini (128k context, cheap, fast -- ideal for RAG generation)
Context budget:
  - System prompt + N chunks: ~3-5k tokens
  - Completion cap: 1,024 tokens
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langsmith import traceable
from loguru import logger
from openai import OpenAI

from src.chunking.schemas import Chunk
from src.generation.prompts import CITATION_TEMPLATE, NO_CONTEXT_RESPONSE, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """Structured result from a single generation call."""

    answer: str
    citations: list[dict]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    pii_redacted: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Approximate cost for gpt-4o-mini pricing ($0.15/M input, $0.60/M output)."""
        return (self.prompt_tokens * 0.15 + self.completion_tokens * 0.60) / 1_000_000


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class RAGGenerator:
    """
    Builds a grounded prompt from reranked chunks and calls OpenAI.

    Each chunk is numbered [1]..[N] in the context and a matching
    citation object is returned so the caller can render source links.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_context_chunks: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.max_context_chunks = max_context_chunks
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI()

    @traceable(name="generate", run_type="llm")
    def generate(
        self,
        query: str,
        reranked_chunks: list[tuple[Chunk, float]],
    ) -> RAGResponse:
        """
        Generate a grounded answer from the reranked chunks.

        Args:
            query:           The user's original question.
            reranked_chunks: (Chunk, relevance_score) pairs sorted best-first.

        Returns:
            RAGResponse with answer text, structured citations, and token stats.
        """
        if not reranked_chunks:
            return RAGResponse(
                answer=NO_CONTEXT_RESPONSE,
                citations=[],
                model=self.model,
            )

        # Build numbered context and citation list
        context_parts: list[str] = []
        citations: list[dict] = []

        for i, (chunk, score) in enumerate(
            reranked_chunks[: self.max_context_chunks], start=1
        ):
            citation_line = CITATION_TEMPLATE.format(
                index=i,
                title=chunk.title,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
            )
            context_parts.append(f"[{i}] {chunk.text}\nSource: {citation_line}")
            citations.append(
                {
                    "index": i,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "source_type": chunk.source_type,
                    "url": chunk.url,
                    "chunk_index": chunk.chunk_index,
                    "relevance_score": round(score, 4),
                }
            )

        context = "\n\n---\n\n".join(context_parts)
        system_message = SYSTEM_PROMPT.format(context=context)

        logger.debug(
            f"[Generator] {self.model} | {len(citations)} context chunks | "
            f"query={query[:60]!r}"
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        logger.info(
            f"[Generator] Done | prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} tokens"
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
