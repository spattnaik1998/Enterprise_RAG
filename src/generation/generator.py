"""
RAG Generator
--------------
Two generator implementations with an identical generate() interface:

  RAGGenerator      -- OpenAI (gpt-4o-mini, gpt-4o)
  AnthropicGenerator -- Anthropic (claude-haiku-4-5, claude-sonnet-4-6)

Both accept (query, reranked_chunks) and return a RAGResponse.
The serving pipeline and API server select the right generator at
request time based on the user's model choice.

Context budget (both generators):
  - System prompt + up to 5 chunks: ~3-6k tokens
  - Completion cap: 1,024 tokens
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langsmith import traceable
from loguru import logger

from src.chunking.schemas import Chunk
from src.generation.prompts import CITATION_TEMPLATE, NO_CONTEXT_RESPONSE, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Model pricing table  (input_$/M, output_$/M)
# ---------------------------------------------------------------------------

_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":               (0.150,  0.600),
    "gpt-4o":                    (2.500, 10.000),
    "claude-haiku-4-5-20251001": (0.800,  4.000),
    "claude-sonnet-4-6":         (3.000, 15.000),
}


def _cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute estimated cost in USD for a given model and token counts."""
    rates = _MODEL_PRICING.get(model, (0.150, 0.600))
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """Structured result from a single generation call (provider-agnostic)."""

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
        return _cost_usd(self.model, self.prompt_tokens, self.completion_tokens)


# ---------------------------------------------------------------------------
# Shared context builder
# ---------------------------------------------------------------------------

def _build_context(
    reranked_chunks: list[tuple[Chunk, float]],
    max_chunks: int,
) -> tuple[str, list[dict]]:
    """
    Number each chunk [1]..[N], format it for the system prompt, and
    produce a parallel list of citation dicts.

    Returns (context_string, citations_list).
    """
    context_parts: list[str] = []
    citations: list[dict] = []

    for i, (chunk, score) in enumerate(reranked_chunks[:max_chunks], start=1):
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
    return context, citations


# ---------------------------------------------------------------------------
# OpenAI Generator
# ---------------------------------------------------------------------------

class RAGGenerator:
    """
    Grounded answer synthesis using OpenAI chat models.

    Supported models: gpt-4o-mini (default, fast), gpt-4o (higher quality).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_context_chunks: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        from openai import OpenAI  # lazy import keeps import graph clean
        self.model = model
        self.max_context_chunks = max_context_chunks
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI()

    @traceable(name="generate_openai", run_type="llm")
    def generate(
        self,
        query: str,
        reranked_chunks: list[tuple[Chunk, float]],
    ) -> RAGResponse:
        if not reranked_chunks:
            return RAGResponse(answer=NO_CONTEXT_RESPONSE, citations=[], model=self.model)

        context, citations = _build_context(reranked_chunks, self.max_context_chunks)
        system_message = SYSTEM_PROMPT.format(context=context)

        logger.debug(
            f"[OpenAIGenerator] {self.model} | {len(citations)} chunks | "
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
            f"[OpenAIGenerator] Done | prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} | "
            f"cost=${_cost_usd(self.model, usage.prompt_tokens, usage.completion_tokens):.5f}"
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )


# ---------------------------------------------------------------------------
# Anthropic Generator
# ---------------------------------------------------------------------------

class AnthropicGenerator:
    """
    Grounded answer synthesis using Anthropic Claude models.

    Supported models:
      claude-haiku-4-5-20251001  (fast, $0.80/M in, $4.00/M out)
      claude-sonnet-4-6          (higher quality, $3.00/M in, $15.00/M out)

    The Anthropic SDK passes the system prompt as a separate `system`
    parameter (not inside the messages list) — handled here transparently.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_context_chunks: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        from anthropic import Anthropic  # lazy import
        self.model = model
        self.max_context_chunks = max_context_chunks
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = Anthropic()

    @traceable(name="generate_anthropic", run_type="llm")
    def generate(
        self,
        query: str,
        reranked_chunks: list[tuple[Chunk, float]],
    ) -> RAGResponse:
        if not reranked_chunks:
            return RAGResponse(answer=NO_CONTEXT_RESPONSE, citations=[], model=self.model)

        context, citations = _build_context(reranked_chunks, self.max_context_chunks)
        system_message = SYSTEM_PROMPT.format(context=context)

        logger.debug(
            f"[AnthropicGenerator] {self.model} | {len(citations)} chunks | "
            f"query={query[:60]!r}"
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_message,
            messages=[{"role": "user", "content": query}],
        )

        answer = response.content[0].text if response.content else ""
        # Anthropic usage: input_tokens / output_tokens
        prompt_tok = response.usage.input_tokens
        comp_tok = response.usage.output_tokens

        logger.info(
            f"[AnthropicGenerator] Done | input={prompt_tok} "
            f"output={comp_tok} | "
            f"cost=${_cost_usd(self.model, prompt_tok, comp_tok):.5f}"
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            model=self.model,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
        )
