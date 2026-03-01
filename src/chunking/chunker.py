"""
Enterprise RAG - Adaptive Chunker
-----------------------------------
Different document types demand different chunking strategies to maximise
retrieval precision.  The chunker inspects `source_type` and document length
to select the right strategy automatically.

Strategy selection logic:
  - SHORT structured docs (invoices, tickets, comms, contracts < 600 tokens):
      KEEP_WHOLE -- the entire document is one chunk.  Splitting a 200-word
      invoice would scatter the invoice ID, amount, and overdue-days across
      separate vectors, destroying the signal for queries like
      "show me all overdue invoices for Lakewood".

  - MEDIUM structured docs (CRM profiles, 400-900 tokens):
      FIXED_OVERLAP -- 512-token windows with 64-token stride so the financial
      summary and contacts sections are each represented without being split.

  - LONG narrative docs (Wikipedia, ArXiv abstracts, RSS, 900+ tokens):
      SENTENCE_WINDOW -- sliding windows of N sentences with M-sentence overlap.
      Keeps semantic continuity and avoids mid-sentence cuts.

All strategies respect a hard token ceiling (MAX_TOKENS) so no chunk ever
exceeds the OpenAI embedding model's 8191-token context window.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import nltk
import tiktoken
from loguru import logger

from src.chunking.schemas import Chunk

# Download sentence tokenizer data if not present (one-time, offline-safe)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_TOKENS = 512          # Hard ceiling per chunk (embedding model limit is 8191)
OVERLAP_TOKENS = 64       # Token overlap for fixed-overlap strategy
SENTENCE_WINDOW = 8       # Sentences per window for sentence-window strategy
SENTENCE_OVERLAP = 2      # Sentence overlap between windows
KEEP_WHOLE_THRESHOLD = 600  # Docs below this token count are kept whole

# Source types that benefit from keep-whole even if slightly longer
STRUCTURED_SOURCE_TYPES = {"billing", "psa", "communications", "contracts"}

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count BPE tokens using the cl100k_base encoder (GPT-3.5/4 / text-embedding-3-*)."""
    return len(_ENC.encode(text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK punkt tokenizer."""
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on period + space
        sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# ── Main Chunker ──────────────────────────────────────────────────────────────

class AdaptiveChunker:
    """
    Selects and applies the appropriate chunking strategy for each document.

    Usage:
        chunker = AdaptiveChunker()
        chunks = list(chunker.chunk_document(validated_doc))
    """

    def __init__(
        self,
        max_tokens: int = MAX_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
        sentence_window: int = SENTENCE_WINDOW,
        sentence_overlap: int = SENTENCE_OVERLAP,
        keep_whole_threshold: int = KEEP_WHOLE_THRESHOLD,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.sentence_window = sentence_window
        self.sentence_overlap = sentence_overlap
        self.keep_whole_threshold = keep_whole_threshold

    def chunk_document(self, doc) -> list[Chunk]:
        """
        Chunk a ValidatedDocument using the appropriate strategy.

        Args:
            doc: A ValidatedDocument instance.

        Returns:
            List of Chunk objects ready for embedding.
        """
        token_count = count_tokens(doc.content)
        source_type = doc.source_type.value

        # Strategy selection
        if self._use_keep_whole(source_type, token_count):
            strategy = "keep_whole"
            chunks = list(self._keep_whole(doc))
        elif self._use_fixed_overlap(source_type, token_count):
            strategy = "fixed_overlap"
            chunks = list(self._fixed_overlap(doc))
        else:
            strategy = "sentence_window"
            chunks = list(self._sentence_window(doc))

        logger.debug(
            f"[Chunker] {doc.id[:12]} | {source_type} | "
            f"{token_count} tokens | {strategy} -> {len(chunks)} chunk(s)"
        )
        return chunks

    def chunk_batch(self, docs: list) -> list[Chunk]:
        """Chunk a list of ValidatedDocuments. Returns flat list of all chunks."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # --- Strategy: Keep Whole ------------------------------------------------

    def _use_keep_whole(self, source_type: str, token_count: int) -> bool:
        """Keep whole if short structured doc, or any doc under threshold."""
        if source_type in STRUCTURED_SOURCE_TYPES and token_count <= self.keep_whole_threshold * 2:
            return True
        return token_count <= self.keep_whole_threshold

    def _keep_whole(self, doc) -> Iterator[Chunk]:
        # Prepend the document title so client/invoice/contract names appear at
        # the very start of every enterprise chunk.  This helps both BM25 (title
        # tokens get the highest positional weight) and dense retrieval (transformer
        # encoders weight early tokens more heavily).
        text = f"[{doc.title}]\n\n{doc.content}"
        yield Chunk(
            doc_id=doc.id,
            chunk_index=0,
            chunk_strategy="keep_whole",
            text=text,
            token_count=count_tokens(text),
            source=doc.source,
            source_type=doc.source_type.value,
            title=doc.title,
            url=doc.url,
            metadata=doc.metadata,
        )

    # --- Strategy: Fixed Overlap ---------------------------------------------

    def _use_fixed_overlap(self, source_type: str, token_count: int) -> bool:
        """CRM profiles and medium docs get fixed-overlap chunking."""
        return source_type == "crm" or (
            token_count <= self.max_tokens * 4
            and source_type in STRUCTURED_SOURCE_TYPES
        )

    def _fixed_overlap(self, doc) -> Iterator[Chunk]:
        tokens = _ENC.encode(doc.content)
        stride = self.max_tokens - self.overlap_tokens
        i = 0
        chunk_index = 0

        while i < len(tokens):
            window = tokens[i: i + self.max_tokens]
            text = _ENC.decode(window)
            yield Chunk(
                doc_id=doc.id,
                chunk_index=chunk_index,
                chunk_strategy="fixed_overlap",
                text=text,
                token_count=len(window),
                source=doc.source,
                source_type=doc.source_type.value,
                title=doc.title,
                url=doc.url,
                metadata=doc.metadata,
            )
            chunk_index += 1
            i += stride
            if i + self.overlap_tokens >= len(tokens):
                break

    # --- Strategy: Sentence Window -------------------------------------------

    def _sentence_window(self, doc) -> Iterator[Chunk]:
        sentences = split_sentences(doc.content)
        if not sentences:
            yield from self._keep_whole(doc)
            return

        i = 0
        chunk_index = 0
        while i < len(sentences):
            window = sentences[i: i + self.sentence_window]
            text = " ".join(window)
            tokens = count_tokens(text)

            # If window still too long, truncate at token boundary
            if tokens > self.max_tokens:
                enc = _ENC.encode(text)
                text = _ENC.decode(enc[: self.max_tokens])
                tokens = self.max_tokens

            if text.strip():
                yield Chunk(
                    doc_id=doc.id,
                    chunk_index=chunk_index,
                    chunk_strategy="sentence_window",
                    text=text,
                    token_count=tokens,
                    source=doc.source,
                    source_type=doc.source_type.value,
                    title=doc.title,
                    url=doc.url,
                    metadata=doc.metadata,
                )
                chunk_index += 1

            i += self.sentence_window - self.sentence_overlap
