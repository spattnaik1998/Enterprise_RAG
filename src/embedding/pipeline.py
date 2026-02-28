"""
Phase II Pipeline - Chunk, Embed, Index
-----------------------------------------
Reads ValidatedDocuments from data/validated/,
chunks them with AdaptiveChunker,
embeds them with the OpenAI embedder,
and builds the dual FAISS + BM25 index.

LangSmith traces every embedding call automatically via
the @traceable decorator in embedder.py.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich import box

from src.chunking.chunker import AdaptiveChunker
from src.chunking.schemas import Chunk
from src.embedding.embedder import Embedder
from src.embedding.faiss_index import FAISSIndex
from src.schemas import ValidatedDocument

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
console = Console()


def load_validated_documents(validated_dir: str = "data/validated") -> list[ValidatedDocument]:
    """Load all ValidatedDocument JSON files from the checkpoint directory."""
    p = Path(validated_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"Validated documents directory not found: {p}\n"
            "Run Phase I first: python -m src.main phase1"
        )

    docs = []
    for json_file in sorted(p.glob("*.json")):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            docs.append(ValidatedDocument(**raw))
        except Exception as exc:
            logger.warning(f"[Phase2] Skipping {json_file.name}: {exc}")

    logger.info(f"[Phase2] Loaded {len(docs)} validated documents from {p}")
    return docs


def run_phase2(
    validated_dir: str = "data/validated",
    index_dir: str = "data/index",
    chunks_dir: str = "data/chunks",
    batch_size: int = 512,
    dense_weight: float = 0.7,
) -> dict:
    """
    Execute the full Phase II pipeline:
      1. Load validated documents from Phase I checkpoint
      2. Chunk with AdaptiveChunker
      3. Embed with OpenAI text-embedding-3-small
      4. Build FAISS + BM25 dual index
      5. Save index to data/index/
      6. Write phase2 checkpoint

    Returns:
        Summary dict with counts and cost estimate.
    """
    started_at = datetime.utcnow()

    console.print()
    console.print(
        Panel(
            "[bold cyan]Enterprise RAG Pipeline[/bold cyan]\n"
            "[white]Phase II - Chunking, Embedding, Indexing[/white]",
            box=box.DOUBLE_EDGE,
            expand=False,
        )
    )

    # -- Step 1: Load documents ------------------------------------------------
    console.print("\n[bold cyan]Step 1 / 4 - Loading validated documents[/bold cyan]")
    docs = load_validated_documents(validated_dir)
    if not docs:
        raise ValueError("No validated documents found. Run Phase I first.")
    console.print(f"[green][OK] {len(docs)} documents loaded[/green]")

    # -- Step 2: Chunk ---------------------------------------------------------
    console.print("\n[bold cyan]Step 2 / 4 - Adaptive chunking[/bold cyan]")
    chunker = AdaptiveChunker()
    chunks: list[Chunk] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Chunking documents...[/cyan]", total=len(docs))
        for doc in docs:
            chunks.extend(chunker.chunk_document(doc))
            progress.advance(task)

    # Chunk strategy breakdown
    strategy_counts: dict[str, int] = {}
    source_chunk_counts: dict[str, int] = {}
    for c in chunks:
        strategy_counts[c.chunk_strategy] = strategy_counts.get(c.chunk_strategy, 0) + 1
        source_chunk_counts[c.source_type] = source_chunk_counts.get(c.source_type, 0) + 1

    console.print(f"[green][OK] {len(chunks)} chunks produced from {len(docs)} documents[/green]")
    for strategy, count in sorted(strategy_counts.items()):
        console.print(f"  {strategy:20s}: {count:5d} chunks")

    # Save chunks to disk (for inspection / debugging)
    chunks_path = Path(chunks_dir)
    chunks_path.mkdir(parents=True, exist_ok=True)
    chunk_data = [c.model_dump(mode="json") for c in chunks]
    (chunks_path / "all_chunks.json").write_text(
        json.dumps(chunk_data, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"[Phase2] Chunks saved -> {chunks_path}/all_chunks.json")

    # -- Step 3: Embed ---------------------------------------------------------
    console.print("\n[bold cyan]Step 3 / 4 - Embedding chunks (OpenAI text-embedding-3-small)[/bold cyan]")
    embedder = Embedder(batch_size=batch_size)
    texts = [c.text for c in chunks]
    total_batches = (len(texts) + batch_size - 1) // batch_size

    embeddings_list: list[np.ndarray] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Embedding {len(texts)} chunks in {total_batches} batches...[/cyan]",
            total=len(texts),
        )
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            batch_emb, tokens = embedder._embed_batch(batch_texts)
            embedder.total_tokens_used += tokens
            embedder.total_api_calls += 1
            embeddings_list.extend(batch_emb)
            progress.advance(task, advance=len(batch_texts))

    embeddings = np.array(embeddings_list, dtype=np.float32)
    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = (embeddings / norms).astype(np.float32)

    usage = embedder.usage_summary()
    console.print(f"[green][OK] {len(embeddings)} embeddings generated[/green]")
    console.print(f"  Model           : {usage['model']}")
    console.print(f"  Total tokens    : {usage['total_tokens_used']:,}")
    console.print(f"  Estimated cost  : ${usage['estimated_cost_usd']:.4f} USD")

    # Save raw embeddings
    np.save(str(chunks_path / "embeddings.npy"), embeddings)
    logger.info(f"[Phase2] Embeddings saved -> {chunks_path}/embeddings.npy")

    # -- Step 4: Build index ---------------------------------------------------
    console.print("\n[bold cyan]Step 4 / 4 - Building FAISS + BM25 dual index[/bold cyan]")
    index = FAISSIndex(dimensions=embeddings.shape[1])
    index.build_from_chunks(chunks, embeddings)
    index.save(Path(index_dir))

    console.print(f"[green][OK] Index built: {index.faiss_index.ntotal} vectors[/green]")
    console.print(f"  FAISS index     : {index_dir}/faiss.index")
    console.print(f"  Chunk metadata  : {index_dir}/chunks.json")
    console.print(f"  BM25 corpus     : {index_dir}/bm25_corpus.json")

    # -- Checkpoint ------------------------------------------------------------
    completed_at = datetime.utcnow()
    checkpoint = {
        "phase": "II",
        "status": "complete",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "documents_loaded": len(docs),
        "chunks_produced": len(chunks),
        "chunk_strategy_breakdown": strategy_counts,
        "chunks_by_source": source_chunk_counts,
        "index_vectors": index.faiss_index.ntotal,
        "embedding_model": usage["model"],
        "tokens_used": usage["total_tokens_used"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
        "index_dir": index_dir,
        "next_phase": "III - Retrieval, Reranking & Generation",
        "next_command": "python -m src.main phase3",
    }
    Path("data/checkpoint_phase2.json").write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8"
    )

    # -- Summary banner --------------------------------------------------------
    console.print()
    console.print(
        Panel(
            "[bold green]Phase II Complete[/bold green]\n\n"
            f"  Documents   : {len(docs):,}\n"
            f"  Chunks      : {len(chunks):,}\n"
            f"  Vectors     : {index.faiss_index.ntotal:,}\n"
            f"  Tokens used : {usage['total_tokens_used']:,}\n"
            f"  Cost        : ${usage['estimated_cost_usd']:.4f} USD\n\n"
            "Index ready for Phase III retrieval.\n"
            "Run: [bold]python -m src.main phase3[/bold]",
            box=box.DOUBLE_EDGE,
            border_style="green",
            expand=False,
        )
    )
    logger.info(f"[Phase2] Checkpoint written -> data/checkpoint_phase2.json")
    return checkpoint
