"""High-level orchestration — the class most users will start with."""

from __future__ import annotations

from typing import Any

from rag_toolkit.chunkers import Chunker
from rag_toolkit.embedders import Embedder
from rag_toolkit.rerankers import Reranker
from rag_toolkit.retrievers import DenseRetriever, HybridRetriever, RerankedRetriever
from rag_toolkit.stores import VectorStore
from rag_toolkit.types import Document, SearchResult


class RAGPipeline:
    """End-to-end ingest → retrieve.

    Composition, not inheritance — swap any component. The `mode` argument
    picks a sensible retriever without you having to wire one manually.

    Example
    -------
    >>> import asyncio
    >>> from rag_toolkit import RAGPipeline, SentenceChunker, FakeEmbedder, InMemoryStore, Document
    >>> pipe = RAGPipeline(
    ...     chunker=SentenceChunker(chunk_size=256, overlap=32),
    ...     embedder=FakeEmbedder(),
    ...     store=InMemoryStore(),
    ... )
    >>> asyncio.run(pipe.ingest([Document(text="The quick brown fox jumps over the lazy dog.")]))
    >>> hits = asyncio.run(pipe.retrieve("brown fox", top_k=1))
    >>> hits[0].chunk.text
    'The quick brown fox jumps over the lazy dog.'
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
        mode: str = "hybrid",
        overfetch_multiplier: int = 4,
    ) -> None:
        if mode not in {"dense", "hybrid"}:
            raise ValueError("mode must be 'dense' or 'hybrid'")

        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.mode = mode

        base = HybridRetriever(embedder, store) if mode == "hybrid" else DenseRetriever(embedder, store)
        self._retriever = (
            RerankedRetriever(base, reranker, overfetch_multiplier=overfetch_multiplier)
            if reranker
            else base
        )

    async def ingest(self, docs: list[Document], *, batch_size: int = 32) -> int:
        """Chunk, embed, and upsert. Returns the number of chunks written."""
        all_chunks = [c for doc in docs for c in self.chunker.chunk(doc)]
        if not all_chunks:
            return 0

        total = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            embeddings = await self.embedder.embed([c.text for c in batch])
            await self.store.upsert(batch, embeddings)
            total += len(batch)
        return total

    async def retrieve(
        self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        return await self._retriever.retrieve(query, top_k=top_k, filter=filter)
