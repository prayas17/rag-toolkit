"""Retrievers glue an embedder to a store, with optional reranking.

- `DenseRetriever` — embed query, dense search, return top-k.
- `HybridRetriever` — dense + sparse fusion (RRF at the store level).
- `RerankedRetriever` — wrap any retriever, overfetch, then rerank with a
  cross-encoder for materially better precision at k.
"""

from __future__ import annotations

from typing import Any, Protocol

from rag_toolkit.embedders import Embedder
from rag_toolkit.rerankers import Reranker
from rag_toolkit.stores import VectorStore
from rag_toolkit.types import SearchResult


class Retriever(Protocol):
    async def retrieve(
        self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    async def retrieve(
        self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        emb = await self.embedder.embed_query(query)
        return await self.store.search_dense(emb, top_k=top_k, filter=filter)


class HybridRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    async def retrieve(
        self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        emb = await self.embedder.embed_query(query)
        return await self.store.search_hybrid(emb, top_k=top_k, filter=filter)


class RerankedRetriever:
    """Overfetch → rerank pattern. Typically improves precision@k by 5-20 points.

    `overfetch_multiplier` controls how many candidates to pull from the base
    retriever before rerank. 4x is a reasonable default; larger = better
    ceiling, slower response.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        overfetch_multiplier: int = 4,
    ) -> None:
        if overfetch_multiplier < 1:
            raise ValueError("overfetch_multiplier must be >= 1")
        self.base = base
        self.reranker = reranker
        self.overfetch_multiplier = overfetch_multiplier

    async def retrieve(
        self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        candidates = await self.base.retrieve(
            query, top_k=top_k * self.overfetch_multiplier, filter=filter
        )
        if not candidates:
            return []
        return await self.reranker.rerank(query, candidates, top_k=top_k)
