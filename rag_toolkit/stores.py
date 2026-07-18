"""Vector stores. Both dense-only and hybrid dense+sparse retrieval.

- `InMemoryStore` — no external deps, good for tests and toy datasets. Naive
  cosine-similarity brute-force search. Not suitable for large corpora.
- `QdrantStore` — production-grade, supports named vectors (dense + sparse)
  with RRF fusion for hybrid queries.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from rag_toolkit.types import Chunk, Embedding, SearchResult


class VectorStore(Protocol):
    async def upsert(self, chunks: list[Chunk], embeddings: list[Embedding]) -> None: ...

    async def search_dense(
        self, query: Embedding, top_k: int = 10, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...

    async def search_hybrid(
        self, query: Embedding, top_k: int = 10, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...


@dataclass
class _StoredItem:
    chunk: Chunk
    embedding: Embedding


class InMemoryStore:
    """Brute-force in-memory store. O(n) per query.

    Fine for <10k chunks or for testing. Use `QdrantStore` in production.
    """

    def __init__(self) -> None:
        self._items: list[_StoredItem] = []

    async def upsert(self, chunks: list[Chunk], embeddings: list[Embedding]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be same length")
        self._items.extend(
            _StoredItem(chunk=c, embedding=e) for c, e in zip(chunks, embeddings, strict=True)
        )

    async def search_dense(
        self, query: Embedding, top_k: int = 10, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        items = self._filter(filter)
        scored = [
            SearchResult(chunk=item.chunk, score=_cosine(query.dense, item.embedding.dense))
            for item in items
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    async def search_hybrid(
        self, query: Embedding, top_k: int = 10, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Dense + sparse fusion via Reciprocal Rank Fusion."""
        items = self._filter(filter)
        dense_ranked = sorted(
            items, key=lambda it: _cosine(query.dense, it.embedding.dense), reverse=True
        )
        sparse_ranked = sorted(
            items,
            key=lambda it: _sparse_dot(query.sparse or {}, it.embedding.sparse or {}),
            reverse=True,
        )
        fused = _rrf([dense_ranked, sparse_ranked], k=60)
        return [SearchResult(chunk=it.chunk, score=score) for it, score in fused[:top_k]]

    def _filter(self, filter: dict[str, Any] | None) -> list[_StoredItem]:
        if not filter:
            return self._items
        return [it for it in self._items if _matches(it.chunk.metadata, filter)]

    def __len__(self) -> int:
        return len(self._items)


class QdrantStore:
    """Qdrant-backed store with named vectors for hybrid search.

    Collection layout uses two named vectors: `dense` (cosine) and `sparse`.
    RRF fusion is performed server-side via Qdrant's `query_points` API.

    Requires `pip install rag-toolkit[qdrant]`.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection: str = "documents",
        dense_dim: int = 1024,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install with `pip install rag-toolkit[qdrant]`.") from e

        self._client = AsyncQdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.dense_dim = dense_dim
        self._ensured = False

    async def upsert(self, chunks: list[Chunk], embeddings: list[Embedding]) -> None:
        await self._ensure_collection()
        from qdrant_client.http import models as qm  # type: ignore[import-not-found]

        points = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            vec: dict[str, Any] = {"dense": emb.dense}
            if emb.sparse:
                vec["sparse"] = qm.SparseVector(
                    indices=list(emb.sparse.keys()),
                    values=list(emb.sparse.values()),
                )
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={"text": chunk.text, "doc_id": chunk.doc_id, **chunk.metadata},
                )
            )
        await self._client.upsert(collection_name=self.collection, points=points)

    async def search_dense(
        self, query: Embedding, top_k: int = 10, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        await self._ensure_collection()
        hits = await self._client.query_points(
            collection_name=self.collection,
            query=query.dense,
            using="dense",
            limit=top_k,
            query_filter=_to_qdrant_filter(filter),
            with_payload=True,
        )
        return [_hit_to_result(h) for h in hits.points]

    async def search_hybrid(
        self, query: Embedding, top_k: int = 10, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        await self._ensure_collection()
        from qdrant_client.http import models as qm  # type: ignore[import-not-found]

        if not query.sparse:
            return await self.search_dense(query, top_k, filter)

        prefetch = [
            qm.Prefetch(query=query.dense, using="dense", limit=top_k * 4),
            qm.Prefetch(
                query=qm.SparseVector(
                    indices=list(query.sparse.keys()), values=list(query.sparse.values())
                ),
                using="sparse",
                limit=top_k * 4,
            ),
        ]
        hits = await self._client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            query_filter=_to_qdrant_filter(filter),
            with_payload=True,
        )
        return [_hit_to_result(h) for h in hits.points]

    async def _ensure_collection(self) -> None:
        if self._ensured:
            return
        from qdrant_client.http import models as qm  # type: ignore[import-not-found]

        collections = await self._client.get_collections()
        if not any(c.name == self.collection for c in collections.collections):
            await self._client.create_collection(
                collection_name=self.collection,
                vectors_config={"dense": qm.VectorParams(size=self.dense_dim, distance=qm.Distance.COSINE)},
                sparse_vectors_config={"sparse": qm.SparseVectorParams()},
            )
        self._ensured = True


# ---------- helpers ----------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _sparse_dot(a: dict[int, float], b: dict[int, float]) -> float:
    small, large = (a, b) if len(a) < len(b) else (b, a)
    return sum(w * large.get(k, 0.0) for k, w in small.items())


def _rrf(rankings: list[list[_StoredItem]], k: int = 60) -> list[tuple[_StoredItem, float]]:
    """Reciprocal Rank Fusion — combine multiple ranked lists into one."""
    scores: dict[int, float] = {}
    seen: dict[int, _StoredItem] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            key = id(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            seen[key] = item
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(seen[key], score) for key, score in ranked]


def _matches(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
    return all(metadata.get(k) == v for k, v in filter.items())


def _to_qdrant_filter(filter: dict[str, Any] | None) -> Any:  # pragma: no cover
    if not filter:
        return None
    from qdrant_client.http import models as qm  # type: ignore[import-not-found]

    return qm.Filter(
        must=[qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in filter.items()]
    )


def _hit_to_result(hit: Any) -> SearchResult:  # pragma: no cover
    payload = hit.payload or {}
    text = payload.pop("text", "")
    doc_id = payload.pop("doc_id", None)
    chunk = Chunk(text=text, index=0, doc_id=doc_id, metadata=payload)
    return SearchResult(chunk=chunk, score=float(hit.score))
