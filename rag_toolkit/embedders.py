"""Embedders — turn text into vectors.

- `BGEM3Embedder` — BAAI/bge-m3, dense + sparse (BM25-like weights). Best
  quality-for-cost open model as of writing.
- `OpenAIEmbedder` — OpenAI's text-embedding-3-* families. Dense only.
- `FakeEmbedder` — deterministic hash-based vectors. Zero deps, used by tests
  and for local dev without model downloads.

External deps are lazy-imported so `import rag_toolkit` doesn't pull them.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Any, Protocol

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_toolkit.types import Embedding


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[Embedding]: ...

    async def embed_query(self, text: str) -> Embedding: ...


class BGEM3Embedder:
    """BAAI/bge-m3 dense + sparse embeddings.

    Requires `pip install rag-toolkit[bge]`. First call downloads the model
    (~2GB) and takes 10-30s. Batches are processed on-device.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str | None = None,
        max_length: int = 8192,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install with `pip install rag-toolkit[bge]`.") from e

        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
        self.max_length = max_length

    async def embed(self, texts: list[str]) -> list[Embedding]:
        # BGEM3 is CPU/GPU-bound, not I/O. Run in a worker thread so we
        # don't block the event loop.
        return await asyncio.to_thread(self._embed_sync, texts, sparse=True)

    async def embed_query(self, text: str) -> Embedding:
        result = await asyncio.to_thread(self._embed_sync, [text], sparse=True)
        return result[0]

    def _embed_sync(self, texts: list[str], sparse: bool) -> list[Embedding]:
        out = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=sparse,
            return_colbert_vecs=False,
            max_length=self.max_length,
        )
        dense = out["dense_vecs"]
        sparse_lists = out.get("lexical_weights", [None] * len(texts)) if sparse else [None] * len(texts)
        return [
            Embedding(dense=list(map(float, d)), sparse={int(k): float(v) for k, v in (s or {}).items()} or None)
            for d, s in zip(dense, sparse_lists, strict=True)
        ]


class OpenAIEmbedder:
    """OpenAI embeddings — dense only.

    Requires `pip install rag-toolkit[openai]` and an `OPENAI_API_KEY`.
    Batches of >100 items are chunked automatically.
    """

    _MAX_BATCH = 100

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install with `pip install rag-toolkit[openai]`.") from e

        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def embed(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        out: list[Embedding] = []
        for i in range(0, len(texts), self._MAX_BATCH):
            batch = texts[i : i + self._MAX_BATCH]
            resp = await self._call(batch)
            out.extend(Embedding(dense=item.embedding) for item in resp.data)
        return out

    async def embed_query(self, text: str) -> Embedding:
        result = await self.embed([text])
        return result[0]

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call(self, batch: list[str]) -> Any:
        try:
            return await self._client.embeddings.create(model=self.model, input=batch)
        except RetryError as e:  # pragma: no cover
            raise RuntimeError("OpenAI embeddings failed after retries") from e


class FakeEmbedder:
    """Deterministic hash-based embeddings. No network, no models — perfect
    for tests and for building/inspecting pipelines locally.

    Same text → same vector. Similarity is loosely related to substring
    overlap (via hashed n-grams into sparse buckets).
    """

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[Embedding]:
        return [self._embed_one(t) for t in texts]

    async def embed_query(self, text: str) -> Embedding:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> Embedding:
        text = text.lower()
        # Sparse: hashed 3-grams of tokens
        sparse: dict[int, float] = {}
        tokens = text.split()
        for n in (1, 2):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                h = int(hashlib.blake2s(ngram.encode()).hexdigest(), 16) % 100_000
                sparse[h] = sparse.get(h, 0.0) + 1.0

        # Dense: fold sparse into `dim` buckets then L2-normalize
        vec = [0.0] * self.dim
        for h, w in sparse.items():
            vec[h % self.dim] += w
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return Embedding(dense=[v / norm for v in vec], sparse=sparse or None)
