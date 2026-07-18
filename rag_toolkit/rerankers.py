"""Cross-encoder rerankers. Return re-scored, re-ordered results.

Rerankers are slower than bi-encoders (they run the LLM on every query-doc
pair) but produce much better precision. Typical pattern: overfetch 20-40
candidates with a bi-encoder, then rerank down to the final 5.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from rag_toolkit.types import SearchResult


class Reranker(Protocol):
    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]: ...


class FlagReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder.

    Small (~600MB), multilingual, strong for the size. Runs on CPU or GPU.
    First call downloads the model. Set `timeout_seconds` to gracefully
    degrade to the un-reranked order on slow machines.

    Requires `pip install rag-toolkit[bge]`.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        try:
            from FlagEmbedding import (
                FlagReranker as _FlagReranker,  # type: ignore[import-not-found]
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install with `pip install rag-toolkit[bge]`.") from e

        self._model = _FlagReranker(model_name, use_fp16=use_fp16, device=device)
        self.timeout_seconds = timeout_seconds

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.text) for r in results]

        try:
            scores = await asyncio.wait_for(
                asyncio.to_thread(self._model.compute_score, pairs),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            # Graceful degradation: return the un-reranked candidates.
            return results[:top_k]

        # `compute_score` returns a scalar for single pairs, list for many.
        score_list = scores if isinstance(scores, list) else [scores]
        rescored = [
            SearchResult(chunk=r.chunk, score=float(s))
            for r, s in zip(results, score_list, strict=True)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]


class IdentityReranker:
    """No-op reranker. Useful as a default / for tests."""

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        return results[:top_k]
