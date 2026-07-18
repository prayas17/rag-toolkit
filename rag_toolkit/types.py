"""Core data types — plain dataclasses, no framework baggage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A source document before chunking."""

    text: str
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk of text produced by a chunker, ready for embedding."""

    text: str
    index: int
    doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable ID: `{doc_id}:{index}` if doc_id present, else just index."""
        return f"{self.doc_id}:{self.index}" if self.doc_id else str(self.index)


@dataclass
class Embedding:
    """Dense embedding vector plus optional sparse component for hybrid search.

    `sparse` is a token-id → weight mapping (e.g., BM25-style or SPLADE weights).
    """

    dense: list[float]
    sparse: dict[int, float] | None = None

    @property
    def dim(self) -> int:
        return len(self.dense)


@dataclass
class SearchResult:
    """A retrieval hit with score. Scores are comparable within one retriever
    call but not across retrievers (dense cosine vs cross-encoder score).
    """

    chunk: Chunk
    score: float

    def __lt__(self, other: SearchResult) -> bool:
        # sort by score descending
        return self.score > other.score
