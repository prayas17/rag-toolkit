"""Text chunkers. Produce `Chunk` objects from raw text.

Two strategies:

- `RecursiveChunker` — splits on a priority list of separators (paragraphs,
  sentences, words). Preserves structural boundaries when possible.
- `SentenceChunker` — packs whole sentences into fixed-size chunks with
  overlap. Best when you want stable retrieval units and don't care about
  paragraph structure.
"""

from __future__ import annotations

import re
from typing import Protocol

from rag_toolkit.types import Chunk, Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class Chunker(Protocol):
    """A chunker turns a `Document` into a list of `Chunk` objects."""

    def chunk(self, doc: Document) -> list[Chunk]: ...


class SentenceChunker:
    """Pack whole sentences into chunks of ~`chunk_size` characters with `overlap`.

    A sentence longer than `chunk_size` is emitted on its own (never split
    mid-sentence). This trades chunk-size uniformity for retrieval quality.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = _split_sentences(doc.text)
        chunks: list[str] = []
        buf: list[str] = []
        buf_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if buf_len + sent_len + 1 > self.chunk_size and buf:
                chunks.append(" ".join(buf))
                # keep a tail as overlap
                buf, buf_len = _tail_by_chars(buf, self.overlap)
            buf.append(sent)
            buf_len += sent_len + 1  # +1 for the join space

        if buf:
            chunks.append(" ".join(buf))

        return [
            Chunk(text=text, index=i, doc_id=doc.id, metadata=dict(doc.metadata))
            for i, text in enumerate(chunks)
        ]


class RecursiveChunker:
    """Recursively split on a priority list of separators until each piece fits.

    Similar in spirit to LangChain's RecursiveCharacterTextSplitter but with
    fewer moving parts and no framework dependency.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or _DEFAULT_SEPARATORS

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self._split(doc.text, self.separators)
        merged = _merge_with_overlap(pieces, self.chunk_size, self.overlap)
        return [
            Chunk(text=text, index=i, doc_id=doc.id, metadata=dict(doc.metadata))
            for i, text in enumerate(merged)
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators or len(text) <= self.chunk_size:
            return [text] if text else []

        sep, *rest = separators
        if sep == "":
            # Char-level fallback: force-split into chunk_size pieces.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        pieces = text.split(sep) if sep else [text]
        out: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._split(piece, rest))
        return out


# ---------- helpers ----------


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _tail_by_chars(buf: list[str], budget: int) -> tuple[list[str], int]:
    """Return the trailing items from `buf` whose combined length ≤ budget."""
    if budget <= 0:
        return [], 0
    tail: list[str] = []
    total = 0
    for item in reversed(buf):
        if total + len(item) + 1 > budget:
            break
        tail.insert(0, item)
        total += len(item) + 1
    return tail, total


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Merge small pieces into ~chunk_size chunks with character overlap."""
    if not pieces:
        return []
    merged: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        if len(current) + len(piece) + 1 <= chunk_size:
            current = f"{current} {piece}"
        else:
            merged.append(current)
            # carry overlap
            current = f"{current[-overlap:]} {piece}" if overlap else piece
    if current:
        merged.append(current)
    return merged
