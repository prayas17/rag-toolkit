"""Chunker unit tests."""

import pytest

from rag_toolkit.chunkers import RecursiveChunker, SentenceChunker
from rag_toolkit.types import Document


class TestSentenceChunker:
    def test_short_text_single_chunk(self) -> None:
        doc = Document(text="One sentence only.", id="d1")
        chunks = SentenceChunker(chunk_size=100).chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "One sentence only."
        assert chunks[0].doc_id == "d1"
        assert chunks[0].index == 0

    def test_packs_sentences_up_to_size(self) -> None:
        text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. Juliet kilo lima."
        chunks = SentenceChunker(chunk_size=40, overlap=0).chunk(Document(text=text))
        # Each sentence is ~20 chars; two sentences ~40 chars fit; then rolls over.
        assert len(chunks) >= 2
        # No chunk exceeds size by more than the length of a single sentence + join char
        assert all(len(c.text) <= 45 for c in chunks)

    def test_overlap_carries_tail(self) -> None:
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = SentenceChunker(chunk_size=40, overlap=25).chunk(Document(text=text))
        # Confirm sequential chunks share at least one sentence in overlap
        if len(chunks) >= 2:
            tail_of_first = chunks[0].text.split(". ")[-1]
            assert tail_of_first.rstrip(".") in chunks[1].text or len(chunks) == 1

    def test_invalid_config_raises(self) -> None:
        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=0)
        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=100, overlap=100)
        with pytest.raises(ValueError):
            SentenceChunker(chunk_size=100, overlap=-1)

    def test_empty_text_returns_empty(self) -> None:
        assert SentenceChunker().chunk(Document(text="")) == []
        assert SentenceChunker().chunk(Document(text="   \n\n  ")) == []

    def test_preserves_metadata(self) -> None:
        doc = Document(text="a. b. c.", id="d1", metadata={"source": "test.md"})
        chunks = SentenceChunker(chunk_size=200, overlap=0).chunk(doc)
        for c in chunks:
            assert c.metadata == {"source": "test.md"}


class TestRecursiveChunker:
    def test_paragraph_split(self) -> None:
        text = "Para one line one.\nPara one line two.\n\nPara two line one.\n\nPara three."
        chunks = RecursiveChunker(chunk_size=30, overlap=0).chunk(Document(text=text))
        assert len(chunks) >= 2

    def test_long_word_forced_split(self) -> None:
        text = "x" * 1000
        chunks = RecursiveChunker(chunk_size=100, overlap=0).chunk(Document(text=text))
        # Char-fallback should split every 100 chars
        assert len(chunks) == 10
        assert all(len(c.text) == 100 for c in chunks)

    def test_invalid_config_raises(self) -> None:
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=0)
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=100, overlap=100)

    def test_chunk_ids_stable(self) -> None:
        doc = Document(text="One. Two. Three.", id="doc42")
        chunks = RecursiveChunker(chunk_size=200, overlap=0).chunk(doc)
        for c in chunks:
            assert c.id.startswith("doc42:")
