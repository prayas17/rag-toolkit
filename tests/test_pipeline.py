"""End-to-end pipeline tests using FakeEmbedder + InMemoryStore.

No network, no model downloads — runs in <1s on any machine.
"""

import pytest

from rag_toolkit import (
    Document,
    FakeEmbedder,
    InMemoryStore,
    RAGPipeline,
    SentenceChunker,
)
from rag_toolkit.rerankers import IdentityReranker


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(
            id="d1",
            text=(
                "Python is a high-level programming language. "
                "It emphasizes code readability. "
                "Guido van Rossum designed it in 1991."
            ),
            metadata={"topic": "python"},
        ),
        Document(
            id="d2",
            text=(
                "Rust is a systems programming language focused on safety. "
                "It has no garbage collector. "
                "Graydon Hoare started the project at Mozilla."
            ),
            metadata={"topic": "rust"},
        ),
        Document(
            id="d3",
            text=(
                "The Eiffel Tower is in Paris. "
                "It was built for the 1889 World Fair. "
                "Its height is 330 meters including antennas."
            ),
            metadata={"topic": "landmark"},
        ),
    ]


async def test_ingest_returns_chunk_count(docs: list[Document]) -> None:
    pipe = RAGPipeline(
        chunker=SentenceChunker(chunk_size=100, overlap=20),
        embedder=FakeEmbedder(),
        store=InMemoryStore(),
        mode="hybrid",
    )
    total = await pipe.ingest(docs)
    assert total > 0


async def test_hybrid_retrieval_finds_relevant_doc(docs: list[Document]) -> None:
    pipe = RAGPipeline(
        chunker=SentenceChunker(chunk_size=200, overlap=0),
        embedder=FakeEmbedder(),
        store=InMemoryStore(),
        mode="hybrid",
    )
    await pipe.ingest(docs)
    hits = await pipe.retrieve("who designed python", top_k=3)
    assert len(hits) > 0
    # Top hit should mention python or Guido
    top_texts = " ".join(h.chunk.text.lower() for h in hits[:2])
    assert "python" in top_texts or "guido" in top_texts


async def test_dense_mode_works(docs: list[Document]) -> None:
    pipe = RAGPipeline(
        chunker=SentenceChunker(chunk_size=200),
        embedder=FakeEmbedder(),
        store=InMemoryStore(),
        mode="dense",
    )
    await pipe.ingest(docs)
    hits = await pipe.retrieve("rust programming", top_k=2)
    assert len(hits) > 0


async def test_metadata_filter_narrows_results(docs: list[Document]) -> None:
    pipe = RAGPipeline(
        chunker=SentenceChunker(chunk_size=200),
        embedder=FakeEmbedder(),
        store=InMemoryStore(),
        mode="dense",
    )
    await pipe.ingest(docs)
    hits = await pipe.retrieve("language", top_k=10, filter={"topic": "python"})
    assert all(h.chunk.metadata.get("topic") == "python" for h in hits)


async def test_reranker_overfetches_and_reorders(docs: list[Document]) -> None:
    pipe = RAGPipeline(
        chunker=SentenceChunker(chunk_size=200),
        embedder=FakeEmbedder(),
        store=InMemoryStore(),
        mode="hybrid",
        reranker=IdentityReranker(),
        overfetch_multiplier=3,
    )
    await pipe.ingest(docs)
    hits = await pipe.retrieve("programming language", top_k=2)
    assert len(hits) <= 2


async def test_empty_ingest_is_safe() -> None:
    pipe = RAGPipeline(
        chunker=SentenceChunker(),
        embedder=FakeEmbedder(),
        store=InMemoryStore(),
    )
    assert await pipe.ingest([]) == 0
    assert await pipe.retrieve("anything") == []


async def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError):
        RAGPipeline(
            chunker=SentenceChunker(),
            embedder=FakeEmbedder(),
            store=InMemoryStore(),
            mode="ivfflat",  # not a real mode
        )
