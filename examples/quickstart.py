"""30-second quickstart. No API keys, no model downloads.

    python examples/quickstart.py
"""

import asyncio

from rag_toolkit import (
    Document,
    FakeEmbedder,
    InMemoryStore,
    RAGPipeline,
    SentenceChunker,
)


async def main() -> None:
    pipeline = RAGPipeline(
        chunker=SentenceChunker(chunk_size=256, overlap=32),
        embedder=FakeEmbedder(dim=256),
        store=InMemoryStore(),
        mode="hybrid",
    )

    docs = [
        Document(
            id="langgraph",
            text=(
                "LangGraph is a library for building stateful, multi-actor "
                "applications with LLMs. It exposes an explicit state graph, "
                "making complex agent orchestration debuggable."
            ),
        ),
        Document(
            id="qdrant",
            text=(
                "Qdrant is a vector search engine written in Rust. It supports "
                "named vectors, sparse vectors, and reciprocal-rank-fusion "
                "hybrid search out of the box."
            ),
        ),
        Document(
            id="rag",
            text=(
                "Retrieval-Augmented Generation grounds LLM responses in "
                "external knowledge. A retriever pulls relevant context; "
                "the LLM generates the answer conditioned on it."
            ),
        ),
    ]

    n = await pipeline.ingest(docs)
    print(f"Ingested {n} chunks.\n")

    for query in ["what is qdrant", "how does RAG work", "stateful agents"]:
        hits = await pipeline.retrieve(query, top_k=1)
        top = hits[0]
        print(f"Q: {query}")
        print(f"  -> doc={top.chunk.doc_id}  score={top.score:.3f}")
        print(f"  {top.chunk.text[:80]}...\n")


if __name__ == "__main__":
    asyncio.run(main())
