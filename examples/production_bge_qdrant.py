"""Production setup: BGE-M3 embeddings + Qdrant + cross-encoder reranker.

Prereqs:
    pip install "rag-toolkit[bge,qdrant]"
    docker run -p 6333:6333 qdrant/qdrant  # or a hosted Qdrant instance

    python examples/production_bge_qdrant.py
"""

import asyncio

from rag_toolkit import (
    BGEM3Embedder,
    Document,
    QdrantStore,
    RAGPipeline,
    SentenceChunker,
)
from rag_toolkit.rerankers import FlagReranker


async def main() -> None:
    pipeline = RAGPipeline(
        chunker=SentenceChunker(chunk_size=512, overlap=64),
        embedder=BGEM3Embedder(use_fp16=True),
        store=QdrantStore(
            url="http://localhost:6333",
            collection="prod_docs",
            dense_dim=1024,  # bge-m3 dimensionality
        ),
        reranker=FlagReranker(use_fp16=True, timeout_seconds=3.0),
        mode="hybrid",
        overfetch_multiplier=4,
    )

    docs = [
        Document(
            id="rag-101",
            text=(
                "Retrieval-Augmented Generation combines a retriever and an LLM. "
                "The retriever pulls relevant passages from a knowledge base; "
                "the LLM generates an answer conditioned on those passages. "
                "This grounds outputs in verifiable sources and reduces "
                "hallucination on domain-specific questions."
            ),
            metadata={"source": "docs/rag-101.md"},
        ),
    ]

    n = await pipeline.ingest(docs)
    print(f"Ingested {n} chunks into Qdrant.")

    hits = await pipeline.retrieve("How does RAG reduce hallucination?", top_k=3)
    for i, hit in enumerate(hits, 1):
        print(f"\n#{i}  score={hit.score:.3f}  doc={hit.chunk.doc_id}")
        print(f"  {hit.chunk.text[:120]}...")


if __name__ == "__main__":
    asyncio.run(main())
