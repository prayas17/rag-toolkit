<h1 align="center">rag-toolkit</h1>

<p align="center">
  <b>Production-grade hybrid retrieval + reranking for RAG.</b><br/>
  Pluggable chunkers · embedders · vector stores · rerankers. Batteries included.
</p>

<p align="center">
  <a href="https://github.com/prayas17/rag-toolkit/actions/workflows/tests.yml"><img src="https://github.com/prayas17/rag-toolkit/actions/workflows/tests.yml/badge.svg" alt="tests"/></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="python 3.10+"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"/>
  <img src="https://img.shields.io/badge/type--hints-strict-3178C6" alt="strict type hints"/>
</p>

---

## Why

Most RAG tutorials show you a `Chroma.from_documents(...)` one-liner. Production RAG needs:

- **Hybrid search** (dense + sparse) with reciprocal rank fusion — dense alone loses on keyword-heavy queries
- **Reranking** with a cross-encoder — the single highest-ROI move for precision@k
- **Chunking that respects sentence boundaries** — mid-sentence splits kill retrieval quality
- **Async I/O** — embedders and stores are network-bound; blocking your event loop is not okay
- **Graceful degradation** — a slow reranker shouldn't 30s-timeout the user's request
- **Zero-lock-in** — swap OpenAI for BGE, swap Qdrant for in-memory, without rewriting

`rag-toolkit` is a small (~1000 LOC), strict-typed, async-native library that gives you all of this without pulling in a framework.

## Install

```bash
pip install rag-toolkit                        # core only
pip install "rag-toolkit[bge]"                 # + BGE-M3 embedder / reranker
pip install "rag-toolkit[openai]"              # + OpenAI embeddings
pip install "rag-toolkit[qdrant]"              # + Qdrant vector store
pip install "rag-toolkit[all]"                 # everything
```

## Quickstart

Zero API keys, zero model downloads — runs anywhere in <1s:

```python
import asyncio
from rag_toolkit import (
    RAGPipeline, SentenceChunker, FakeEmbedder, InMemoryStore, Document
)

async def main():
    pipe = RAGPipeline(
        chunker=SentenceChunker(chunk_size=256, overlap=32),
        embedder=FakeEmbedder(),
        store=InMemoryStore(),
        mode="hybrid",
    )

    await pipe.ingest([
        Document(id="rag", text="RAG grounds LLM answers in retrieved context."),
        Document(id="qdrant", text="Qdrant supports hybrid vector search via RRF."),
    ])

    hits = await pipe.retrieve("hybrid search", top_k=1)
    print(hits[0].chunk.text)

asyncio.run(main())
```

## Production setup

```python
from rag_toolkit import RAGPipeline, SentenceChunker, BGEM3Embedder, QdrantStore
from rag_toolkit.rerankers import FlagReranker

pipe = RAGPipeline(
    chunker=SentenceChunker(chunk_size=512, overlap=64),
    embedder=BGEM3Embedder(use_fp16=True),
    store=QdrantStore(url="http://localhost:6333", collection="docs", dense_dim=1024),
    reranker=FlagReranker(timeout_seconds=3.0),   # graceful degradation
    mode="hybrid",
    overfetch_multiplier=4,                        # rerank 20, return top 5
)
```

## Architecture

```
     Document
        │
        ▼
   ┌─────────┐         ┌──────────┐        ┌─────────────┐
   │ Chunker │──────▶  │ Embedder │──────▶ │ VectorStore │
   └─────────┘         └──────────┘        └─────────────┘
                                                    │
                              ┌─── query ───────────┘
                              ▼
                        ┌──────────┐        ┌──────────┐
                        │Retriever │──────▶ │ Reranker │────▶ SearchResult[]
                        │  hybrid  │        │(optional)│
                        └──────────┘        └──────────┘
```

Every layer is a `Protocol` — bring your own implementation.

## Components

| Component | Options | Notes |
|-----------|---------|-------|
| **Chunker** | `SentenceChunker`, `RecursiveChunker` | Sentence-aware; no LLM dependency |
| **Embedder** | `BGEM3Embedder`, `OpenAIEmbedder`, `FakeEmbedder` | Async; BGE-M3 returns dense + sparse |
| **Store** | `QdrantStore`, `InMemoryStore` | Qdrant does server-side RRF hybrid |
| **Reranker** | `FlagReranker`, `IdentityReranker` | Cross-encoder; timeout → graceful fallback |
| **Retriever** | `DenseRetriever`, `HybridRetriever`, `RerankedRetriever` | Compose freely |
| **Pipeline** | `RAGPipeline` | High-level convenience; use it when the defaults fit |

## Design decisions

**Why async everywhere?** Embedders and stores are network- or GPU-bound. Sync APIs force you to spawn threads at the boundary anyway; async keeps the whole call chain honest.

**Why Reciprocal Rank Fusion (RRF)?** It's parameter-free, empirically strong, and Qdrant supports it server-side. Weighted-sum fusion needs tuning and rarely beats RRF meaningfully.

**Why BGE-M3 as the default?** Best open-source model per dollar/latency as of 2026. Multilingual, produces both dense and sparse vectors from one forward pass, ~2GB.

**Why a timeout on the reranker?** Cross-encoders are slow. A hung reranker shouldn't take down user requests — return the un-reranked candidates instead. Better slightly-worse precision than a 500.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Tests use `FakeEmbedder` + `InMemoryStore` — no network calls, no model downloads. Runs in <1s.

## Roadmap

- [ ] `.ingest_from()` helpers for URLs, PDFs, HTML
- [ ] SPLADE sparse embedder
- [ ] Cohere reranker
- [ ] Weaviate + pgvector stores
- [ ] Sync API shim for scripts

## License

MIT — see [LICENSE](./LICENSE).

## Author

Built by [Prayas Jain](https://github.com/prayas17) — Forward Deployed AI Engineer. Extracted from patterns proven in production RAG systems.

Open to freelance & contract work on LLM/RAG systems: [prayas1711@gmail.com](mailto:prayas1711@gmail.com) · [Portfolio](https://prayas17.github.io/portfolio)
