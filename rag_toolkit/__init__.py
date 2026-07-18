"""rag-toolkit — production-grade hybrid retrieval for RAG."""

from rag_toolkit.chunkers import RecursiveChunker, SentenceChunker
from rag_toolkit.embedders import BGEM3Embedder, FakeEmbedder, OpenAIEmbedder
from rag_toolkit.pipeline import RAGPipeline
from rag_toolkit.rerankers import FlagReranker
from rag_toolkit.retrievers import DenseRetriever, HybridRetriever, RerankedRetriever
from rag_toolkit.stores import InMemoryStore, QdrantStore
from rag_toolkit.types import Chunk, Document, Embedding, SearchResult

__version__ = "0.1.0"

__all__ = [
    "BGEM3Embedder",
    "Chunk",
    "DenseRetriever",
    "Document",
    "Embedding",
    "FakeEmbedder",
    "FlagReranker",
    "HybridRetriever",
    "InMemoryStore",
    "OpenAIEmbedder",
    "QdrantStore",
    "RAGPipeline",
    "RecursiveChunker",
    "RerankedRetriever",
    "SearchResult",
    "SentenceChunker",
]
