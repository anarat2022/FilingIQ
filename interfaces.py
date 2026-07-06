"""
Abstract interfaces for the RAG pipeline components.

Every concrete implementation (Indexer, Datastore, Retriever, ResponseGenerator,
Evaluator) is written against one of these contracts. This is what lets you swap
OpenAI for Gemini, or ChromaDB for LanceDB, without touching the pipeline code
that wires everything together (see main.py).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single unit of retrievable text, plus where it came from."""
    id: str
    text: str
    source: str          # e.g. "apple_10k_2024.txt"
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A Chunk plus how relevant it was to a specific query."""
    chunk: Chunk
    score: float          # similarity score, higher = more relevant


@dataclass
class RagAnswer:
    """The final output of the pipeline for one question."""
    question: str
    answer: str
    sources: list[RetrievedChunk]
    grounded: bool         # False if retrieval found nothing usable


class BaseIndexer(ABC):
    """Turns raw documents into Chunks."""

    @abstractmethod
    def load_and_chunk(self, path: str) -> list[Chunk]:
        """Read a file or directory and split it into Chunks."""
        raise NotImplementedError


class BaseDatastore(ABC):
    """Embeds and stores Chunks; supports similarity search."""

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> int:
        """Embed and persist chunks. Returns number of chunks stored."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks to the query."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored data."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Number of chunks currently stored."""
        raise NotImplementedError


class BaseRetriever(ABC):
    """Wraps a Datastore with retrieval-specific logic (thresholds, filtering)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        raise NotImplementedError


class BaseResponseGenerator(ABC):
    """Turns a query + retrieved chunks into a grounded answer."""

    @abstractmethod
    def generate(self, question: str, context: list[RetrievedChunk]) -> RagAnswer:
        raise NotImplementedError


class BaseEvaluator(ABC):
    """Scores generated answers against expected answers."""

    @abstractmethod
    def evaluate(self, question: str, expected: str, actual: str) -> dict:
        """Return a dict with at least {'score': float, 'reasoning': str}."""
        raise NotImplementedError
