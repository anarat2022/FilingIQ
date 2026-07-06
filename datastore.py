"""
Datastore: embeds Chunks and stores/searches them in a local ChromaDB
collection (persisted to disk, no server to run).

ChromaDatastore itself is provider-agnostic -- it just needs something that
implements chromadb's EmbeddingFunction protocol. Three are provided below:
GeminiEmbeddingFunction (free tier, rate-limited), OllamaEmbeddingFunction
(fully local, no limits, needs Ollama installed), and OpenAIEmbeddingFunction
(paid, from chromadb's own utils). This is the interface pattern from
src/interfaces.py in action: swapping embedding providers means adding one
small class, not touching ChromaDatastore.
"""

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from chromadb.utils import embedding_functions as chroma_embedding_functions

from src.interfaces import BaseDatastore, Chunk, RetrievedChunk
from src.retry import with_retry

DEFAULT_COLLECTION = "rag_documents"
DEFAULT_PERSIST_DIR = "./chroma_db"

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Embeds text with Google's Gemini API. Free tier: no credit card,
    but rate-limited (as low as 5 requests/minute on some models/accounts
    as of mid-2026) -- retries with backoff on 429s."""

    def __init__(self, api_key: str, model: str = GEMINI_EMBEDDING_MODEL):
        from google import genai  # imported lazily so other-provider installs don't need it

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        result = with_retry(
            lambda: self._client.models.embed_content(model=self._model, contents=texts)
        )
        return [e.values for e in result.embeddings]

    def name(self) -> str:
        return "gemini-embedding-function"

    def get_config(self) -> dict:
        return {"model": self._model}

    @staticmethod
    def build_from_config(config: dict) -> "GeminiEmbeddingFunction":
        import os

        return GeminiEmbeddingFunction(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            model=config.get("model", GEMINI_EMBEDDING_MODEL),
        )


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Embeds text with a local Ollama server -- no API key, no rate limits,
    no cost, runs entirely on your machine. Requires Ollama installed
    (https://ollama.com) and the embedding model pulled once via
    `ollama pull nomic-embed-text`."""

    def __init__(self, model: str = OLLAMA_EMBEDDING_MODEL, host: str | None = None):
        import ollama  # imported lazily so other-provider installs don't need it

        self._client = ollama.Client(host=host) if host else ollama
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:
        response = self._client.embed(model=self._model, input=list(input))
        return list(response.embeddings)

    def name(self) -> str:
        return "ollama-embedding-function"

    def get_config(self) -> dict:
        return {"model": self._model}

    @staticmethod
    def build_from_config(config: dict) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(model=config.get("model", OLLAMA_EMBEDDING_MODEL))


def build_embedding_function(provider: str, api_key: str) -> EmbeddingFunction:
    """Factory: returns the right embedding function for the chosen provider."""
    if provider == "gemini":
        return GeminiEmbeddingFunction(api_key=api_key)
    if provider == "openai":
        return chroma_embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=OPENAI_EMBEDDING_MODEL
        )
    if provider == "ollama":
        return OllamaEmbeddingFunction()
    raise ValueError(f"Unknown embedding provider: {provider!r} (expected 'gemini', 'openai', or 'ollama')")


class ChromaDatastore(BaseDatastore):
    def __init__(
        self,
        embedding_function: EmbeddingFunction,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embedding_fn = embedding_function
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, **c.metadata} for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        results = self._collection.query(query_texts=[query], n_results=top_k)
        if not results["ids"] or not results["ids"][0]:
            return []

        retrieved: list[RetrievedChunk] = []
        for i in range(len(results["ids"][0])):
            chunk = Chunk(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                source=results["metadatas"][0][i].get("source", "unknown"),
                metadata=results["metadatas"][0][i],
            )
            # Chroma returns a cosine *distance* (0 = identical); convert to a
            # similarity score in [0, 1] so higher always means more relevant.
            distance = results["distances"][0][i]
            similarity = 1 - distance
            retrieved.append(RetrievedChunk(chunk=chunk, score=similarity))
        return retrieved

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
        )

    def count(self) -> int:
        return self._collection.count()
