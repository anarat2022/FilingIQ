"""
Retriever: wraps a Datastore with retrieval-specific policy. Right now that
policy is a minimum similarity threshold, so a query that doesn't actually
match anything in the corpus returns nothing instead of forcing the top_k
weakest matches on the generator (which is how RAG systems hallucinate).
"""

from src.interfaces import BaseDatastore, BaseRetriever, RetrievedChunk

MIN_SIMILARITY = 0.15  # empirical floor; tune once you see real query scores


class SimpleRetriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore, min_similarity: float = MIN_SIMILARITY):
        self._datastore = datastore
        self._min_similarity = min_similarity

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        results = self._datastore.search(query, top_k=top_k)
        return [r for r in results if r.score >= self._min_similarity]
