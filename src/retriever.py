import time
from typing import List, Dict, Any
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.query_expander import QueryExpander

class Retriever:
    """
    Orchestrates different retrieval strategies.
    Strategy A: Raw Embedding Retrieval.
    Strategy B: Query Expanded Retrieval.
    """
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.query_expander = QueryExpander()

    def retrieve_strategy_a(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Executes raw embedding-based vector similarity retrieval.
        """
        start_time = time.time()
        query_vec = self.embedding_model.get_embeddings(query)
        results = self.vector_store.search(query_vec, k=k)
        latency = time.time() - start_time
        
        return {
            "strategy": "A (Raw)",
            "query": query,
            "results": results,
            "latency_ms": latency * 1000
        }

    def retrieve_strategy_b(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Executes AI-enhanced retrieval using query expansion before vector search.
        """
        start_time = time.time()
        
        # Step 1: Expand Query
        expanded_query = self.query_expander.expand(query)
        
        # Step 2: Vector Search with expanded query
        query_vec = self.embedding_model.get_embeddings(expanded_query)
        results = self.vector_store.search(query_vec, k=k)
        
        latency = time.time() - start_time
        
        return {
            "strategy": "B (Expanded)",
            "original_query": query,
            "expanded_query": expanded_query,
            "results": results,
            "latency_ms": latency * 1000
        }
