import pytest
import numpy as np
from src.vector_store import VectorStore
from src.embeddings import EmbeddingModel
from src.retriever import Retriever

@pytest.fixture
def sample_data():
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large amounts of data.",
        "Cloud computing provides scalable resources over the internet."
    ]
    model = EmbeddingModel()
    embeddings = model.get_embeddings(docs)
    return docs, embeddings, model

def test_vector_store_indexing(sample_data):
    docs, embeddings, model = sample_data
    store = VectorStore(dimension=model.dimension)
    
    assert store.get_document_count() == 0
    store.add_documents(docs, embeddings)
    assert store.get_document_count() == 3

def test_vector_store_search(sample_data):
    docs, embeddings, model = sample_data
    store = VectorStore(dimension=model.dimension)
    store.add_documents(docs, embeddings)
    
    query = "AI needs data"
    q_emb = model.get_embeddings(query)
    
    results = store.search(q_emb, k=1)
    assert len(results) == 1
    assert "Machine learning" in results[0]["document"]

def test_retriever_strategies(sample_data):
    docs, embeddings, model = sample_data
    store = VectorStore(dimension=model.dimension)
    store.add_documents(docs, embeddings)
    
    retriever = Retriever(store, model)
    query = "scalable internet resources"
    
    res_a = retriever.retrieve_strategy_a(query, k=1)
    assert res_a["strategy"] == "A (Raw)"
    assert "Cloud computing" in res_a["results"][0]["document"]
    
    res_b = retriever.retrieve_strategy_b(query, k=1)
    assert res_b["strategy"] == "B (Expanded)"
    assert res_b["expanded_query"] != query
    assert "Cloud computing" in res_b["results"][0]["document"]
