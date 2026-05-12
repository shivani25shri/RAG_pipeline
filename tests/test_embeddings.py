import pytest
import numpy as np
from src.embeddings import EmbeddingModel

def test_embedding_dimension():
    model = EmbeddingModel()
    assert model.dimension == 384  # all-MiniLM-L6-v2 dimension

def test_single_embedding():
    model = EmbeddingModel()
    text = "This is a test sentence."
    embedding = model.get_embeddings(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 384)
    # Check normalization (L2 norm should be roughly 1.0)
    norm = np.linalg.norm(embedding[0])
    assert np.isclose(norm, 1.0, atol=1e-5)

def test_batch_embedding():
    model = EmbeddingModel()
    texts = ["Sentence one.", "Sentence two.", "Sentence three."]
    embeddings = model.get_embeddings(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)
