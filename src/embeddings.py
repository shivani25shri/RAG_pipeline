import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

class EmbeddingModel:
    """
    A wrapper around sentence-transformers to provide a modular embedding interface.
    Uses 'all-MiniLM-L6-v2' by default.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generates normalized embeddings for the given text or list of texts.
        
        Args:
            texts: A single string or a list of strings to embed.
            
        Returns:
            A numpy array of shape (n, dim) with normalized embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Generating embeddings with normalization (unit length)
        # This allows us to use Inner Product (IndexFlatIP) in FAISS as Cosine Similarity.
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    @property
    def dimension(self) -> int:
        """Returns the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
