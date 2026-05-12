import faiss
import numpy as np
from typing import List, Tuple, Dict

class VectorStore:
    """
    A FAISS-based vector store for semantic document retrieval.
    Uses Inner Product (IndexFlatIP) on normalized vectors to calculate Cosine Similarity.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        # IndexFlatIP is equivalent to cosine similarity if vectors are normalized
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[str] = []

    def add_documents(self, texts: List[str], embeddings: np.ndarray):
        """
        Adds documents and their corresponding embeddings to the index.
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")
        
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(texts)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, any]]:
        """
        Searches for the top-k most similar documents.
        
        Returns:
            A list of dictionaries containing 'document', 'score', and 'chunk_id'.
        """
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:  # FAISS returns -1 if not enough neighbors are found
                results.append({
                    "chunk_id": int(idx),
                    "document": self.documents[idx],
                    "score": float(scores[0][i])
                })
        
        return results

    def get_document_count(self) -> int:
        return len(self.documents)
