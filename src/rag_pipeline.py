import os
from typing import List
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.benchmark import BenchmarkRunner

class RAGPipeline:
    """
    Orchestrates the entire RAG assessment pipeline:
    Loading data, indexing, and running benchmarks.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(dimension=self.embedding_model.dimension)
        self.retriever = Retriever(self.vector_store, self.embedding_model)
        self.benchmark = BenchmarkRunner()

    def load_and_index_data(self):
        """Loads technical dataset and indexes it into FAISS."""
        print(f"Loading data from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            # Simple chunking: split by double newlines (paragraphs)
            content = f.read().strip()
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]

        if not chunks:
            raise ValueError("No valid text chunks found in the dataset.")

        print(f"Loaded {len(chunks)} chunks. Generating embeddings...")
        embeddings = self.embedding_model.get_embeddings(chunks)
        
        print("Indexing documents in FAISS...")
        self.vector_store.add_documents(chunks, embeddings)
        print("Indexing complete.")

    def run_assessment(self, queries: List[str]):
        """Runs the comparison benchmark for a list of queries."""
        print("\nStarting Benchmark Assessment...\n" + "="*40)
        
        for q in queries:
            print(f"Evaluating Query: '{q}'")
            res_a = self.retriever.retrieve_strategy_a(q)
            res_b = self.retriever.retrieve_strategy_b(q)
            self.benchmark.add_result(res_a, res_b)

    def print_results(self):
        """Prints the JSON and Markdown results."""
        print("\n--- Benchmark JSON ---")
        print(self.benchmark.to_json())
        print("\n--- Benchmark Markdown ---")
        print(self.benchmark.to_markdown())

    def save_results(self, filepath: str):
        """Saves the markdown results to a file."""
        self.benchmark.save_markdown(filepath)
        print(f"\nResults saved to {filepath}")
