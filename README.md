
# RAG_pipeline
# Local RAG Assessment Engine

A production-style, modular Retrieval-Augmented Generation (RAG) assessment project built in Python. This project compares two semantic retrieval strategies:
1. **Strategy A**: Raw embedding-based vector similarity retrieval.
2. **Strategy B**: AI-enhanced retrieval using query expansion before vector search.

## Architecture

The project is structured modularly:
- **`src/embeddings.py`**: Wraps `sentence-transformers` (`all-MiniLM-L6-v2`) to generate L2-normalized embeddings.
- **`src/vector_store.py`**: Uses FAISS (`IndexFlatIP`) for high-performance top-k similarity search.
- **`src/mocks.py`**: Simulates Vertex AI SDK (`TextEmbeddingModel`, `GenerativeModel`) to enable local execution without cloud dependencies.
- **`src/query_expander.py`**: Implements deterministic, mocked LLM query rewriting based on keyword heuristics.
- **`src/retriever.py`**: Orchestrates Strategy A and Strategy B.
- **`src/benchmark.py`**: Evaluates strategies based on latency, retrieved documents, and similarity scores.

## Cosine Similarity & Embeddings

### What is Cosine Similarity?
Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It evaluates how similar two documents are by looking at their orientation rather than magnitude. A score of `1.0` means identical direction, `0.0` means orthogonal (unrelated), and `-1.0` means opposite.

### Why Cosine Similarity over Euclidean Distance?
For text embeddings, the length (magnitude) of a vector often correlates with the length or verbosity of the text, rather than its underlying semantic meaning. Cosine similarity normalizes this effect, focusing entirely on the "direction" (semantic topic) of the text.

In this project, we normalize all embeddings to unit length (L2 norm = 1). When vectors are normalized, their **Inner Product** equals their **Cosine Similarity**. This allows us to use FAISS's highly optimized `IndexFlatIP` for rapid, accurate retrieval.

## Migration to Vertex AI Matching Engine

To migrate this local setup to GCP's Vertex AI Matching Engine (Vector Search):
1. **Embeddings**: Replace the local `SentenceTransformer` with Vertex AI `TextEmbeddingModel`. Our `mocks.py` structure already mirrors this API.
2. **Index**: Export the normalized embeddings to Google Cloud Storage (GCS) in JSONL format.
3. **Deployment**: Create a Vertex AI Vector Search Index using the "dot product" (inner product) distance measure and deploy it to an endpoint.
4. **Search**: Replace the local FAISS `IndexFlatIP.search()` call with the Vertex AI Vector Search API.

## Setup and Execution

### Prerequisites
- Python 3.10+
- `pip`

### Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### Execution
To run the end-to-end benchmark (data loading, indexing, retrieval, and evaluation):
```bash
python main.py
```

### Testing
Run the pytest suite to verify module integrity:
```bash
pytest tests/
```
>>>>>>> 20ef246 (Initial commit for RAG_pipeline assessment)
