"""Quick test validation of core RAG components."""

def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.embeddings import EmbeddingModel
        from src.vector_store import VectorStore
        from src.retriever import Retriever
        from src.query_expander import QueryExpander
        from src.mocks import MockGenerativeModel, MockTextEmbeddingModel
        from src.benchmark import BenchmarkRunner
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_embeddings():
    """Test embedding generation."""
    try:
        from src.embeddings import EmbeddingModel
        model = EmbeddingModel()
        
        # Test single embedding
        embedding = model.get_embeddings("Test sentence")
        assert embedding.shape == (1, 384), f"Expected (1, 384), got {embedding.shape}"
        
        # Test batch embedding
        embeddings = model.get_embeddings(["Text 1", "Text 2"])
        assert embeddings.shape == (2, 384), f"Expected (2, 384), got {embeddings.shape}"
        
        print("✓ Embeddings: PASS")
        return True
    except Exception as e:
        print(f"✗ Embeddings: FAIL - {e}")
        return False

def test_vector_store():
    """Test vector store functionality."""
    try:
        from src.vector_store import VectorStore
        from src.embeddings import EmbeddingModel
        
        model = EmbeddingModel()
        store = VectorStore(dimension=384)
        
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        embeddings = model.get_embeddings(docs)
        store.add_documents(docs, embeddings)
        
        assert store.get_document_count() == 3, f"Expected 3 docs, got {store.get_document_count()}"
        
        # Test search
        query_embedding = model.get_embeddings("Doc")[0:1]
        results = store.search(query_embedding, k=2)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        
        print("✓ Vector Store: PASS")
        return True
    except Exception as e:
        print(f"✗ Vector Store: FAIL - {e}")
        return False

def test_query_expansion():
    """Test query expansion."""
    try:
        from src.query_expander import QueryExpander
        
        expander = QueryExpander()
        original = "How to handle autoscaling"
        expanded = expander.expand(original)
        
        assert isinstance(expanded, str), "Expansion should return string"
        assert len(expanded) > len(original), "Expanded query should be longer"
        
        print("✓ Query Expansion: PASS")
        return True
    except Exception as e:
        print(f"✗ Query Expansion: FAIL - {e}")
        return False

def test_retriever():
    """Test retriever orchestration."""
    try:
        from src.retriever import Retriever
        from src.embeddings import EmbeddingModel
        
        model = EmbeddingModel()
        docs = ["Autoscaling helps with traffic spikes", "Load balancing distributes requests"]
        embeddings = model.get_embeddings(docs)
        
        retriever = Retriever(docs, embeddings, model)
        
        # Test both strategies
        query = "How to handle traffic?"
        results_a = retriever.retrieve_strategy_a(query, k=1)
        results_b = retriever.retrieve_strategy_b(query, k=1)
        
        assert 'results' in results_a, "Strategy A should return results dict"
        assert 'results' in results_b, "Strategy B should return results dict"
        
        print("✓ Retriever: PASS")
        return True
    except Exception as e:
        print(f"✗ Retriever: FAIL - {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("RAG Assessment - Test Summary")
    print("=" * 50)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("Vector Store", test_vector_store()))
    results.append(("Query Expansion", test_query_expansion()))
    results.append(("Retriever", test_retriever()))
    
    print("=" * 50)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    print("=" * 50)
