import numpy as np
from typing import List, Any
from src.embeddings import EmbeddingModel

class MockTextEmbeddingModel:
    """
    Mocks the Vertex AI TextEmbeddingModel API.
    """
    def __init__(self, model_id: str = "textembedding-gecko@003"):
        self.model_id = model_id
        self._internal_model = EmbeddingModel()

    @classmethod
    def from_pretrained(cls, model_id: str):
        return cls(model_id)

    def get_embeddings(self, texts: List[str]) -> List[Any]:
        """
        Simulates the Vertex AI get_embeddings call.
        Returns a list of mock embedding objects with a 'values' attribute.
        """
        embeddings = self._internal_model.get_embeddings(texts)
        
        class MockEmbeddingResponse:
            def __init__(self, values):
                self.values = values.tolist()
        
        return [MockEmbeddingResponse(v) for v in embeddings]

class MockGenerativeModel:
    """
    Mocks the Vertex AI GenerativeModel API (e.g., Gemini).
    Used for query expansion simulations.
    """
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        self.model_name = model_name
        # Pre-defined mapping for deterministic query expansion simulation
        self.expansion_rules = {
            "autoscaling": "dynamic resource allocation, traffic spikes, scaling virtual machines, horizontal scaling",
            "load balancing": "traffic distribution, backend servers, high availability, network bottleneck",
            "caching": "Redis, in-memory storage, latency reduction, throughput optimization",
            "security": "authentication, authorization, OAuth 2.0, JWT, identity management",
            "reliability": "fault tolerance, redundancy, failover, graceful degradation",
            "performance": "latency, throughput, response time, resource utilization"
        }

    def generate_content(self, prompt: str) -> Any:
        """
        Simulates the generate_content call.
        Returns a mock response object with a 'text' attribute.
        """
        # Simple extraction of keywords from prompt to decide expansion
        query = prompt.lower()
        expanded_terms = []
        for key, value in self.expansion_rules.items():
            if key in query:
                expanded_terms.append(value)
        
        if not expanded_terms:
            expanded_text = query + " technical details and best practices"
        else:
            expanded_text = f"{query} including {', '.join(expanded_terms)}"

        class MockResponse:
            def __init__(self, text):
                self.text = text
        
        return MockResponse(expanded_text)
