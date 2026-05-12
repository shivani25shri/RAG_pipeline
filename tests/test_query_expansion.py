import pytest
from src.query_expander import QueryExpander
from src.mocks import MockGenerativeModel

def test_query_expansion_with_keywords():
    expander = QueryExpander()
    query = "How to handle autoscaling"
    expanded = expander.expand(query)
    
    # "autoscaling" should trigger specific expansion rules in the mock
    assert "dynamic resource allocation" in expanded
    assert "traffic spikes" in expanded

def test_query_expansion_without_keywords():
    expander = QueryExpander()
    query = "What is the meaning of life?"
    expanded = expander.expand(query)
    
    # Should fall back to generic expansion
    assert "technical details" in expanded

def test_mock_generative_model():
    model = MockGenerativeModel()
    response = model.generate_content("caching")
    assert "Redis" in response.text
