from src.mocks import MockGenerativeModel

class QueryExpander:
    """
    Simulates an LLM-based query expansion system.
    Rewrites a simple query into a more descriptive semantic version.
    """
    def __init__(self):
        self.llm = MockGenerativeModel()

    def expand(self, query: str) -> str:
        """
        Expands the input query using the mock LLM.
        
        Args:
            query: The original user query.
            
        Returns:
            The expanded query string.
        """
        prompt = f"Rewrite this search query to be more descriptive and semantically rich for a technical documentation search: '{query}'"
        response = self.llm.generate_content(prompt)
        return response.text
