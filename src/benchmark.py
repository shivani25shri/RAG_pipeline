import json
from typing import List, Dict, Any
from tabulate import tabulate

class BenchmarkRunner:
    """
    Evaluates and compares different retrieval strategies.
    Outputs results in JSON and Markdown formats.
    """
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add_result(self, strategy_a: Dict[str, Any], strategy_b: Dict[str, Any]):
        """
        Stores the comparison results for a single query.
        """
        self.results.append({
            "query": strategy_a["query"],
            "strategy_a": strategy_a,
            "strategy_b": strategy_b
        })

    def to_json(self) -> str:
        """Returns the results as a formatted JSON string."""
        return json.dumps(self.results, indent=2)

    def to_markdown(self) -> str:
        """Returns the results as a Markdown table."""
        table_data = []
        for res in self.results:
            q = res["query"]
            expanded_q = res["strategy_b"].get("expanded_query", "N/A")
            
            # Get top result for each
            top_a = res["strategy_a"]["results"][0] if res["strategy_a"]["results"] else {"document": "None", "score": 0}
            top_b = res["strategy_b"]["results"][0] if res["strategy_b"]["results"] else {"document": "None", "score": 0}
            
            table_data.append([
                q,
                expanded_q,
                f"{top_a['score']:.4f}",
                f"{top_b['score']:.4f}",
                top_a['document'][:100] + "...",
                top_b['document'][:100] + "..."
            ])

        headers = ["Original Query", "Expanded Query", "Score (A)", "Score (B)", "Top Doc (A)", "Top Doc (B)"]
        return tabulate(table_data, headers=headers, tablefmt="github")

    def save_markdown(self, filepath: str):
        """Saves the markdown comparison to a file."""
        with open(filepath, 'w') as f:
            f.write("# RAG Retrieval Benchmark Results\n\n")
            f.write("Comparing Strategy A (Raw) vs Strategy B (Query Expansion)\n\n")
            f.write(self.to_markdown())
            f.write("\n")
