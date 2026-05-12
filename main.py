import os
from src.rag_pipeline import RAGPipeline

def main():
    # Define dataset path
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_docs.txt')
    
    # Initialize pipeline
    pipeline = RAGPipeline(data_path=data_path)
    
    # Load and Index Data
    pipeline.load_and_index_data()
    
    # Define 3 complex technical queries
    queries = [
        "How do I handle sudden spikes in user traffic without crashing?",
        "What is the best way to reduce database load and improve response latency?",
        "How can microservices securely communicate and verify identity?"
    ]
    
    # Run Benchmark
    pipeline.run_assessment(queries)
    
    # Output Results
    pipeline.print_results()
    
    # Save Markdown
    output_md = os.path.join(os.path.dirname(__file__), 'retrieval_benchmark.md')
    pipeline.save_results(output_md)
    print("Assessment Complete.")

if __name__ == "__main__":
    main()
