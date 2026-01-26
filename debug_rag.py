
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.rag.vector_store import VectorStore
from src.rag.pipeline import RAGPipeline

def debug():
    print("Initializing Vector Store...")
    vs = VectorStore()
    
    print("Initializing Pipeline...")
    pipeline = RAGPipeline(vs)
    
    # query = "What is the penalty for fighting?" 
    # Use a generic query that should hit the PDF or video
    query = "Who won the Kings vs Panthers game?" 
    
    print(f"Query: {query}")
    result = pipeline.query(query)
    
    print("\n--- Answer ---")
    print(result["answer"])
    
    print("\n--- Contexts (First 500 chars) ---")
    for i, ctx in enumerate(result["contexts"]):
        print(f"Ctx {i}: {ctx[:200] if ctx else '(empty)'}...")
        
    print("\n--- Sources Key ---")
    print(result.get("sources"))

if __name__ == "__main__":
    debug()
