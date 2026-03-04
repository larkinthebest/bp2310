import os
import sys
from typing import Any
from dotenv import load_dotenv

# Load env before imports
print("DEBUG: Loading .env file...")
load_dotenv()

print("DEBUG: Importing MultimodalLoader...")
from src.ingestion.multimodal_loader import MultimodalLoader
print("DEBUG: Importing VectorStore...")
from src.rag.vector_store import VectorStore
print("DEBUG: Importing RAGPipeline...")
from src.rag.pipeline import RAGPipeline
print("DEBUG: Importing Metrics...")
from src.eval.metrics import check_ragas_metrics

import json

TRACKING_FILE = "processed_files.json"


def metric_value(scores: dict[str, Any], key: str) -> float:
    value = scores.get(key, 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def load_processed_files():
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, 'r') as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load tracking file: {e}")
            return set()
    return set()

def save_processed_file(filename):
    processed = load_processed_files()
    processed.add(filename)
    with open(TRACKING_FILE, 'w') as f:
        json.dump(list(processed), f)

def ingest_files(data_dir: str, vector_store: VectorStore):
    print(f"\n--- Ingestion Phase ---")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        print("Please add files to this directory and try again.")
        return

    all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    if not all_files:
        print(f"No files found in {data_dir}. Please add .txt, .pdf, .mp3, .mp4 files.")
        return

    processed_files = load_processed_files()
    files_to_process = [f for f in all_files if f not in processed_files]
    
    if not files_to_process:
        print("All files in 'data/' have already been processed.")
        return

    print(f"Found {len(files_to_process)} new files to process.")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY not set.")
        return
        
    loader = MultimodalLoader(openai_api_key=openai_key)
    
    for filename in files_to_process:
        file_path = os.path.join(data_dir, filename)
        print(f"Processing: {filename}...")
        try:
            documents = loader.load_file(file_path)
            if documents:
                vector_store.add_documents(documents)
                print(f"Successfully ingested {filename}")
                save_processed_file(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def chat_loop(vector_store: VectorStore):
    print(f"\n--- Chat Phase ---")
    print("Initializing RAG Pipeline (loading CLIP model for query embedding)...")
    pipeline = RAGPipeline(vector_store)
    
    print("\nSystem Ready. Type 'exit' to return to menu.")
    while True:
        question = input("\nYou: ")
        if question.lower() in ['exit', 'quit']:
            break
        
        try:
            print("Thinking...")
            
            # === Standard Retrieval (No Reranking) ===
            print("\n" + "="*50)
            print("=== STANDARD RETRIEVAL (No Reranking) ===")
            print("="*50)
            
            result_standard = pipeline.query(question, use_reranking=False)
            answer_standard = result_standard["answer"]
            contexts_standard = result_standard["contexts"]
            sources_standard = result_standard.get("sources", [])
            
            print(f"\nAI: {answer_standard}")
            if sources_standard:
                print(f"Sources: {', '.join(sources_standard)}")
            
            print("\nComputing RAGAS Metrics...")
            scores_standard = check_ragas_metrics(question, answer_standard, contexts_standard)
            
            if "error" in scores_standard:
                print(f"RAGAS Error: {scores_standard['error']}")
                faith_std = rel_std = prec_std = rec_std = 0.0
            else:
                faith_std = metric_value(scores_standard, "faithfulness")
                rel_std = metric_value(scores_standard, "answer_relevancy")
                prec_std = metric_value(scores_standard, "context_precision")
                rec_std = metric_value(scores_standard, "context_recall")
                
                print(f"Faithfulness: {faith_std:.4f} | Relevancy: {rel_std:.4f}")
                print(f"Context Precision: {prec_std:.4f} | Context Recall: {rec_std:.4f}")
            
            # === Reranker-Enhanced Retrieval ===
            print("\n" + "="*50)
            print("=== RERANKER-ENHANCED RETRIEVAL ===")
            print("="*50)
            
            result_reranked = pipeline.query(question, use_reranking=True)
            answer_reranked = result_reranked["answer"]
            contexts_reranked = result_reranked["contexts"]
            sources_reranked = result_reranked.get("sources", [])
            
            print(f"\nAI: {answer_reranked}")
            if sources_reranked:
                print(f"Sources: {', '.join(sources_reranked)}")
            
            print("\nComputing RAGAS Metrics...")
            scores_reranked = check_ragas_metrics(question, answer_reranked, contexts_reranked)
            
            if "error" in scores_reranked:
                print(f"RAGAS Error: {scores_reranked['error']}")
                faith_rr = rel_rr = prec_rr = rec_rr = 0.0
            else:
                faith_rr = metric_value(scores_reranked, "faithfulness")
                rel_rr = metric_value(scores_reranked, "answer_relevancy")
                prec_rr = metric_value(scores_reranked, "context_precision")
                rec_rr = metric_value(scores_reranked, "context_recall")
                
                print(f"Faithfulness: {faith_rr:.4f} | Relevancy: {rel_rr:.4f}")
                print(f"Context Precision: {prec_rr:.4f} | Context Recall: {rec_rr:.4f}")
            
            # === Comparison Summary ===
            print("\n" + "="*50)
            print("=== COMPARISON SUMMARY ===")
            print("="*50)
            
            faith_diff = faith_rr - faith_std
            rel_diff = rel_rr - rel_std
            
            print(f"Faithfulness: Standard={faith_std:.4f} | Reranked={faith_rr:.4f} | Diff={faith_diff:+.4f}")
            print(f"Relevancy:    Standard={rel_std:.4f} | Reranked={rel_rr:.4f} | Diff={rel_diff:+.4f}")
            
            # Determine winner
            if rel_rr > rel_std and faith_rr >= faith_std:
                print("\n✅ Reranker improved the answer quality!")
            elif rel_std > rel_rr and faith_std >= faith_rr:
                print("\n⚠️ Standard retrieval performed better this time.")
            else:
                print("\n➡️ Mixed results - check individual metrics above.")
                
        except Exception as e:
            print(f"Error during query: {repr(e)}")
            import traceback
            traceback.print_exc()

def main():
    print("Initializing Vector Store...")
    try:
        vector_store = VectorStore()
    except Exception as e:
        print(f"Failed to initialize VectorStore: {e}")
        print("Check your .env configuration.")
        return

    data_dir = os.path.join(os.getcwd(), "data")

    while True:
        print("\n=== Multimodal RAG CLI ===")
        print("1. Ingest Files from 'data/'")
        print("2. Chat with your Data")
        print("3. Exit")
        
        choice = input("Select option (1-3): ")
        
        if choice == '1':
            ingest_files(data_dir, vector_store)
        elif choice == '2':
            chat_loop(vector_store)
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
