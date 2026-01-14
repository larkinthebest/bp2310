import os
import sys
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

def load_processed_files():
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, 'r') as f:
                return set(json.load(f))
        except:
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
            result = pipeline.query(question)
            answer = result["answer"]
            contexts = result["contexts"]
            print(f"\nAI: {answer}")
            
            # RAGAS Evaluation
            print("\nComputing RAGAS Metrics (Faithfulness, Answer Relevance)...")
            scores = check_ragas_metrics(question, answer, contexts)
            # print(f"DEBUG Scores: {scores}") # Valid debug line if needed
            
            if "error" in scores:
                print(f"RAGAS Error: {scores['error']}")
            else:
                print("\n--- RAGAS Scores ---")
                print(f"Faithfulness: {scores.get('faithfulness', 0):.4f}")
                print(f"Answer Relevancy: {scores.get('answer_relevancy', 0):.4f}")
                print(f"Context Precision: {scores.get('context_precision', 0):.4f}")
                print(f"Context Recall: {scores.get('context_recall', 0):.4f}")
                
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
