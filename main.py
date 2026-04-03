"""
CLI entry point for the Multimodal RAG system.

Default chat is simple: one query → one answer.
Benchmark mode (--benchmark) adds A/B reranking comparison + RAGAS metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import cfg
from src.ingestion.multimodal_loader import MultimodalLoader
from src.rag.vector_store import VectorStore
from src.rag.pipeline import RAGPipeline

TRACKING_FILE = cfg.tracking_file
_tracking_lock = threading.Lock()


def load_processed_files() -> set:
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, IOError):
            return set()
    return set()


def save_processed_file(filename: str):
    processed = load_processed_files()
    processed.add(filename)
    with open(TRACKING_FILE, "w") as f:
        json.dump(list(processed), f)


def _process_single_file(loader, vs, data_dir, filename):
    file_path = os.path.join(data_dir, filename)
    try:
        documents = loader.load_file(file_path)
        if documents:
            vs.add_documents(documents)
            with _tracking_lock:
                save_processed_file(filename)
            print(f"  ✅ {filename}")
            return filename, True, None
        return filename, True, "No documents"
    except Exception as e:
        print(f"  ❌ {filename}: {e}")
        return filename, False, str(e)


def ingest_files(data_dir: str, vector_store: VectorStore):
    print("\n--- Ingestion Phase ---")
    os.makedirs(data_dir, exist_ok=True)

    all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    processed = load_processed_files()
    to_process = [f for f in all_files if f not in processed]

    if not to_process:
        print("All files already processed.")
        return

    if not cfg.openai_api_key:
        print("Error: OPENAI_API_KEY not set.")
        return

    loader = MultimodalLoader()
    max_workers = cfg.max_ingest_workers
    print(f"Processing {len(to_process)} files (workers={max_workers})…")

    start = time.time()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_single_file, loader, vector_store, data_dir, fn): fn for fn in to_process}
        for future in as_completed(futures):
            _, success, _ = future.result()
            if success:
                ok += 1
            else:
                fail += 1

    print(f"\n--- Done in {time.time() - start:.1f}s — {ok} succeeded, {fail} failed ---")


def chat_loop(vector_store: VectorStore, *, benchmark: bool = False):
    print("\n--- Chat Mode ---")
    pipeline = RAGPipeline(vector_store)

    if benchmark:
        print("(Benchmark mode: comparing standard vs reranked retrieval)")
        _run_benchmark_chat(pipeline)
    else:
        _run_simple_chat(pipeline)


def _run_simple_chat(pipeline: RAGPipeline):
    print("Type 'exit' to return.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        try:
            result = pipeline.query(question)
            print(f"\nAI: {result.answer}")
            if result.sources:
                print(f"Sources: {', '.join(result.sources)}")
            print()
        except Exception as e:
            print(f"Error: {e}\n")


def _run_benchmark_chat(pipeline: RAGPipeline):
    """A/B comparison with optional RAGAS — expensive, for development only."""
    try:
        from src.eval.metrics import check_ragas_metrics
    except ImportError:
        print("Warning: RAGAS not available. Install 'ragas' to use benchmark mode.")
        _run_simple_chat(pipeline)
        return

    print("Type 'exit' to return.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break

        try:
            # Standard
            print("\n=== STANDARD (no reranking) ===")
            r1 = pipeline.query(question, use_reranking=False)
            print(f"AI: {r1.answer}")

            # Reranked
            print("\n=== RERANKED ===")
            r2 = pipeline.query(question, use_reranking=True)
            print(f"AI: {r2.answer}")
            print(f"  (rerank applied: {r2.rerank_applied})")

            # RAGAS
            print("\nComputing RAGAS metrics…")
            s1 = check_ragas_metrics(question, r1.answer, r1.contexts)
            s2 = check_ragas_metrics(question, r2.answer, r2.contexts)

            print(f"Standard  — {s1}")
            print(f"Reranked  — {s2}")
            print()
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Multimodal RAG CLI")
    parser.add_argument("--benchmark", action="store_true", help="Enable A/B benchmark mode with RAGAS")
    args = parser.parse_args()

    try:
        vector_store = VectorStore()
    except Exception as e:
        print(f"Failed to init VectorStore: {e}")
        return

    data_dir = cfg.data_dir

    while True:
        print("\n=== Multimodal RAG CLI ===")
        print("1. Ingest files from 'data/'")
        print("2. Chat")
        print("3. Exit")

        choice = input("Select (1-3): ").strip()
        if choice == "1":
            ingest_files(data_dir, vector_store)
        elif choice == "2":
            chat_loop(vector_store, benchmark=args.benchmark)
        elif choice == "3":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
