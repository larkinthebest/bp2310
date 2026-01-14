from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import os
from rag.pipeline import RAGPipeline
from rag.vector_store import VectorStore

def run_evaluation():
    # 1. Setup
    vector_store = VectorStore()
    pipeline = RAGPipeline(vector_store)
    
    # 2. Define Test Data (Synthetic or Manual)
    # Ideally, you'd generate this or load from a file.
    questions = [
        "What is the main topic of the video?",
        "Describe the image content.",
    ]
    ground_truths = [
        ["The video discusses the impact of AI on society."],
        ["The image shows a cat sitting on a sofa."],
    ]
    
    answers = []
    contexts = []
    
    # 3. Run RAG
    for q in questions:
        # Note: This assumes the vector store is already populated
        result = pipeline.query(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"]) 

    # 4. Prepare Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # 5. Run RAGAS
    results = evaluate(
        dataset = dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    print(results)
    return results

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY")
    else:
        run_evaluation()
