from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import os
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

def check_ragas_metrics(question, answer, contexts):
    """
    Runs RAGAS faithfulness, answer_relevancy, and context_relevancy metrics.
    Returns a dictionary of scores.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("DEBUG: OPENAI_API_KEY missing")
        return {"error": "OPENAI_API_KEY missing"}

    print(f"DEBUG: Computing metrics for Question: '{question}' with {len(contexts)} contexts.")
    
    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
        'ground_truth': [answer], # Needed for some metrics
        'reference': [answer], # Needed for others (newer versions)
    }
    
    dataset = Dataset.from_dict(data)
    
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm, 
            embeddings=evaluator_embeddings
        )
        print(f"DEBUG: Raw Results Object: {results}")
        print(f"DEBUG: Results Type: {type(results)}")
        
        # Convert to dict safely using pandas
        try:
            # Ragas Result object usually supports to_pandas()
            df = results.to_pandas()
            # We have only one row
            clean_scores = df.iloc[0].to_dict()
            
            # Filter to keep only the metrics we asked for (remove 'question', 'answer', etc if present)
            wanted_metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
            final_scores = {k: v for k, v in clean_scores.items() if k in wanted_metrics}
            
            return final_scores
        except Exception as e:
            print(f"DEBUG: Error converting results to pandas: {e}")
            import traceback
            traceback.print_exc()
            # Emergency fallback: parse string representation? No, that's too brittle.
            # Try accessing .scores if it exists
            return {} 
            
    except Exception as e:
        print(f"DEBUG: Error inside evaluate: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
