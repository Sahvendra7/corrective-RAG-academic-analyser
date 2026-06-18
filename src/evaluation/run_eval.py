import os
import sys
import pandas as pd
from pathlib import Path

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.graph import build_graph
from src.pipeline.state import create_initial_state
from src.pipeline.llm_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

# --- Custom LLM-as-a-Judge Evaluators ---

def evaluate_faithfulness(question: str, answer: str, context: str, llm) -> float:
    """Checks if the answer is grounded in the retrieved context."""
    if not context or context == "No context retrieved.":
        return 0.0
        
    sys_prompt = "You are an expert evaluator. Given a question, an answer, and the retrieved context, determine if the answer is completely faithful to the context (i.e., it contains no hallucinations). Output ONLY '1' if faithful, or '0' if it contains unsupported claims."
    human_prompt = f"Question: {question}\nContext: {context}\nAnswer: {answer}\n\nScore (1 or 0):"
    
    try:
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
        score = response.content.strip()
        # Validate the response is actually 0 or 1
        if score in ("0", "1", "0.0", "1.0"):
            return float(score)
        logger.warning(f"Unexpected faithfulness score from LLM: '{score}', defaulting to 0.0")
        return 0.0
    except (ValueError, AttributeError, Exception) as e:
        logger.error(f"Faithfulness evaluation failed: {e}")
        return 0.0

def evaluate_relevance(question: str, answer: str, ground_truth: str, llm) -> float:
    """Checks if the generated answer accurately addresses the question compared to the ground truth."""
    sys_prompt = "You are an expert evaluator. Compare the generated answer against the ground truth answer for the given question. Does the generated answer successfully and accurately address the question? Output ONLY '1' for yes, or '0' for no."
    human_prompt = f"Question: {question}\nGround Truth: {ground_truth}\nGenerated Answer: {answer}\n\nScore (1 or 0):"
    
    try:
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
        score = response.content.strip()
        # Validate the response is actually 0 or 1
        if score in ("0", "1", "0.0", "1.0"):
            return float(score)
        logger.warning(f"Unexpected relevance score from LLM: '{score}', defaulting to 0.0")
        return 0.0
    except (ValueError, AttributeError, Exception) as e:
        logger.error(f"Relevance evaluation failed: {e}")
        return 0.0

# --- Logging ---
import logging
logger = logging.getLogger(__name__)

# --- Main Evaluation Execution ---

def run_evaluation():
    # 1. Define your Ground Truth Test Set
    eval_data = [
        {
            "question": "What is Corrective Retrieval Augmented Generation (CRAG)?",
            "ground_truth": "CRAG is a method that evaluates retrieved documents and triggers web searches if the documents are ambiguous or irrelevant."
        },
        {
            "question": "What does the hallucination node do in Self-RAG?",
            "ground_truth": "It checks if the generated answer is fully supported by the retrieved documents and flags unsupported claims."
        },
        {
            "question": "How does the rewriter node improve query quality in CRAG?",
            "ground_truth": "The rewriter uses HyDE (Hypothetical Document Embeddings) to generate a hypothetical ideal answer and then rewrites the original query to be more specific and semantically rich for better retrieval."
        },
        {
            "question": "What is the role of the grader node in the CRAG pipeline?",
            "ground_truth": "The grader evaluates each retrieved document for relevance to the query, assigning grades of relevant, ambiguous, or irrelevant, and makes an overall decision that determines the next step in the pipeline."
        },
        {
            "question": "How does FAISS enable fast similarity search in the CRAG system?",
            "ground_truth": "FAISS uses inner product search on L2-normalized embeddings (equivalent to cosine similarity) to find the most semantically similar document chunks to a query embedding."
        },
    ]

    # Initialize your LangGraph pipeline & LLM Judge
    app_graph = build_graph()
    judge_llm = get_llm()
    
    results = []

    print("🚀 Running Test Set through CRAG Pipeline...\n")
    
    # 2. Run the pipeline and evaluate each test question
    for i, item in enumerate(eval_data):
        query = item["question"]
        print(f"[{i+1}/{len(eval_data)}] Processing: '{query}'")
        
        # Execute the graph
        initial_state = create_initial_state(query)
        final_state = app_graph.invoke(initial_state, config={"configurable": {"thread_id": f"eval-{i}"}})
        
        # Extract data
        generated_answer = final_state.get("generation", "No answer generated.")
        retrieved_docs = final_state.get("documents", [])
        
        # Safely extract text whether LangGraph kept it as an object or serialized it to a dict
        context_list = []
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content'):
                context_list.append(doc.page_content)
            elif isinstance(doc, dict):
                # Look for page_content, fallback to text, fallback to converting dict to string
                context_list.append(doc.get("page_content", doc.get("text", str(doc))))
                
        combined_context = "\n".join(context_list) if context_list else "No context retrieved."
        # Evaluate using our custom Judge functions
        faithfulness_score = evaluate_faithfulness(query, generated_answer, combined_context, judge_llm)
        relevance_score = evaluate_relevance(query, generated_answer, item["ground_truth"], judge_llm)
        
        results.append({
            "Question": query,
            "Answer": generated_answer[:100] + "...", # Truncate for display
            "Faithfulness": faithfulness_score,
            "Relevance": relevance_score
        })

    # 3. Output the results using Pandas
    df = pd.DataFrame(results)
    
    print("\n✅ Evaluation Complete! Final Scores:\n")
    print(df.to_string(index=False))
    
    # Compute and display aggregate metrics
    avg_faithfulness = df["Faithfulness"].mean()
    avg_relevance = df["Relevance"].mean()
    print(f"\n📊 Aggregate Metrics:")
    print(f"  Average Faithfulness : {avg_faithfulness:.2f}")
    print(f"  Average Relevance    : {avg_relevance:.2f}")
    
    # Save to CSV for tracking over time
    output_path = "data/evaluation_results.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n📁 Detailed results saved to {output_path}")

if __name__ == "__main__":
    run_evaluation()