import json
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.vectorstore.faiss_store import FAISSStore

def segment_evaluation_results():
    crag_csv = Path("data/processed/evaluation_results_crag_gemini-3-1-flash-lite.csv")
    naive_csv = Path("data/processed/evaluation_results_naive_gemini-3-1-flash-lite.csv")
    eval_json = Path("data/processed/eval_dataset.json")
    
    if not crag_csv.exists() or not naive_csv.exists() or not eval_json.exists():
        print("Error: Required data files are missing.")
        return

    # Load dataframes
    df_crag = pd.read_csv(crag_csv)
    df_naive = pd.read_csv(naive_csv)
    
    # Load eval dataset
    with open(eval_json, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    q_to_arxiv = {item["question"]: item.get("arxiv_id") for item in eval_data}
    
    # Initialize FAISSStore
    store = FAISSStore()
    
    # Create merged dataframe on question
    df_merged = pd.merge(
        df_crag[["question", "web_search_used", "pipeline_grade", "faithfulness", "answer_relevancy"]],
        df_naive[["question", "faithfulness", "answer_relevancy"]],
        on="question",
        suffixes=("_crag", "_naive")
    )
    
    # Classify each row
    group_labels = []
    
    for idx, row in df_merged.iterrows():
        question = row["question"]
        web_search = row["web_search_used"]
        expected_arxiv_id = q_to_arxiv.get(question)
        
        if not web_search:
            group_labels.append("relevant_direct_hit")
        else:
            # Check FAISS retrieved IDs
            results = store.search(question, top_k=5)
            top_ids = [r["arxiv_id"] for r in results]
            
            if expected_arxiv_id in top_ids:
                group_labels.append("grader_rejection")
            else:
                group_labels.append("retriever_miss")
                
    df_merged["segment"] = group_labels
    
    # Group and compute means
    summary = []
    for segment in ["retriever_miss", "grader_rejection", "relevant_direct_hit"]:
        sub_df = df_merged[df_merged["segment"] == segment]
        count = len(sub_df)
        
        mean_faith_crag = sub_df["faithfulness_crag"].mean()
        mean_rel_crag = sub_df["answer_relevancy_crag"].mean()
        
        mean_faith_naive = sub_df["faithfulness_naive"].mean()
        mean_rel_naive = sub_df["answer_relevancy_naive"].mean()
        
        summary.append({
            "segment": segment,
            "count": count,
            "mean_faith_crag": mean_faith_crag,
            "mean_rel_crag": mean_rel_crag,
            "mean_faith_naive": mean_faith_naive,
            "mean_rel_naive": mean_rel_naive
        })
        
    print("=" * 80)
    print("COMPARATIVE SEGMENTATION REPORT")
    print("=" * 80)
    for s in summary:
        print(f"Segment: {s['segment'].upper()} (Count: {s['count']})")
        print(f"  CRAG  | Faithfulness: {s['mean_faith_crag']:.4f} | Answer Relevancy: {s['mean_rel_crag']:.4f}")
        print(f"  Naive | Faithfulness: {s['mean_faith_naive']:.4f} | Answer Relevancy: {s['mean_rel_naive']:.4f}")
        print(f"  Delta | Faithfulness: {(s['mean_faith_crag'] - s['mean_faith_naive']):+.4f} | Answer Relevancy: {(s['mean_rel_crag'] - s['mean_rel_naive']):+.4f}")
        print("-" * 80)

if __name__ == "__main__":
    segment_evaluation_results()
