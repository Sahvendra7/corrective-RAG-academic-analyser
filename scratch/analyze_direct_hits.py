import pandas as pd
import json
import sys
from pathlib import Path

def safe_print(text):
    if not isinstance(text, str):
        text = str(text)
    # Safely print using stdout encoding, falling back to 'replace'
    enc = sys.stdout.encoding or 'utf-8'
    print(text.encode(enc, errors='replace').decode(enc))

def analyze_direct_hits():
    crag_csv = Path("data/processed/evaluation_results_crag_gemini-3-1-flash-lite.csv")
    naive_csv = Path("data/processed/evaluation_results_naive_gemini-3-1-flash-lite.csv")
    
    df_crag = pd.read_csv(crag_csv)
    df_naive = pd.read_csv(naive_csv)
    
    df_merged = pd.merge(
        df_crag,
        df_naive,
        on="question",
        suffixes=("_crag", "_naive")
    )
    
    # Filter to direct-hits (web_search_used = False in CRAG)
    df_dh = df_merged[df_merged["web_search_used_crag"] == False].copy()
    
    # Find cases where Naive scored higher in either faithfulness or answer_relevancy
    df_dh["faith_diff"] = df_dh["faithfulness_naive"] - df_dh["faithfulness_crag"]
    df_dh["rel_diff"] = df_dh["answer_relevancy_naive"] - df_dh["answer_relevancy_crag"]
    
    # Sort by maximum discrepancy in either metric
    df_dh["max_diff"] = df_dh[["faith_diff", "rel_diff"]].max(axis=1)
    df_dh_discrepant = df_dh[df_dh["max_diff"] > 0.05].sort_values(by="max_diff", ascending=False)
    
    safe_print(f"Found {len(df_dh_discrepant)} Direct-Hit cases where Naive scored meaningfully higher than CRAG.\n")
    
    # Print the top 3 cases in detail
    for i, (idx, row) in enumerate(df_dh_discrepant.head(3).iterrows()):
        safe_print("="*100)
        safe_print(f"CASE {i+1}: QUESTION: {row['question']}")
        safe_print("="*100)
        safe_print(f"Metrics Compare:")
        safe_print(f"  Naive | Faithfulness: {row['faithfulness_naive']:.3f} | Relevancy: {row['answer_relevancy_naive']:.3f}")
        safe_print(f"  CRAG  | Faithfulness: {row['faithfulness_crag']:.3f} | Relevancy: {row['answer_relevancy_crag']:.3f}")
        safe_print(f"  Delta | Faithfulness: {-row['faith_diff']:.3f} | Relevancy: {-row['rel_diff']:.3f}")
        safe_print("-" * 100)
        
        # Parse contexts to list lengths to verify difference
        try:
            import ast
            c_crag = ast.literal_eval(row["contexts_crag"])
            c_naive = ast.literal_eval(row["contexts_naive"])
        except Exception:
            c_crag = [row["contexts_crag"]]
            c_naive = [row["contexts_naive"]]
            
        safe_print(f"Context Counts:")
        safe_print(f"  Naive Context Chunks: {len(c_naive)}")
        safe_print(f"  CRAG Context Chunks:  {len(c_crag)}")
        safe_print(f"  Identical Contexts?:  {c_crag == c_naive}")
        safe_print("-" * 100)
        
        safe_print("NAIVE CONTEXTS (First 200 chars per chunk):")
        for idx_c, chunk in enumerate(c_naive):
            safe_print(f"  Chunk {idx_c+1}: {chunk[:200]}...")
        safe_print("-" * 100)
        
        safe_print("CRAG CONTEXTS (First 200 chars per chunk):")
        for idx_c, chunk in enumerate(c_crag):
            safe_print(f"  Chunk {idx_c+1}: {chunk[:200]}...")
        safe_print("-" * 100)
        
        safe_print("NAIVE ANSWER:")
        safe_print(row["answer_naive"])
        safe_print("-" * 100)
        
        safe_print("CRAG ANSWER:")
        safe_print(row["answer_crag"])
        safe_print("\n\n")

if __name__ == "__main__":
    analyze_direct_hits()
