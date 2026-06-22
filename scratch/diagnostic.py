import json
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.vectorstore.faiss_store import FAISSStore

def run_diagnostics():
    # 1. Corpus size check
    store = FAISSStore()
    total_vectors = store.index.ntotal
    embedding_dim = store.index.d
    
    unique_arxiv_ids = set()
    for chunk in store.registry.values():
        unique_arxiv_ids.add(chunk.get("arxiv_id"))
        
    print("=" * 60)
    print("1. CORPUS SIZE CHECK")
    print(f"Total vectors in FAISS store: {total_vectors}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of unique arxiv_ids represented: {len(unique_arxiv_ids)}")
    print("=" * 60)

    # Load evaluation dataset
    dataset_path = Path("data/processed/eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    # 2. Query-to-corpus relevance spot check
    scores = []
    results_list = []
    
    for i, item in enumerate(eval_data):
        q = item["question"]
        expected_arxiv_id = item.get("arxiv_id")
        
        # Search FAISS directly
        results = store.search(q, top_k=5)
        if results:
            top_score = results[0]["score"]
            top_arxiv_id = results[0]["arxiv_id"]
            top_title = results[0]["title"]
            top_text = results[0]["text"]
        else:
            top_score = 0.0
            top_arxiv_id = None
            top_title = None
            top_text = None
            
        scores.append(top_score)
        results_list.append({
            "index": i,
            "question": q,
            "expected_arxiv_id": expected_arxiv_id,
            "top_score": top_score,
            "top_arxiv_id": top_arxiv_id,
            "top_title": top_title,
            "top_text": top_text
        })
        
    df_scores = pd.DataFrame(results_list)
    
    print("2. QUERY-TO-CORPUS RELEVANCE SPOT CHECK")
    print(f"Top-1 Similarity Score Distribution across {len(eval_data)} queries:")
    print(f"  Min   : {df_scores['top_score'].min():.4f}")
    print(f"  Max   : {df_scores['top_score'].max():.4f}")
    print(f"  Mean  : {df_scores['top_score'].mean():.4f}")
    print(f"  Median: {df_scores['top_score'].median():.4f}")
    print(f"  Std   : {df_scores['top_score'].std():.4f}")
    
    print("\nScore Ranges:")
    ranges = [
        ("< 0.3", df_scores['top_score'] < 0.3),
        ("0.3 - 0.4", (df_scores['top_score'] >= 0.3) & (df_scores['top_score'] < 0.4)),
        ("0.4 - 0.5", (df_scores['top_score'] >= 0.4) & (df_scores['top_score'] < 0.5)),
        ("0.5 - 0.6", (df_scores['top_score'] >= 0.5) & (df_scores['top_score'] < 0.6)),
        (">= 0.6", df_scores['top_score'] >= 0.6),
    ]
    for label, mask in ranges:
        count = mask.sum()
        pct = (count / len(eval_data)) * 100
        print(f"  {label:<10} : {count:2d} ({pct:.1f}%)")
    print("=" * 60)
    
    # 3. Manual relevance sample (Lowest 10 scores)
    print("3. MANUAL RELEVANCE SAMPLE (LOWEST TOP-1 SCORES)")
    df_lowest = df_scores.sort_values(by="top_score").head(10)
    for idx, row in df_lowest.iterrows():
        print(f"\n[Index {row['index']}] Question: {row['question']}")
        print(f"  Top-1 Score: {row['top_score']:.4f} | Top arxiv_id: {row['top_arxiv_id']} | Expected: {row['expected_arxiv_id']}")
        print(f"  Top Chunk Title: {row['top_title']}")
        snippet = row['top_text'][:300].replace('\n', ' ')
        print(f"  Top Chunk Snippet: {snippet}...")
    print("=" * 60)

    # 4. Coverage check against eval_dataset.json
    print("4. COVERAGE CHECK AGAINST EVAL_DATASET.JSON")
    matches = 0
    missing_ids = set()
    for item in eval_data:
        expected = item.get("arxiv_id")
        if expected:
            # Check if any chunk in the registry has this arxiv_id
            found = False
            for chunk in store.registry.values():
                if chunk.get("arxiv_id") == expected:
                    found = True
                    break
            if found:
                matches += 1
            else:
                missing_ids.add(expected)
                
    print(f"Total questions with expected arxiv_id: {len(eval_data)}")
    print(f"Questions where expected paper is indexed: {matches} / {len(eval_data)} ({matches/len(eval_data)*100:.1f}%)")
    print(f"Missing arxiv_ids in the corpus: {sorted(list(missing_ids))}")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()
