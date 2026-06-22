import json
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from src.vectorstore.faiss_store import FAISSStore

def analyze_fallbacks():
    # Load the new CRAG CSV
    csv_path = Path("data/processed/evaluation_results_crag_gemini-3-1-flash-lite.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return
    df = pd.read_csv(csv_path)

    # Load eval dataset
    dataset_path = Path("data/processed/eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # Map question -> expected arxiv_id
    q_to_arxiv = {item["question"]: item.get("arxiv_id") for item in eval_data}

    store = FAISSStore()

    # Filter to fallback rows (web_search_used = True)
    fallback_df = df[df["web_search_used"] == True]

    print("=" * 70)
    print(f"ANALYZING {len(fallback_df)} FALLBACK ROWS (web_search_used = True)")
    print("=" * 70)

    grader_rejections = 0
    retriever_misses = 0

    rejection_rows = []
    miss_rows = []

    for idx, row in fallback_df.iterrows():
        question = row["question"]
        expected_arxiv_id = q_to_arxiv.get(question)
        pipeline_grade = row["pipeline_grade"]

        # Run direct FAISS search
        results = store.search(question, top_k=5)
        top_ids = [r["arxiv_id"] for r in results]

        is_retrieved = expected_arxiv_id in top_ids

        info = {
            "index": idx,
            "question": question,
            "expected_arxiv_id": expected_arxiv_id,
            "pipeline_grade": pipeline_grade,
            "top_retrieved_ids": top_ids,
            "top_scores": [r["score"] for r in results]
        }

        if is_retrieved:
            grader_rejections += 1
            rejection_rows.append(info)
        else:
            retriever_misses += 1
            miss_rows.append(info)

    print(f"Summary:")
    print(f"  Total Fallback Rows         : {len(fallback_df)}")
    print(f"  Retriever Misses            : {retriever_misses} ({retriever_misses/len(fallback_df)*100:.1f}%)")
    print(f"    (The target paper was NOT in the top-5 retrieved chunks, fallback was correct)")
    print(f"  Grader Rejections           : {grader_rejections} ({grader_rejections/len(fallback_df)*100:.1f}%)")
    print(f"    (The target paper WAS in the top-5 local chunks, but grader triggered web fallback)")
    print("-" * 70)

    if grader_rejections > 0:
        print("\nDETAILED GRADER REJECTIONS (Potential Over-conservatism):")
        for r in rejection_rows:
            print(f"\n[Index {r['index']}] Question: {r['question']}")
            print(f"  Expected ID: {r['expected_arxiv_id']} | Pipeline Grade: {r['pipeline_grade'].upper()}")
            print(f"  Top 5 FAISS IDs: {r['top_retrieved_ids']}")
            print(f"  Top 5 Scores: {r['top_scores']}")
    
    if retriever_misses > 0:
        print("\nSAMPLE RETRIEVER MISSES (Retriever didn't surface expected paper):")
        for r in miss_rows[:5]:  # print first 5 sample misses
            print(f"\n[Index {r['index']}] Question: {r['question']}")
            print(f"  Expected ID: {r['expected_arxiv_id']} | Pipeline Grade: {r['pipeline_grade'].upper()}")
            print(f"  Top 5 FAISS IDs: {r['top_retrieved_ids']}")
            print(f"  Top 5 Scores: {r['top_scores']}")

if __name__ == "__main__":
    analyze_fallbacks()
