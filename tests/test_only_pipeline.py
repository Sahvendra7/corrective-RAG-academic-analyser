"""
test_single_query.py
--------------------
A lightweight script to test the CRAG pipeline's LLM routing, 
grading, and generation on a single question without re-running 
the document chunking or embedding steps.
"""

import os
import time
import json
from dotenv import load_dotenv

import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# Load API keys from your .env file
load_dotenv()

# Map the Gemini key for LangChain just like in the main test script
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

from src.pipeline.graph import run_query

def test_single_query(query: str):
    print(f"\n{'='*70}")
    print(f"  Testing CRAG Pipeline: Single Query Execution")
    print(f"{'='*70}")
    print(f"  Query: '{query}'\n")

    try:
        start_time = time.time()
        
        # Fire the query at the pipeline
        print(" ⏳ Running pipeline (this may take ~20 seconds due to grader throttling)...")
        result = run_query(query, verbose=False)
        
        elapsed = time.time() - start_time

        # Extract telemetry
        generation = result.get("generation", "No generation found.")
        grade = result.get("grade", "N/A")
        retries = result.get("retry_count", 0)
        web_search = result.get("web_search_used", False)
        source = result.get("source", "N/A")
        docs = result.get("documents", [])

        print(f"\n  PIPELINE EXECUTED SUCCESSFULLY!")
        print(f"  Time Taken       : {elapsed:.2f} seconds")
        print(f"  Final Grade      : {grade.upper()}")
        print(f"  Retries          : {retries}")
        print(f"  Web Search Used  : {web_search}")
        print(f"  Data Source      : {source}")
        print(f"  Chunks Retrieved : {len(docs)}")

        print(f"\n {'-'*68}")
        print(f"  GENERATED ANSWER:")
        print(f" {'-'*68}")
        # Format the text nicely for the terminal
        print(f"\n{generation}\n")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n  PIPELINE FAILED: {e}\n")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # CHOOSE YOUR TEST METHOD (Uncomment the one you want)
    # ---------------------------------------------------------
    
    # OPTION 1: Use a custom hardcoded question
    test_question = "How does Corrective RAG handle irrelevant retrieved documents?"
    
    # OPTION 2: Pull the very first question from your generated eval dataset
    """
    dataset_path = "data/processed/eval_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    test_question = eval_data[0]["question"]
    """

    test_single_query(test_question)