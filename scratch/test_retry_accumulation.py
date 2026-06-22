import os
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Set Gemini API Key from environment if not set
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

from src.pipeline.state import create_initial_state, GRADE_AMBIGUOUS, GRADE_RELEVANT
from unittest.mock import patch

# Mock grader node to force ambiguous on first pass (retry_count == 0)
# and relevant on second pass (retry_count == 1) to end it.
from src.pipeline.nodes.grader import grader_node as original_grader_node

def mock_grader_node(state):
    retry_count = state.get("retry_count", 0)
    print(f"\n[MOCK GRADER] grader_node called. retry_count: {retry_count}")
    # Run the original grader first to populate document/relevant_documents
    res = original_grader_node(state)
    if retry_count == 0:
        res["grade"] = GRADE_AMBIGUOUS
        print(f"[MOCK GRADER] Forcing GRADE_AMBIGUOUS to trigger retry loop.")
    else:
        res["grade"] = GRADE_RELEVANT
        print(f"[MOCK GRADER] Forcing GRADE_RELEVANT to finish pipeline.")
    return res

@patch("src.pipeline.graph.grader_node", side_effect=mock_grader_node)
def run_test(mock_grader):
    query = "How does Corrective RAG handle irrelevant retrieved documents?"
    print(f"Running pipeline for query: '{query}'")
    
    # We rebuild/recompile the graph inside run_test so it picks up the mock
    from src.pipeline.graph import build_graph
    graph = build_graph()
    initial_state = create_initial_state(query)
    
    start_time = time.perf_counter()
    final_state = graph.invoke(initial_state)
    total_latency_ms = (time.perf_counter() - start_time) * 1000
    
    print("\n" + "="*40)
    print("TEST RESULTS:")
    print("="*40)
    print(f"Retry count        : {final_state.get('retry_count')}")
    print(f"Retrieval latency  : {final_state.get('retrieval_ms'):.2f} ms")
    print(f"Generation latency : {final_state.get('generation_ms'):.2f} ms")
    print(f"Total loop latency : {total_latency_ms:.2f} ms")
    print(f"State keys         : {list(final_state.keys())}")
    print("="*40)

if __name__ == "__main__":
    run_test()
