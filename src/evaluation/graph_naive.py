"""
graph_naive.py
---------------
Wires a pure NAIVE RAG pipeline into a LangGraph state machine.

This orchestrator defines a strict, straight-line baseline:
    1. Retrieve documents from the vector store.
    2. Blindly trust them (no scoring, no filtering).
    3. Generate an answer.

Pipeline flow:
    START
      ↓
    custom_naive_retriever
      ↓
    generator_node
      ↓
    END

Usage:
    from src.evaluation.graph_naive import run_naive_query

    result = run_naive_query("How does naive RAG work?")
    print(result["generation"])
"""

import logging
import sys
from pathlib import Path

from langgraph.graph import StateGraph, START, END

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import your EXACT SAME state to avoid Pydantic/LangGraph TypeErrors
from src.pipeline.state import CRAGState, create_initial_state

# 🚨 IMPORT YOUR RAW RETRIEVER HERE 🚨
# Change this path if your vectorstore is located somewhere else
from src.vectorstore.faiss_store import FAISSStore

from langchain_core.prompts import ChatPromptTemplate
from src.pipeline.llm_factory import get_generator_llm
store = FAISSStore()
# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Node Name Constants ───────────────────────────────────────────────────────

NODE_RETRIEVER = "naive_retriever"
NODE_GENERATOR = "generator"


# ── Custom Naive Nodes ────────────────────────────────────────────────────────

def custom_naive_generator(state: CRAGState) -> dict:
    """A totally isolated generator that doesn't care about grades."""
    docs = state.get("documents", [])
    query = state.get("query", "")
    
    # 1. Format the raw text
    context_str = "\n\n".join(doc.get("text", "") for doc in docs)
    
    # 2. Build a simple prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Answer the question using ONLY the provided context.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    # 3. Generate and return
    chain = prompt | get_generator_llm()
    import time
    start_time = time.perf_counter()
    response = chain.invoke({"context": context_str, "question": query})
    generation_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        "generation": response.content,
        "generation_ms": generation_ms
    }

def custom_naive_retriever(state: CRAGState) -> dict:
    """
    A dedicated naive retriever node that blindly fetches top 5 results.
    """
    query = state.get("query", "")
    logger.info(f"[NAIVE GRAPH] Retrieving raw documents for: '{query}'")
    
    # 1. Fetch the raw documents
    import time
    start_time = time.perf_counter()
    docs = store.search(query, top_k=5) 
    retrieval_ms = (time.perf_counter() - start_time) * 1000
    
    # 2. THE FIX: Rubber-stamp every document as "relevant"
    # This spoofs the CRAG grader so the generator accepts them blindly!
    for doc in docs:
        doc["grade"] = "relevant" 
    
    logger.info(f"[NAIVE GRAPH] Blindly passing {len(docs)} documents to generator.")
    
    return {
        "documents": docs,
        "source": "faiss_naive",
        "retrieval_ms": retrieval_ms
    }


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_naive_graph() -> StateGraph:
    """
    Build and compile the Naive LangGraph state machine.
    """
    logger.info("[NAIVE GRAPH] Building pure Naive RAG pipeline...")

    # Step 1: Create the graph with the EXACT SAME CRAG schema
    graph = StateGraph(CRAGState)

    # Step 2: Add our two nodes
    graph.add_node(NODE_RETRIEVER, custom_naive_retriever)
    graph.add_node(NODE_GENERATOR, custom_naive_generator)

    # Step 3: Add straight-line edges (No conditional routing!)
    graph.add_edge(START, NODE_RETRIEVER)
    graph.add_edge(NODE_RETRIEVER, NODE_GENERATOR)
    graph.add_edge(NODE_GENERATOR, END)

    # Step 4: Set entry point
    graph.set_entry_point(NODE_RETRIEVER)

    # Step 5: Compile
    compiled = graph.compile()
    logger.info("[NAIVE GRAPH] Graph compiled successfully")

    return compiled

# ── Query Runner ──────────────────────────────────────────────────────────────

def run_naive_query(query: str, verbose: bool = False) -> CRAGState:
    """
    Run a single query through the Naive RAG pipeline.
    """
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"[NAIVE GRAPH] New query: '{query}'")
        logger.info(f"{'='*60}")

    graph = build_naive_graph()
    
    # Use your existing state creator to ensure all keys are correctly initialized
    initial_state = create_initial_state(query)

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"[NAIVE GRAPH] Pipeline error: {e}")
        return {
            **initial_state,
            "generation" : f"Pipeline error: {str(e)}",
            "error"      : str(e),
        }

    return final_state

# ── Visualise Graph ───────────────────────────────────────────────────────────

def print_naive_structure():
    """Print a text representation of the naive graph structure."""
    print("""
Naive Pipeline Graph Structure:
================================

  START
    │
    ▼
[naive_retriever]
    │
    ▼
[generator]
    │
    ▼
   END

Nodes   : naive_retriever, generator
Edges   : 3 total (All fixed, 0 conditional)
Max path: START -> naive_retriever -> generator -> END
""")

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_naive_structure()

    test_query = "What is the baseline capability of this framework?"
    print(f"Running test query: '{test_query}'\n")
    
    result = run_naive_query(test_query, verbose=True)

    print("\nEXECUTION TRACE:")
    print(f"  Documents retrieved : {len(result.get('documents', []))}")
    print(f"  Source              : {result.get('source', 'N/A')}")
    print(f"  Answer Preview      : {result.get('generation', '')[:100]}...")