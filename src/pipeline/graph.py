"""
graph.py
---------
Wires all CRAG pipeline nodes into a LangGraph state machine.

This is the orchestrator — it defines:
    1. Which nodes exist
    2. Which node runs first
    3. What the conditional routing logic is
       (which node to go to next based on state)
    4. Where the pipeline ends

Pipeline flow:
    START
      ↓
    retriever_node
      ↓
    grader_node
      ↓ (conditional routing based on state["grade"])
      ├── "relevant"   → generator_node → hallucination_node → END
      ├── "ambiguous"  → (retry < MAX) → rewriter_node → retriever_node (loop)
      │                → (retry >= MAX) → web_search_node → generator_node → hallucination_node → END
      └── "irrelevant" → web_search_node → generator_node → hallucination_node → END

Usage:
    from src.pipeline.graph import build_graph, run_query

    # Option A: Run a query directly
    result = run_query("How does corrective RAG work?")
    print(result["generation"])

    # Option B: Use the compiled graph directly
    graph  = build_graph()
    state  = create_initial_state("How does corrective RAG work?")
    result = graph.invoke(state)
"""

import logging

from langgraph.graph import StateGraph, START, END


from src.pipeline.state import (
    CRAGState,
    create_initial_state,
    GRADE_RELEVANT,
    GRADE_AMBIGUOUS,
    GRADE_IRRELEVANT,
    MAX_RETRIES,
)
from src.pipeline.nodes.retriever    import retriever_node
from src.pipeline.nodes.grader       import grader_node
from src.pipeline.nodes.rewriter     import rewriter_node
from src.pipeline.nodes.web_search   import web_search_node
from src.pipeline.nodes.generator    import generator_node
from src.pipeline.nodes.hallucination import hallucination_node

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Node Name Constants ───────────────────────────────────────────────────────
# Use constants instead of raw strings to avoid typos

NODE_RETRIEVER     = "retriever"
NODE_GRADER        = "grader"
NODE_REWRITER      = "rewriter"
NODE_WEB_SEARCH    = "web_search"
NODE_GENERATOR     = "generator"
NODE_HALLUCINATION = "hallucination"


# ── Conditional Routing Functions ─────────────────────────────────────────────

def route_after_grader(state: CRAGState) -> str:
    """
    Conditional edge function called after the grader node.

    Reads state["grade"] and state["retry_count"] to decide
    which node to run next.

    Routing logic:
        "relevant"   → generator (we have good docs, generate answer)
        "irrelevant" → web_search (local KB failed, try the web)
        "ambiguous"  → rewriter if retries remain, else web_search
                       (try to improve the query before giving up on local KB)

    Args:
        state: Current CRAGState

    Returns:
        Name of the next node to run
    """
    grade       = state.get("grade", "")
    retry_count = state.get("retry_count", 0)
    error       = state.get("error", "")

    # Short-circuit to generator if there's an error
    # — don't keep retrying if something is broken
    if error:
        logger.warning(f"[GRAPH] Error in state — routing to generator: {error}")
        return NODE_GENERATOR

    if grade == GRADE_RELEVANT:
        logger.info("[GRAPH] Grade=RELEVANT → routing to generator")
        return NODE_GENERATOR

    elif grade == GRADE_IRRELEVANT:
        logger.info("[GRAPH] Grade=IRRELEVANT → routing to web_search")
        return NODE_WEB_SEARCH

    elif grade == GRADE_AMBIGUOUS:
        if retry_count < MAX_RETRIES:
            logger.info(
                f"[GRAPH] Grade=AMBIGUOUS, retry={retry_count}/{MAX_RETRIES} "
                f"→ routing to rewriter"
            )
            return NODE_REWRITER
        else:
            logger.info(
                f"[GRAPH] Grade=AMBIGUOUS, max retries reached ({MAX_RETRIES}) "
                f"→ routing to web_search"
            )
            return NODE_WEB_SEARCH

    else:
        # Unknown grade — fall back safely to web search
        logger.warning(f"[GRAPH] Unknown grade '{grade}' → defaulting to web_search")
        return NODE_WEB_SEARCH


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the CRAG LangGraph state machine.

    Steps:
        1. Create a StateGraph with CRAGState schema
        2. Add all nodes
        3. Add edges (fixed and conditional)
        4. Set entry point
        5. Compile and return

    Returns:
        Compiled LangGraph StateGraph ready to invoke
    """
    logger.info("[GRAPH] Building CRAG pipeline graph...")

    # Step 1: Create the graph with our state schema
    graph = StateGraph(CRAGState)

    # ── Step 2: Add all nodes ─────────────────────────────────────────────────
    # Each node is a Python function that takes state and returns
    # a dict of state fields to update

    graph.add_node(NODE_RETRIEVER,     retriever_node)
    graph.add_node(NODE_GRADER,        grader_node)
    graph.add_node(NODE_REWRITER,      rewriter_node)
    graph.add_node(NODE_WEB_SEARCH,    web_search_node)
    graph.add_node(NODE_GENERATOR,     generator_node)
    graph.add_node(NODE_HALLUCINATION, hallucination_node)

    logger.info(f"[GRAPH] Added {6} nodes")

    # ── Step 3: Add edges ─────────────────────────────────────────────────────

    # Fixed edge: START → retriever
    # Every query starts with retrieval
    graph.add_edge(START, NODE_RETRIEVER)

    # Fixed edge: retriever → grader
    # After retrieval, always grade the documents
    graph.add_edge(NODE_RETRIEVER, NODE_GRADER)

    # Conditional edge: grader → (generator | web_search | rewriter)
    # This is the core CRAG routing logic
    graph.add_conditional_edges(
        source=NODE_GRADER,
        path=route_after_grader,
        path_map={
            NODE_GENERATOR  : NODE_GENERATOR,
            NODE_WEB_SEARCH : NODE_WEB_SEARCH,
            NODE_REWRITER   : NODE_REWRITER,
        },
    )

    # Fixed edge: rewriter → retriever
    # After rewriting the query, go back to retrieval
    # This creates the retry loop: retriever → grader → rewriter → retriever
    graph.add_edge(NODE_REWRITER, NODE_RETRIEVER)

    # Fixed edge: web_search → generator
    # After web search, always generate an answer
    graph.add_edge(NODE_WEB_SEARCH, NODE_GENERATOR)

    # Fixed edge: generator → hallucination
    # After generation, always check for hallucinations
    graph.add_edge(NODE_GENERATOR, NODE_HALLUCINATION)

    # Fixed edge: hallucination → END
    # Currently always ends — the hallucination node annotates the answer
    # with a warning rather than discarding it, so we always proceed to END.
    graph.add_edge(NODE_HALLUCINATION, END)

    logger.info("[GRAPH] Edges configured")

    # ── Step 5: Compile ───────────────────────────────────────────────────────
    compiled = graph.compile()
    logger.info("[GRAPH] Graph compiled successfully")

    return compiled


# ── Cached Graph ──────────────────────────────────────────────────────────────

_cached_graph = None

def _get_cached_graph():
    """Return a cached compiled graph, building it once on first call."""
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_graph()
    return _cached_graph


# ── Query Runner ──────────────────────────────────────────────────────────────

def run_query(query: str, verbose: bool = True) -> CRAGState:
    """
    Run a single query through the complete CRAG pipeline.

    This is the main public API for the pipeline.
    Import and call this from your FastAPI routes or Streamlit UI.

    Args:
        query: The user's natural language question
        verbose: If True, print a formatted summary after completion

    Returns:
        Final CRAGState dict with all fields populated including
        state["generation"] which contains the final answer
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"[GRAPH] New query: '{query}'")
    logger.info(f"{'='*60}")

    # Use cached graph (built once, reused for all queries)
    graph = _get_cached_graph()

    # Create initial state
    initial_state = create_initial_state(query)

    # Run the pipeline
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.error(f"[GRAPH] Pipeline error: {e}")
        return {
            **initial_state,
            "generation" : f"Pipeline error: {str(e)}",
            "error"      : str(e),
        }

    # Print summary if verbose
    if verbose:
        _print_summary(query, final_state)

    return final_state


# ── Summary Printer ───────────────────────────────────────────────────────────

def _print_summary(query: str, state: CRAGState):
    """Print a formatted summary of the pipeline execution."""
    print(f"\n{'='*60}")
    print(f"CRAG PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Query          : {query}")
    print(f"Grade          : {state.get('grade', 'N/A').upper()}")
    print(f"Source         : {state.get('source', 'N/A')}")
    print(f"Retry count    : {state.get('retry_count', 0)}")
    print(f"Web search     : {state.get('web_search_used', False)}")
    print(f"Docs retrieved : {len(state.get('documents', []))}")
    print(f"Hallucination  : {state.get('hallucination', False)}")
    print(f"{'='*60}")
    print(f"\nANSWER:")
    print(f"{'─'*60}")
    print(state.get("generation", "No answer generated"))
    print(f"{'='*60}\n")


# ── Visualise Graph ───────────────────────────────────────────────────────────

def print_graph_structure():
    """
    Print a text representation of the graph structure.
    Useful for debugging and documentation.
    """
    print("""
CRAG Pipeline Graph Structure:
================================

START
  │
  ▼
[retriever] ──────────────────────────────────────┐
  │                                               │
  ▼                                               │
[grader]                                          │
  │                                               │
  ├── grade=RELEVANT ──────────────────────────►[generator]
  │                                               │
  ├── grade=IRRELEVANT ──────────────►[web_search]│
  │                                      │        │
  └── grade=AMBIGUOUS                    │        │
        │                                │        │
        ├── retry < 3 ──►[rewriter]──────┘        │
        │                    │                    │
        │                    └──────────────────► ┘
        │                                         │
        └── retry >= 3 ─────►[web_search]         │
                                  │               │
                                  └──────────────►│
                                                  │
                                              [hallucination]
                                                  │
                                                 END

Nodes   : retriever, grader, rewriter, web_search, generator, hallucination
Edges   : 7 total (4 fixed, 2 conditional)
Max path: START → retriever → grader → rewriter → retriever (×3) → web_search → generator → hallucination → END
""")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Print graph structure
    print_graph_structure()

    # Run a test query
    test_query = "How does Corrective RAG handle irrelevant retrieved documents?"

    print(f"Running test query: '{test_query}'\n")
    result = run_query(test_query, verbose=True)

    # Show execution trace
    print("\nEXECUTION TRACE:")
    print(f"  Documents retrieved : {len(result.get('documents', []))}")
    print(f"  Overall grade       : {result.get('grade', 'N/A')}")
    print(f"  Retry count         : {result.get('retry_count', 0)}")
    print(f"  Web search used     : {result.get('web_search_used', False)}")
    print(f"  Source              : {result.get('source', 'N/A')}")
    print(f"  Hallucination       : {result.get('hallucination', False)}")
    if result.get("hallucination_reasoning"):
        print(f"  Hal. reasoning      : {result['hallucination_reasoning'][:100]}")
