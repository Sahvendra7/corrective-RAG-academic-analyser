"""
LangGraph wiring of the Corrective RAG pipeline.

    START → retriever → grader
      grader ─ relevant ──────────────────────────────────────────► generator
             ─ grader failed ─────────────────────────────────────► generator (unverified passages)
             ─ ambiguous, rewrites left ──► rewriter → retriever (loop)
             ─ irrelevant, web unavailable, rewrites left ─► rewriter → retriever (loop)
             ─ otherwise ─────────────────► web_search ──────────► generator
      generator ─ answer written ─► hallucination_checker → END
                ─ abstained / failed ─────────────────────────────► END

Termination: the only cycle is rewriter → retriever → grader, and the
rewriter increments ``retry_count``, which routing compares with
``Settings.max_rewrites`` (≤ 5). ``recursion_limit`` is derived from that bound
and passed explicitly, so a routing bug fails loudly instead of looping.

LLM calls per question: 1 grading call per retrieval round, 1 per rewrite,
≤1 web grading call, 1 generation call, 1 verification call. With the
defaults the worst case is 3 + 2 + 1 + 1 + 1 = 8 calls (v0.1: up to
3 × (2 + top_k) + top_k + 2).

Usage::

    from src.pipeline.graph import run_query
    result = run_query("What does the CRAG retrieval evaluator do?")
    print(result["generation"])
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.pipeline.llm_factory import LLMNotConfiguredError
from src.pipeline.nodes.generator import generator_node
from src.pipeline.nodes.grader import grader_node
from src.pipeline.nodes.hallucination import hallucination_node
from src.pipeline.nodes.retriever import retriever_node
from src.pipeline.nodes.rewriter import rewriter_node
from src.pipeline.nodes.web_search import web_search_node
from src.pipeline.state import GRADE_RELEVANT, CRAGState, create_initial_state

logger = logging.getLogger(__name__)

LLM_NOT_CONFIGURED = "The language model is not configured (set GEMINI_API_KEY); no answer was generated."

NODE_RETRIEVER = "retriever"
NODE_GRADER = "grader"
NODE_REWRITER = "rewriter"
NODE_WEB_SEARCH = "web_search"
NODE_GENERATOR = "generator"
NODE_HALLUCINATION = "hallucination_checker"


def web_available(state: CRAGState) -> bool:
    settings = get_settings()
    return bool(settings.web_search_enabled and state.get("allow_web_search", True) and settings.tavily_key())


def route_after_grader(state: CRAGState) -> str:
    grade = state.get("grade", "")
    rewrites_left = state.get("retry_count", 0) < get_settings().max_rewrites
    if grade == GRADE_RELEVANT or state.get("grader_failed"):
        return NODE_GENERATOR
    if rewrites_left and (grade != "irrelevant" or not web_available(state)):
        return NODE_REWRITER
    return NODE_WEB_SEARCH


def route_after_generator(state: CRAGState) -> str:
    return NODE_HALLUCINATION if state.get("evidence") and state.get("answer_text") else END


def recursion_limit() -> int:
    """Upper bound on node executions: 3 per retrieval round + web + generate + verify + slack."""
    return 3 * (get_settings().max_rewrites + 1) + 6


def build_graph():
    graph = StateGraph(CRAGState)
    graph.add_node(NODE_RETRIEVER, retriever_node)
    graph.add_node(NODE_GRADER, grader_node)
    graph.add_node(NODE_REWRITER, rewriter_node)
    graph.add_node(NODE_WEB_SEARCH, web_search_node)
    graph.add_node(NODE_GENERATOR, generator_node)
    graph.add_node(NODE_HALLUCINATION, hallucination_node)

    graph.add_edge(START, NODE_RETRIEVER)
    graph.add_edge(NODE_RETRIEVER, NODE_GRADER)
    graph.add_conditional_edges(
        NODE_GRADER,
        route_after_grader,
        {NODE_GENERATOR: NODE_GENERATOR, NODE_WEB_SEARCH: NODE_WEB_SEARCH, NODE_REWRITER: NODE_REWRITER},
    )
    graph.add_edge(NODE_REWRITER, NODE_RETRIEVER)
    graph.add_edge(NODE_WEB_SEARCH, NODE_GENERATOR)
    graph.add_conditional_edges(
        NODE_GENERATOR, route_after_generator, {NODE_HALLUCINATION: NODE_HALLUCINATION, END: END}
    )
    graph.add_edge(NODE_HALLUCINATION, END)
    return graph.compile()


_cached_graph = None


def _get_cached_graph():
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_graph()
    return _cached_graph


def graph_config() -> dict[str, Any]:
    return {"recursion_limit": recursion_limit()}


def run_query(
    query: str,
    verbose: bool = False,
    paper_ids: list[str] | None = None,
    allow_web_search: bool = True,
) -> CRAGState:
    """Run one question through the pipeline and return the final state.

    Exceptions inside the graph are converted into a state with ``error`` set
    and a generic message, so callers always receive a state dict.
    """
    initial = create_initial_state(query, paper_filter=paper_ids, allow_web_search=allow_web_search)
    try:
        final = _get_cached_graph().invoke(initial, config=graph_config())
    except LLMNotConfiguredError as exc:
        logger.error("[GRAPH] %s", exc)
        return {**initial, "generation": LLM_NOT_CONFIGURED, "error": type(exc).__name__}
    except Exception as exc:
        logger.exception("[GRAPH] pipeline error")
        return {**initial, "generation": "The pipeline failed to answer this question.", "error": type(exc).__name__}
    if verbose:
        _print_summary(final)
    return final


def _print_summary(state: CRAGState) -> None:
    print(f"\nQuery   : {state['query']}")
    print(f"Grade   : {state.get('grade')}  rewrites={state.get('retry_count')}  web={state.get('web_search_used')}")
    print(f"Source  : {state.get('source')}  LLM calls={state.get('llm_calls')}")
    for t in state.get("trace", []):
        print(f"  - {t['node']:<22} {t['ms']:>8.1f} ms  {t['detail']}")
    for w in state.get("warnings", []):
        print(f"  ! {w}")
    print("\n" + state.get("generation", ""))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    run_query(" ".join(sys.argv[1:]) or "How does Corrective RAG handle irrelevant retrieved documents?", verbose=True)
