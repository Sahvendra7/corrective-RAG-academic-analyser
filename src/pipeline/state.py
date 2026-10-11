"""
State shared by the nodes of the CRAG LangGraph pipeline.

Nodes return partial dicts that LangGraph merges into the state (list fields
are *replaced*, not appended, so nodes always return the full updated list).

Key fields
----------
query / rewritten_query   original question and the latest rewrite
documents                 every passage seen so far (local and web), each with
                          a ``grade``; de-duplicated by ``chunk_id``
relevant_documents        local passages graded "relevant" (recomputed, never appended)
evidence                  the numbered passages the generator was given; citation
                          ``[n]`` refers to ``evidence[n-1]``
citations                 evidence entries actually cited in the answer
grade                     overall decision: relevant | ambiguous | irrelevant
retry_count               query rewrites performed (bounded by Settings.max_rewrites)
grounding                 result of the grounding check (see nodes/hallucination.py)
trace                     one entry per node execution: node, duration, short detail
warnings                  user-visible problems (index missing, grader failure, …)
"""

from __future__ import annotations

from typing import Any, TypedDict

# ── Grade / source constants ──────────────────────────────────────────────────

GRADE_RELEVANT = "relevant"
GRADE_AMBIGUOUS = "ambiguous"
GRADE_IRRELEVANT = "irrelevant"
GRADE_UNGRADED = "ungraded"  # grader failed; passage is unverified

SOURCE_FAISS = "faiss"
SOURCE_WEB = "web"
SOURCE_MIXED = "mixed"
SOURCE_NONE = "none"


def __getattr__(name: str) -> Any:
    # v0.1 exposed a hard-coded MAX_RETRIES = 3 here, while config.py said 2 and
    # the retriever used yet another number. The single source of truth is now
    # Settings.max_rewrites; this alias keeps old imports working.
    if name == "MAX_RETRIES":
        from src.config import get_settings

        return get_settings().max_rewrites
    raise AttributeError(name)


class Document(TypedDict, total=False):
    """A retrieved passage (local chunk or web result) with provenance."""

    chunk_id: str
    arxiv_id: str  # "" for web results
    text: str
    score: float  # cosine similarity (local) or Tavily score (web); not a probability
    title: str
    authors: list[str]
    published: str
    url: str
    section: str
    heading: str
    page_start: int | None
    page_end: int | None
    grade: str
    grade_reason: str
    source: str  # "faiss" | "web"


class CRAGState(TypedDict, total=False):
    # Input
    query: str
    paper_filter: list[str]
    allow_web_search: bool

    # Retrieval
    rewritten_query: str
    documents: list[Document]
    relevant_documents: list[Document]

    # Grading
    grade: str
    document_grades: list[dict[str, Any]]
    grader_failed: bool

    # Generation
    evidence: list[Document]
    generation: str
    answer_text: str
    citations: list[dict[str, Any]]

    # Grounding
    hallucination: bool
    hallucination_reasoning: str
    grounding: dict[str, Any]

    # Control flow / observability
    retry_count: int
    web_search_used: bool
    source: str
    error: str
    warnings: list[str]
    trace: list[dict[str, Any]]
    llm_calls: int
    retrieval_ms: float
    generation_ms: float


def create_initial_state(
    query: str,
    paper_filter: list[str] | None = None,
    allow_web_search: bool = True,
) -> CRAGState:
    """Fresh state for one question. Raises ``ValueError`` for an empty query."""
    if not query or not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")
    return CRAGState(
        query=query.strip(),
        paper_filter=list(paper_filter or []),
        allow_web_search=allow_web_search,
        rewritten_query="",
        documents=[],
        relevant_documents=[],
        grade="",
        document_grades=[],
        grader_failed=False,
        evidence=[],
        generation="",
        answer_text="",
        citations=[],
        hallucination=False,
        hallucination_reasoning="",
        grounding={},
        retry_count=0,
        web_search_used=False,
        source=SOURCE_NONE,
        error="",
        warnings=[],
        trace=[],
        llm_calls=0,
        retrieval_ms=0.0,
        generation_ms=0.0,
    )


def trace_entry(state: CRAGState, node: str, started: float, detail: str) -> list[dict[str, Any]]:
    """Return ``state['trace']`` with one more entry (for nodes to put in their update)."""
    import time

    return [
        *state.get("trace", []),
        {"node": node, "ms": round((time.perf_counter() - started) * 1000, 1), "detail": detail},
    ]
