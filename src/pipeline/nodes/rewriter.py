"""
Query rewriter: called when local retrieval was *ambiguous* and rewrite budget
(``Settings.max_rewrites``) remains.

One LLM call per rewrite. v0.1 first generated a "HyDE" hypothetical passage
and then pasted it into a second prompt; the passage was never embedded, so it
was not HyDE and only added a call per retry.

Each rewrite starts from the *original* question (not from the previous
rewrite) so repeated rewriting cannot drift away from what the user asked.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.pipeline.llm_factory import get_creative_llm, response_text
from src.pipeline.state import CRAGState, trace_entry

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 400

REWRITER_SYSTEM_PROMPT = """You rewrite questions to improve passage retrieval from a database of \
machine-learning research papers.

Given the question and feedback on why the retrieved passages did not answer it:
- add the technical terminology a paper answering it would use
- expand abbreviations (e.g. "RAG" → "retrieval-augmented generation (RAG)")
- keep every named paper, method, dataset and number from the question
- do not change what is being asked

Output only the rewritten query (under 400 characters), no explanation or quotes."""

REWRITER_HUMAN_PROMPT = """Question: {query}

Why the retrieved passages were not sufficient:
{feedback}

Previous rewrites (avoid repeating them): {previous}

Rewritten query:"""


def build_feedback_string(document_grades: list[dict]) -> str:
    lines = [f"- {g['grade']}: {g['reasoning']}" for g in document_grades if g.get("grade") != "relevant"]
    return "\n".join(lines) or "No passage directly answered the question."


def rewrite_query(query: str, feedback: str, previous: str, llm) -> str:
    messages = [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(content=REWRITER_HUMAN_PROMPT.format(query=query, feedback=feedback, previous=previous or "none")),
    ]
    try:
        rewritten = response_text(llm.invoke(messages)).strip().strip('"').strip()
    except Exception as exc:
        logger.error("[REWRITER] rewrite failed, keeping original query: %s", exc)
        return query
    if not rewritten:
        return query
    return rewritten[:MAX_QUERY_CHARS]


def rewriter_node(state: CRAGState) -> dict:
    started = time.perf_counter()
    query = state["query"]
    rewritten = rewrite_query(
        query,
        build_feedback_string(state.get("document_grades", [])),
        state.get("rewritten_query", ""),
        get_creative_llm(),
    )
    retry_count = state.get("retry_count", 0) + 1
    logger.info("[REWRITER] attempt %d: %r", retry_count, rewritten[:100])
    return {
        "rewritten_query": rewritten,
        "retry_count": retry_count,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "trace": trace_entry(state, "rewriter", started, rewritten[:120]),
    }
