"""
Retrieval evaluator: grades passages as relevant / ambiguous / irrelevant.

All ungraded passages are graded in **one** structured LLM call. v0.1 made
one sequential call per passage (and again for every retry), which dominated
latency (30 s per CRAG query vs 1.8 s for naive RAG in the project's own
evaluation) and hit free-tier rate limits.

The overall grade is computed over *all* local passages graded so far, not
only the latest batch:

    relevant   at least one local passage is relevant
    irrelevant every local passage is irrelevant (or none was retrieved)
    ambiguous  otherwise

If the grading call fails, passages are marked ``ungraded`` and
``grader_failed`` is set; the graph then answers from the unverified passages
with a warning instead of looping through rewrites that would fail the same way.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_settings
from src.pipeline.evidence import UNTRUSTED_NOTICE, format_evidence, is_web
from src.pipeline.llm_factory import get_llm
from src.pipeline.state import (
    GRADE_AMBIGUOUS,
    GRADE_IRRELEVANT,
    GRADE_RELEVANT,
    GRADE_UNGRADED,
    CRAGState,
    Document,
    trace_entry,
)

logger = logging.getLogger(__name__)

GRADER_FAILED = "Relevance grading failed; the answer uses passages that were not checked for relevance."


class PassageGrade(BaseModel):
    id: int = Field(description="The id attribute of the <source> being graded")
    grade: Literal["relevant", "ambiguous", "irrelevant"]
    reason: str = Field(description="One short sentence")


class BatchGrades(BaseModel):
    grades: list[PassageGrade]


# Kept for backwards compatibility with code importing the v0.1 schema name.
DocumentGrade = PassageGrade

GRADER_SYSTEM_PROMPT = f"""You grade retrieved passages for a question-answering system over research papers.

For each <source>, decide whether it helps answer the question:
- "relevant": it contains information that directly answers the question or a necessary part of it
- "ambiguous": it is on the same topic but does not itself answer the question
- "irrelevant": it is unrelated to the question

Grade every source exactly once, using its id. Judge only the passage text, not the title alone.
{UNTRUSTED_NOTICE}"""

GRADER_HUMAN_PROMPT = """Question: {query}

{sources}

Return one grade per source id."""


def grade_documents(query: str, docs: list[Document], llm) -> list[tuple[str, str]] | None:
    """Grade ``docs`` in one call. Returns ``[(grade, reason), ...]`` or ``None`` on failure."""
    settings = get_settings()
    messages = [
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(content=GRADER_HUMAN_PROMPT.format(query=query, sources=format_evidence(docs, settings.max_evidence_chars))),
    ]
    try:
        result = llm.with_structured_output(BatchGrades).invoke(messages)
    except Exception as exc:
        logger.error("[GRADER] grading call failed: %s", exc)
        return None
    if result is None:
        logger.error("[GRADER] grader returned no structured output")
        return None
    by_id = {g.id: g for g in result.grades}
    out = []
    for n in range(1, len(docs) + 1):
        g = by_id.get(n)
        # A passage the model skipped is treated as ambiguous, not relevant.
        out.append((g.grade, g.reason) if g else (GRADE_AMBIGUOUS, "not graded by the model"))
    return out


def overall_grade(local_grades: list[str]) -> str:
    if not local_grades:
        return GRADE_IRRELEVANT
    if GRADE_RELEVANT in local_grades:
        return GRADE_RELEVANT
    if all(g == GRADE_IRRELEVANT for g in local_grades):
        return GRADE_IRRELEVANT
    return GRADE_AMBIGUOUS


def grade_pending(state: CRAGState, docs: list[Document]) -> tuple[list[Document], list[dict], bool, int]:
    """Grade every document without a grade. Returns (docs, new grade records, failed, llm_calls)."""
    pending = [i for i, d in enumerate(docs) if not d.get("grade")]
    if not pending:
        return docs, [], False, 0
    batch = [docs[i] for i in pending]
    grades = grade_documents(state["query"], batch, get_llm())
    docs = [dict(d) for d in docs]  # type: ignore[misc]
    records = []
    for i, result in zip(pending, grades or [(GRADE_UNGRADED, "grader unavailable")] * len(pending)):
        docs[i]["grade"], docs[i]["grade_reason"] = result
        records.append({"chunk_id": docs[i]["chunk_id"], "grade": result[0], "reasoning": result[1]})
    return docs, records, grades is None, 1


def grader_node(state: CRAGState) -> dict:
    started = time.perf_counter()
    docs, records, failed, calls = grade_pending(state, list(state.get("documents", [])))
    local = [d for d in docs if not is_web(d)]
    warnings = list(state.get("warnings", []))
    if failed and GRADER_FAILED not in warnings:
        warnings.append(GRADER_FAILED)
    grade = overall_grade([d["grade"] for d in local if d["grade"] != GRADE_UNGRADED])
    if failed and any(d["grade"] == GRADE_UNGRADED for d in local):
        grade = GRADE_AMBIGUOUS
    logger.info("[GRADER] %d graded, overall=%s%s", len(records), grade, " (grader failed)" if failed else "")
    return {
        "documents": docs,
        "relevant_documents": [d for d in local if d["grade"] == GRADE_RELEVANT],
        "document_grades": records,
        "grade": grade,
        "grader_failed": state.get("grader_failed", False) or failed,
        "warnings": warnings,
        "llm_calls": state.get("llm_calls", 0) + calls,
        "trace": trace_entry(
            state,
            "grader",
            started,
            f"overall={grade}; " + ", ".join(f"{r['grade']}" for r in records) if records else f"overall={grade}",
        ),
    }
