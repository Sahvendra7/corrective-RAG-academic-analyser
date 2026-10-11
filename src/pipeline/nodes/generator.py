"""
Generator: writes the answer from the selected evidence.

Evidence selection (CRAG actions)
--------------------------------
* local retrieval **correct** (≥1 relevant passage) → relevant local passages only
* otherwise → ambiguous local passages + relevant/ambiguous web passages
* grader failed → the retrieved passages, flagged as unverified

With no evidence the node **abstains** with a fixed message and makes no LLM
call; it never answers from the model's parametric memory.

The model cites with ``[n]`` markers only; invalid markers are removed and the
"Sources" list is rendered from passage metadata (see :mod:`src.pipeline.evidence`).
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.pipeline.evidence import (
    UNTRUSTED_NOTICE,
    build_citations,
    format_evidence,
    is_web,
    normalise_citations,
    render_sources,
)
from src.pipeline.llm_factory import get_generator_llm, response_text
from src.pipeline.state import (
    GRADE_AMBIGUOUS,
    GRADE_IRRELEVANT,
    GRADE_RELEVANT,
    GRADE_UNGRADED,
    SOURCE_FAISS,
    SOURCE_MIXED,
    SOURCE_NONE,
    SOURCE_WEB,
    CRAGState,
    Document,
    trace_entry,
)

logger = logging.getLogger(__name__)

ABSTAIN_MESSAGE = (
    "I could not find evidence in the indexed papers{web} that answers this question, "
    "so I am not answering it. Try rephrasing, naming the paper, or widening the paper filter."
)
GENERATION_FAILED = "The answer could not be generated because the language model call failed."
NO_CITATIONS = "The answer contains no citations; treat it as unsupported."

GENERATOR_SYSTEM_PROMPT = f"""You are a research assistant answering questions about machine-learning papers.

Rules:
1. Use ONLY the information in the <source> passages. Do not use outside knowledge.
2. Support every factual sentence with citation markers such as [1] or [2][3], using the source ids.
3. Cite only a source that actually states the claim. Never cite a source for something it does not say.
4. Sources with type="web" are web pages, not the papers; say so when you rely on them.
5. If the passages answer only part of the question, answer that part and state clearly what is not covered.
   If they do not answer it at all, say that the provided sources do not contain the answer.
6. If sources disagree, report the disagreement and cite both sides.
7. Be precise and concise. Do not add a list of sources or references at the end; it is generated separately.
{UNTRUSTED_NOTICE}"""

GENERATOR_HUMAN_PROMPT = """Question: {query}

{sources}

Answer the question using only these sources, with [n] citations."""

MAX_CONTEXT_DOCS = 6  # v0.1 name; the active cap is Settings.max_evidence_docs


def select_evidence(state: CRAGState) -> list[Document]:
    settings = get_settings()
    docs = state.get("documents", [])
    local = [d for d in docs if not is_web(d)]
    web = [d for d in docs if is_web(d)]
    relevant = [d for d in local if d.get("grade") == GRADE_RELEVANT]
    if relevant:
        chosen = relevant
    elif state.get("grader_failed"):
        chosen = [d for d in local if d.get("grade") in (GRADE_UNGRADED, GRADE_AMBIGUOUS, GRADE_RELEVANT)]
    else:
        chosen = [d for d in local if d.get("grade") == GRADE_AMBIGUOUS]
        chosen += sorted(
            [d for d in web if d.get("grade") in (GRADE_RELEVANT, GRADE_AMBIGUOUS)],
            key=lambda d: d.get("grade") != GRADE_RELEVANT,
        )
    return [d for d in chosen if d.get("grade") != GRADE_IRRELEVANT][: settings.max_evidence_docs]


def source_label(evidence: list[Document]) -> str:
    has_web = any(is_web(d) for d in evidence)
    has_local = any(not is_web(d) for d in evidence)
    if has_web and has_local:
        return SOURCE_MIXED
    return SOURCE_WEB if has_web else SOURCE_FAISS if has_local else SOURCE_NONE


def generate_answer(query: str, evidence: list[Document], llm) -> str:
    """Raw model answer (raises on LLM failure)."""
    messages = [
        SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=GENERATOR_HUMAN_PROMPT.format(
                query=query, sources=format_evidence(evidence, get_settings().max_evidence_chars)
            )
        ),
    ]
    answer = response_text(llm.invoke(messages))
    return answer.removeprefix("<answer>").removesuffix("</answer>").strip()


def generator_node(state: CRAGState) -> dict:
    started = time.perf_counter()
    warnings = list(state.get("warnings", []))
    evidence = select_evidence(state)

    if not evidence:
        web_note = "" if not state.get("web_search_used") else " or in the web search results"
        message = ABSTAIN_MESSAGE.format(web=web_note)
        return {
            "evidence": [],
            "generation": message,
            "answer_text": message,
            "citations": [],
            "source": SOURCE_NONE,
            "generation_ms": 0.0,
            "trace": trace_entry(state, "generator", started, "abstained: no usable evidence"),
        }

    gen_started = time.perf_counter()
    try:
        raw = generate_answer(state["query"], evidence, get_generator_llm())
    except Exception as exc:
        logger.error("[GENERATOR] generation failed: %s", exc)
        warnings.append(GENERATION_FAILED)
        return {
            "evidence": evidence,
            "generation": GENERATION_FAILED,
            "answer_text": "",
            "citations": [],
            "source": source_label(evidence),
            "error": f"generation failed: {type(exc).__name__}",
            "warnings": warnings,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "generation_ms": (time.perf_counter() - gen_started) * 1000,
            "trace": trace_entry(state, "generator", started, "LLM call failed"),
        }
    generation_ms = (time.perf_counter() - gen_started) * 1000

    answer, cited, invalid = normalise_citations(raw, len(evidence))
    citations = build_citations(evidence, cited)
    if not cited and NO_CITATIONS not in warnings:
        warnings.append(NO_CITATIONS)
    if invalid:
        logger.warning("[GENERATOR] removed citations to non-existent sources: %s", invalid)

    sources_md = render_sources(citations)
    generation = answer + (f"\n\n{sources_md}" if sources_md else "")
    detail = f"{len(evidence)} passages, cited {cited or 'none'}" + (f", removed invalid {invalid}" if invalid else "")
    return {
        "evidence": evidence,
        "generation": generation,
        "answer_text": answer,
        "citations": citations,
        "source": source_label(evidence),
        "grounding": {**state.get("grounding", {}), "invalid_citations": invalid},
        "warnings": warnings,
        "llm_calls": state.get("llm_calls", 0) + 1,
        "generation_ms": generation_ms,
        "trace": trace_entry(state, "generator", started, detail),
    }
