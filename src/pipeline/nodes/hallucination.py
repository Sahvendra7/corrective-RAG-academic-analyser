"""
Grounding verifier: checks the generated answer against the evidence it was
generated from.

Two signals, reported separately in ``state["grounding"]``:

1. LLM claim check — the model lists claims not supported by the sources.
2. Deterministic number check — numbers in the answer that appear in no
   evidence passage (``unsupported_numbers``).

Neither proves an answer correct; a clean check means "no problem found".
If the verifier itself fails, the answer is marked *unverified*
(``hallucination=True``, ``checked=False``), never silently "grounded".

The verifier sees the answer **without** the generated Sources list and the
evidence rendered exactly as the generator saw it.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_settings
from src.pipeline.evidence import UNTRUSTED_NOTICE, format_evidence, unsupported_numbers
from src.pipeline.llm_factory import get_llm
from src.pipeline.state import CRAGState, trace_entry

logger = logging.getLogger(__name__)

VERIFY_FAILED = "The grounding check could not run; the answer is unverified."


class HallucinationGrade(BaseModel):
    hallucination: bool = Field(description="True if any claim in the answer is not supported by the sources")
    reasoning: str = Field(description="Brief explanation")
    unsupported_claims: list[str] = Field(default_factory=list, description="Unsupported claims, quoted briefly")


HALLUCINATION_SYSTEM_PROMPT = f"""You check whether an answer is supported by its sources.

A claim is unsupported if no source states it, if it contradicts a source, if it adds specifics
(numbers, names, dates, causal statements) absent from the sources, or if its citation marker [n]
points to a source that does not say it. Statements that the sources do not cover something are fine.
{UNTRUSTED_NOTICE}"""

HALLUCINATION_HUMAN_PROMPT = """Question: {query}

{sources}

Answer to check:
<answer>
{answer}
</answer>

List every unsupported claim."""


def check_hallucination(query: str, answer: str, sources: str, llm) -> dict:
    messages = [
        SystemMessage(content=HALLUCINATION_SYSTEM_PROMPT),
        HumanMessage(content=HALLUCINATION_HUMAN_PROMPT.format(query=query, sources=sources, answer=answer[:4000])),
    ]
    try:
        result = llm.with_structured_output(HallucinationGrade).invoke(messages)
        if result is None:
            raise ValueError("no structured output")
    except Exception as exc:
        logger.error("[VERIFIER] check failed: %s", exc)
        return {
            "checked": False,
            "verifier_failed": True,
            "hallucination": True,
            "reasoning": VERIFY_FAILED,
            "unsupported_claims": [],
        }
    claims = [c for c in result.unsupported_claims if c.strip()]
    return {
        "checked": True,
        "hallucination": bool(result.hallucination or claims),
        "reasoning": result.reasoning,
        "unsupported_claims": claims,
    }


def annotate_answer(generation: str, grounding: dict) -> str:
    notes = []
    if grounding.get("verifier_failed"):
        notes.append("⚠️ **Unverified**: the grounding check could not run.")
    elif grounding.get("unsupported_claims"):
        claims = "\n".join(f"  - {c}" for c in grounding["unsupported_claims"][:5])
        notes.append(f"⚠️ **Possibly unsupported by the sources** — verify before relying on:\n{claims}")
    if grounding.get("unsupported_numbers"):
        notes.append("⚠️ Numbers not found in any source: " + ", ".join(grounding["unsupported_numbers"][:8]))
    return generation if not notes else generation + "\n\n" + "\n\n".join(notes)


def hallucination_node(state: CRAGState) -> dict:
    started = time.perf_counter()
    evidence = state.get("evidence", [])
    answer = state.get("answer_text", "")
    grounding = dict(state.get("grounding", {}))
    warnings = list(state.get("warnings", []))

    if not evidence or not answer:
        grounding.update(checked=False, applicable=False)
        return {"grounding": grounding, "trace": trace_entry(state, "hallucination_checker", started, "skipped")}

    grounding["unsupported_numbers"] = unsupported_numbers(answer, evidence)
    calls = 0
    if get_settings().verify_grounding:
        sources = format_evidence(evidence, get_settings().max_evidence_chars)
        grounding.update(check_hallucination(state["query"], answer, sources, get_llm()))
        calls = 1
    else:
        grounding.update(checked=False, hallucination=False, reasoning="grounding check disabled", unsupported_claims=[])
    grounding["applicable"] = True
    if grounding.get("verifier_failed") and VERIFY_FAILED not in warnings:
        warnings.append(VERIFY_FAILED)

    detail = "unverified" if not grounding["checked"] else (
        f"{len(grounding['unsupported_claims'])} unsupported claims" if grounding["hallucination"] else "no unsupported claims found"
    )
    return {
        "hallucination": grounding["hallucination"],
        "hallucination_reasoning": grounding.get("reasoning", ""),
        "grounding": grounding,
        "generation": annotate_answer(state.get("generation", ""), grounding),
        "warnings": warnings,
        "llm_calls": state.get("llm_calls", 0) + calls,
        "trace": trace_entry(state, "hallucination_checker", started, detail),
    }
