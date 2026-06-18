"""
nodes/hallucination.py
-----------------------
The hallucination checking node in the CRAG pipeline.

Runs after the generator node as a quality gate.
Checks whether the generated answer is fully grounded
in the retrieved documents or contains fabricated claims.
"""

import logging
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.pipeline.state import CRAGState, Document
from src.pipeline.llm_factory import get_llm

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MAX_CONTEXT_LEN = 6000  # Max chars of context to send to hallucination checker


# ── Pydantic Schema for Structured Output ─────────────────────────────────────

class HallucinationGrade(BaseModel):
    """Structured output schema for the hallucination checker."""
    hallucination: bool = Field(
        description="True if the answer contains fabricated claims, False if fully grounded."
    )
    reasoning: str = Field(
        description="Clear explanation of what is or is not grounded."
    )
    unsupported_claims: list[str] = Field(
        default=[],
        description="List of specific claims that are not supported. Empty if none."
    )

# ── Prompt Templates ──────────────────────────────────────────────────────────

HALLUCINATION_SYSTEM_PROMPT = """You are an expert fact-checker for AI-generated answers \
about academic papers.

Your job is to determine whether a generated answer is fully grounded in the \
provided source documents, or whether it contains hallucinated (fabricated) claims.

A hallucination is any claim in the answer that:
- Is not supported by ANY of the provided source documents
- Contradicts information stated in the source documents
- Adds specific details (statistics, names, dates, numbers) not present in sources
- Makes causal claims not explicitly stated in the documents

An answer is NOT hallucinated if:
- All specific claims can be traced back to the source documents
- General statements are reasonable summaries of the documents
- The answer acknowledges uncertainty where documents are unclear"""


HALLUCINATION_HUMAN_PROMPT = """Original Question: {query}

Source Documents:
{context}

Generated Answer to Check:
{answer}

Is this answer fully grounded in the source documents?
Check every specific claim, number, name, and statement."""


# ── Context Builder ───────────────────────────────────────────────────────────

def build_check_context(documents: list[Document]) -> str:
    """
    Build a compact context string for the hallucination checker.
    Directly uses the already-curated relevant_documents from state.
    """
    context_parts = []
    total_chars   = 0

    for i, doc in enumerate(documents, start=1):
        # Truncate individual doc to keep total context manageable
        text = doc.get("text", "")[:800]
        title = doc.get("title", "Unknown")

        part = f"[Doc {i}] {title}\n{text}"
        part_chars = len(part)

        # Stop adding docs if we'd exceed the context limit
        if total_chars + part_chars > MAX_CONTEXT_LEN:
            logger.info(
                f"[HALLUCINATION] Context limit reached at doc {i} — truncating context"
            )
            break

        context_parts.append(part)
        total_chars += part_chars

    return "\n\n".join(context_parts)


# ── Core Check Logic ──────────────────────────────────────────────────────────

def check_hallucination(
    query: str,
    answer: str,
    context: str,
    llm,
) -> dict:
    """
    Ask the LLM to check whether the answer is grounded in context using Pydantic.
    """
    answer_to_check = answer[:3000] if len(answer) > 3000 else answer

    messages = [
        SystemMessage(content=HALLUCINATION_SYSTEM_PROMPT),
        HumanMessage(
            content=HALLUCINATION_HUMAN_PROMPT.format(
                query=query,
                context=context,
                answer=answer_to_check,
            )
        ),
    ]

    structured_llm = llm.with_structured_output(HallucinationGrade)

    try:
        result = structured_llm.invoke(messages)

        if result.hallucination:
            logger.warning(
                f"[HALLUCINATION] DETECTED — {len(result.unsupported_claims)} unsupported claims"
            )
            for claim in result.unsupported_claims:
                logger.warning(f"  Unsupported: {claim[:100]}")
        else:
            logger.info("[HALLUCINATION] CLEAR — answer is grounded in sources")

        return {
            "hallucination": result.hallucination,
            "reasoning": result.reasoning,
            "unsupported_claims": result.unsupported_claims,
        }

    except Exception as e:
        logger.error(f"[HALLUCINATION] Check failed with error: {e}", exc_info=True)
        return {
            "hallucination": True,  # Fail-closed: assume hallucinated if check fails
            "reasoning": "Hallucination check failed due to an internal error. Marking as potentially hallucinated for safety.",
            "unsupported_claims": ["Unable to verify — hallucination check encountered an error"],
        }


# ── Answer Annotation ─────────────────────────────────────────────────────────

def annotate_answer(answer: str, check_result: dict) -> str:
    """Optionally annotate the answer with a hallucination warning."""
    if not check_result["hallucination"]:
        return answer

    unsupported = check_result["unsupported_claims"]
    if not unsupported:
        return answer

    warning = (
        "\n\n⚠️ **Reliability Warning**: This answer may contain claims "
        "not fully supported by the retrieved documents. "
        "Please verify the following before relying on this answer:\n"
    )
    for claim in unsupported[:3]:  # Show max 3 unsupported claims
        warning += f"  • {claim}\n"

    return answer + warning


# ── Node Function ─────────────────────────────────────────────────────────────

def hallucination_node(state: CRAGState) -> dict:
    """LangGraph node: check generated answer for hallucinations."""
    query      = state["query"]
    answer     = state.get("generation", "")
    
    # CRITICAL FIX: Read from relevant_documents to match Generator's exact context
    documents  = state.get("relevant_documents", [])

    logger.info(f"[HALLUCINATION] Checking answer for query: '{query[:60]}'")

    if not answer.strip():
        logger.warning("[HALLUCINATION] No answer in state — skipping check")
        return {
            "hallucination": False,
            "hallucination_reasoning": "No answer to check",
        }

    if not documents:
        logger.warning(
            "[HALLUCINATION] No documents in state — "
            "cannot verify answer, marking as potentially hallucinated"
        )
        return {
            "hallucination": True,
            "hallucination_reasoning": "No source documents available to verify answer against",
        }

    llm = get_llm()
    context = build_check_context(documents)
    check_result = check_hallucination(query, answer, context, llm)
    annotated_answer = annotate_answer(answer, check_result)

    return {
        "hallucination": check_result["hallucination"],
        "hallucination_reasoning": check_result["reasoning"],
        "generation": annotated_answer,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_state: CRAGState = {
        "query": "How does Corrective RAG decide when to use web search?",
        "rewritten_query": "",
        "documents": [],
        "relevant_documents": [
            {
                "chunk_id": "2401.15884_chunk_0",
                "arxiv_id": "2401.15884",
                "text": (
                    "Corrective Retrieval Augmented Generation (CRAG) introduces "
                    "a lightweight retrieval evaluator to assess the quality of "
                    "retrieved documents. When documents are graded as incorrect, "
                    "CRAG triggers a web search via external APIs."
                ),
                "score": 0.93,
                "title": "Corrective Retrieval Augmented Generation",
                "authors": ["Shi-Qi Yan", "Jia-Chen Gu"],
                "published": "2024-01-29",
                "url": "http://arxiv.org/abs/2401.15884",
                "abstract": "CRAG paper abstract",
                "grade": "relevant",
                "source": "faiss",
            }
        ],
        "grade": "relevant",
        "document_grades": [],
        "generation": "CRAG uses a retrieval evaluator to assess document quality. When documents are irrelevant, it triggers web search.",
        "hallucination": False,
        "hallucination_reasoning": "",
        "retry_count": 0,
        "web_search_used": False,
        "source": "faiss",
        "error": "",
    }

    print("\nRunning hallucination node test...")
    result = hallucination_node(test_state)
    print(f"Hallucination: {result['hallucination']}")
    print(f"Reasoning: {result['hallucination_reasoning']}")