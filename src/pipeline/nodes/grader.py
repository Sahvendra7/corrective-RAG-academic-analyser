"""
nodes/grader.py
----------------
The second node in the CRAG pipeline.

Receives retrieved documents from state and grades each one
for relevance to the original query using an LLM (Gemini 1.5 Flash).

Grades each document as:
    - "relevant"   : Document clearly answers or supports the query
    - "ambiguous"  : Document is loosely related but not directly useful
    - "irrelevant" : Document has no meaningful relation to the query

Then makes an OVERALL grade decision for the whole set:
    - "relevant"   : Enough relevant docs to generate a good answer
    - "ambiguous"  : Some relevant docs but query should be rewritten
    - "irrelevant" : No useful docs found — trigger web search fallback

This overall grade is what graph.py uses to decide the next node.
"""

import logging
import sys
from pathlib import Path
import time

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.pipeline.state import (
    CRAGState,
    Document,
    GRADE_RELEVANT,
    GRADE_AMBIGUOUS,
    GRADE_IRRELEVANT,
)
from src.pipeline.llm_factory import get_llm

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Pydantic Schema for Structured Output ─────────────────────────────────────

class DocumentGrade(BaseModel):
    """
    Structured output schema for a single document grade.
    LangChain uses this to force the LLM to return valid JSON.
    """
    grade: str = Field(
        description="Relevance grade: 'relevant', 'ambiguous', or 'irrelevant'"
    )
    reasoning: str = Field(
        description="One sentence explanation of why this grade was assigned"
    )

# ── Prompt Templates ──────────────────────────────────────────────────────────

GRADER_SYSTEM_PROMPT = """You are an expert document relevance grader for a research paper Q&A system.

Your job is to assess whether a retrieved document chunk is relevant to the user's question.

Grade each document as exactly one of:
- "relevant"   : The document directly addresses, answers, or strongly supports the question
- "ambiguous"  : The document is related to the topic but does not directly answer the question
- "irrelevant" : The document has no meaningful connection to the question
You MUST respond with ONLY a valid JSON object in this exact format:
{"grade": "relevant", "reasoning": "brief reason"}

Be strict. A document about a related topic but not the specific question is "ambiguous", not "relevant"."""

GRADER_HUMAN_PROMPT = """User Question: {query}

Document to grade:
---
{text}
---

Grade this document's relevance to the question."""


# ── Grade Single Document ─────────────────────────────────────────────────────

def grade_document(query: str, doc: Document, llm) -> dict:
    """
    Grade a single document chunk for relevance to the query using Pydantic.
    """
    chunk_id = doc["chunk_id"]

    messages = [
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(
            content=GRADER_HUMAN_PROMPT.format(
                query=query,
                text=doc["text"][:1500],  # Truncate to avoid token limits
            )
        ),
    ]

    # Bind the Pydantic schema to the LLM
    structured_llm = llm.with_structured_output(DocumentGrade)

    try:
        # LLM returns the Pydantic object directly
        result = structured_llm.invoke(messages)
        
        grade = result.grade.lower().strip()

        # Validate grade value
        if grade not in [GRADE_RELEVANT, GRADE_AMBIGUOUS, GRADE_IRRELEVANT]:
            logger.warning(f"[GRADER] Invalid grade '{grade}' for {chunk_id} — defaulting to ambiguous")
            grade = GRADE_AMBIGUOUS

        logger.info(f"[GRADER] {chunk_id} - {grade.upper()} | {result.reasoning[:80]}")

        return {
            "chunk_id"  : chunk_id,
            "grade"     : grade,
            "reasoning" : result.reasoning,
        }

    except Exception as e:
        logger.error(f"[GRADER] Unexpected error grading {chunk_id}: {e}")
        return {
            "chunk_id"  : chunk_id,
            "grade"     : GRADE_AMBIGUOUS,
            "reasoning" : f"Grading error — defaulting to ambiguous: {str(e)}",
        }


# ── Overall Grade Decision ────────────────────────────────────────────────────

def decide_overall_grade(document_grades: list[dict]) -> str:
    """
    Aggregate individual document grades into one overall decision.
    """
    if not document_grades:
        logger.warning("[GRADER] No document grades to aggregate — defaulting to irrelevant")
        return GRADE_IRRELEVANT

    grades = [g["grade"] for g in document_grades]

    relevant_count   = grades.count(GRADE_RELEVANT)
    ambiguous_count  = grades.count(GRADE_AMBIGUOUS)
    irrelevant_count = grades.count(GRADE_IRRELEVANT)

    logger.info(
        f"[GRADER] Grade breakdown — "
        f"relevant: {relevant_count}, "
        f"ambiguous: {ambiguous_count}, "
        f"irrelevant: {irrelevant_count}"
    )

    if relevant_count >= 1:
        overall = GRADE_RELEVANT
    elif irrelevant_count == len(grades):
        overall = GRADE_IRRELEVANT
    else:
        overall = GRADE_AMBIGUOUS

    logger.info(f"[GRADER] Overall grade: {overall.upper()}")
    return overall


# ── Node Function ─────────────────────────────────────────────────────────────

def grader_node(state: CRAGState) -> dict:
    """
    LangGraph node: grade retrieved documents for relevance.
    """
    query     = state["query"]
    documents = state.get("documents", [])

    logger.info(f"[GRADER] Grading {len(documents)} documents for query: '{query[:60]}'")

    if not documents:
        logger.warning("[GRADER] No documents in state — marking as irrelevant")
        return {
            "document_grades"    : [],
            "relevant_documents" : [],
            "grade"              : GRADE_IRRELEVANT,
        }

    llm = get_llm()

    # Grade each document
    document_grades = []
    relevant_documents = []
    
    for doc in documents:
        grade_result = grade_document(query, doc, llm)
        document_grades.append(grade_result)

        # Option A Filter Logic: Only keep documents that passed the grade.
        # We include both "relevant" and "ambiguous" here as potential context, 
        # dropping the purely "irrelevant" noise.
        if grade_result["grade"] in [GRADE_RELEVANT, GRADE_AMBIGUOUS]:
            filtered_doc = dict(doc)
            filtered_doc["grade"] = grade_result["grade"]
            relevant_documents.append(filtered_doc)

    overall_grade = decide_overall_grade(document_grades)

    # Return the filtered list to the new state key, safely overwriting old data
    return {
        "document_grades"    : document_grades,
        "relevant_documents" : relevant_documents, 
        "grade"              : overall_grade,
    }


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_state: CRAGState = {
        "query": "How does corrective RAG handle irrelevant documents?",
        "rewritten_query": "",
        "documents": [
            {
                "chunk_id"  : "2401.15884_chunk_0",
                "arxiv_id"  : "2401.15884",
                "text"      : "Corrective Retrieval Augmented Generation (CRAG) proposes a lightweight retrieval evaluator to assess the overall quality of retrieved documents for a query. When documents are deemed incorrect or ambiguous, the system triggers web searches as a corrective action to supplement the retrieved documents.",
                "score"     : 0.91,
                "title"     : "CRAG",
                "authors"   : ["Author"],
                "published" : "2024",
                "url"       : "url",
                "abstract"  : "abstract",
                "grade"     : "",
                "source"    : "faiss",
            },
            {
                "chunk_id"  : "BAD_CHUNK_1",
                "arxiv_id"  : "0000.00000",
                "text"      : "Apples are a great source of fiber and vitamin C.",
                "score"     : 0.21,
                "title"     : "Nutrition",
                "authors"   : ["Author"],
                "published" : "2024",
                "url"       : "url",
                "abstract"  : "abstract",
                "grade"     : "",
                "source"    : "faiss",
            }
        ],
        "relevant_documents"     : [],
        "grade"                  : "",
        "document_grades"        : [],
        "generation"             : "",
        "hallucination"          : False,
        "hallucination_reasoning": "",
        "retry_count"            : 0,
        "web_search_used"        : False,
        "source"                 : "faiss",
        "error"                  : "",
    }

    print("\nRunning grader node test...")
    result = grader_node(test_state)

    print(f"\nOverall grade: {result['grade'].upper()}")
    print(f"\nOriginal Documents Count: {len(test_state['documents'])}")
    print(f"Filtered Relevant Documents Count: {len(result['relevant_documents'])}")
    
    print(f"\nPer-document grades:")
    for g in result["document_grades"]:
        print(f"  {g['chunk_id']} - {g['grade'].upper()}")
        print(f"  Reasoning: {g['reasoning']}")
        print()