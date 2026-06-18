"""
nodes/rewriter.py
------------------
The query rewriting node in the CRAG pipeline.

Triggered when the grader decides retrieval was AMBIGUOUS —
meaning some documents were retrieved but none were clearly
relevant enough to generate a good answer.

Takes the original query and rewrites it to be more specific,
precise, and likely to retrieve better documents on the next
FAISS search attempt.

Uses HyDE-style rewriting (Hypothetical Document Embeddings):
    1. First generates a hypothetical ideal answer
    2. Uses that hypothetical answer to craft a better query
    This works because embedding a hypothetical answer is often
    closer in vector space to real relevant documents than
    embedding the original short question.

Flow:
    CRAGState.query              (original question)
    CRAGState.document_grades    (grades + reasoning = context for rewriting)
    CRAGState.retry_count        (incremented here to track loop depth)
        ↓
    LLM rewrites query
        ↓
    CRAGState.rewritten_query    (improved query for next retrieval)
    CRAGState.retry_count        (incremented by 1)
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage


from src.pipeline.state import CRAGState
from src.pipeline.llm_factory import get_creative_llm

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Prompt Templates ──────────────────────────────────────────────────────────

REWRITER_SYSTEM_PROMPT = """You are an expert at rewriting research questions to improve \
document retrieval from a scientific paper database.

Your task:
Given an original question and feedback about why the initial retrieval failed,
rewrite the question to be more specific, precise, and semantically rich.

Rewriting strategies to apply:
1. Add technical terminology that would appear in relevant papers
2. Expand abbreviations (e.g. "RAG" → "Retrieval Augmented Generation")
3. Include related concepts that papers on this topic would discuss
4. Make implicit assumptions explicit
5. Use the vocabulary a researcher would use in a paper title or abstract

Rules:
- Output ONLY the rewritten query — no explanation, no preamble, no quotes
- Keep it as a question or search phrase, not a full sentence paragraph
- Do not make it longer than 2-3 sentences
- Do not change the core intent of the original question"""


REWRITER_HUMAN_PROMPT = """Original question: {query}

Retrieval feedback (why the initial retrieval was ambiguous):
{feedback}

Hypothetical ideal answer (use this to guide your rewriting):
{hypothetical_answer}

Rewrite the question to retrieve better documents:"""


HYDE_SYSTEM_PROMPT = """You are a research scientist. Given a question, write a short \
hypothetical abstract or passage (2-3 sentences) that would appear in a paper that \
perfectly answers this question.

This hypothetical passage will be used to find real papers via semantic search.
Write it in the style of an academic paper abstract — use technical terminology.
Output ONLY the hypothetical passage, nothing else."""


HYDE_HUMAN_PROMPT = """Question: {query}

Write a hypothetical paper passage that would perfectly answer this question:"""


# ── HyDE: Generate Hypothetical Answer ───────────────────────────────────────

def generate_hypothetical_answer(query: str, llm) -> str:
    """
    Generate a hypothetical ideal answer using HyDE technique.

    Instead of embedding the short question, we embed a hypothetical
    passage that would answer it. This lands closer in vector space
    to real relevant documents.

    Args:
        query: The original user question
        llm: LLM instance

    Returns:
        Hypothetical passage string
    """
    messages = [
        SystemMessage(content=HYDE_SYSTEM_PROMPT),
        HumanMessage(content=HYDE_HUMAN_PROMPT.format(query=query)),
    ]

    try:
        response = llm.invoke(messages)
        raw_content = response.content
        
        # Safely extract text whether response is a list or a string
        if isinstance(raw_content, list):
            content_str = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw_content
            )
        else:
            content_str = str(raw_content)
            
        hypothetical = content_str.strip()
        logger.info(f"[REWRITER] HyDE passage: {hypothetical[:100]}...")
        return hypothetical

    except Exception as e:
        logger.error(f"[REWRITER] HyDE generation failed: {e}")
        return ""


# ── Build Feedback String ─────────────────────────────────────────────────────

def build_feedback_string(document_grades: list[dict]) -> str:
    """
    Summarise grading feedback to inform the rewriter.

    Extracts reasoning from ambiguous/irrelevant document grades
    so the rewriter understands WHY the initial retrieval failed.

    Args:
        document_grades: List of per-doc grade dicts from grader node

    Returns:
        Human-readable feedback string
    """
    if not document_grades:
        return "No documents were retrieved — the query may be too vague."

    feedback_lines = []
    for g in document_grades:
        if g["grade"] in ["ambiguous", "irrelevant"]:
            feedback_lines.append(
                f"- Document '{g['chunk_id']}' was {g['grade']}: {g['reasoning']}"
            )

    if not feedback_lines:
        # All docs were relevant — shouldn't be calling rewriter in this case
        return "Retrieved documents were relevant but insufficient."

    return "\n".join(feedback_lines)


# ── Rewrite Query ─────────────────────────────────────────────────────────────

def rewrite_query(
    query: str,
    feedback: str,
    hypothetical_answer: str,
    llm,
) -> str:
    """
    Rewrite the query using LLM to improve retrieval on next attempt.

    Args:
        query: Original user question
        feedback: Summary of why initial retrieval failed
        hypothetical_answer: HyDE-generated hypothetical passage
        llm: LLM instance

    Returns:
        Rewritten query string
    """
    messages = [
        SystemMessage(content=REWRITER_SYSTEM_PROMPT),
        HumanMessage(
            content=REWRITER_HUMAN_PROMPT.format(
                query=query,
                feedback=feedback,
                hypothetical_answer=hypothetical_answer,
            )
        ),
    ]

    try:
        response = llm.invoke(messages)
        raw_content = response.content
        
        # Safely extract text whether response is a list or a string
        if isinstance(raw_content, list):
            content_str = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw_content
            )
        else:
            content_str = str(raw_content)
            
        rewritten = content_str.strip()

        # Sanity check — don't use rewrite if it's empty or identical
        if not rewritten or rewritten.lower() == query.lower():
            logger.warning(
                "[REWRITER] Rewritten query same as original or empty — "
                "using original query"
            )
            return query

        logger.info(f"[REWRITER] Original : '{query}'")
        logger.info(f"[REWRITER] Rewritten: '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.error(f"[REWRITER] Query rewriting failed: {e}")
        return query  # Fall back to original query


# ── Node Function ─────────────────────────────────────────────────────────────

def rewriter_node(state: CRAGState) -> dict:
    """
    LangGraph node: rewrite the query to improve retrieval.

    Called when grader returns overall grade = "ambiguous".
    Increments retry_count to track how deep we are in the loop.
    After MAX_RETRIES the graph will route to web search instead.

    Args:
        state: Current CRAGState

    Returns:
        Dict with keys to update in state:
            - rewritten_query: improved query for next retrieval attempt
            - retry_count: incremented by 1
    """
    # Use the most recent query — if we've already rewritten, build on that
    rewritten = state.get("rewritten_query", "").strip()
    original_query = state["query"]
    query = rewritten if rewritten else original_query
    document_grades = state.get("document_grades", [])
    retry_count     = state.get("retry_count", 0)

    logger.info(
        f"[REWRITER] Rewriting query (attempt {retry_count + 1}) — "
        f"'{query[:60]}'"
    )

    llm = get_creative_llm()

    # Step 1: Build feedback string from grading results
    feedback = build_feedback_string(document_grades)
    logger.info(f"[REWRITER] Feedback:\n{feedback}")

    # Step 2: Generate hypothetical ideal answer (HyDE)
    hypothetical_answer = generate_hypothetical_answer(query, llm)

    # Step 3: Rewrite the query using feedback + hypothetical answer
    rewritten = rewrite_query(query, feedback, hypothetical_answer, llm)

    # Step 4: Increment retry count
    new_retry_count = retry_count + 1
    logger.info(
        f"[REWRITER] Retry count: {retry_count} → {new_retry_count}"
    )

    return {
        "rewritten_query" : rewritten,
        "retry_count"     : new_retry_count,
    }


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_state: CRAGState = {
        "query"           : "How does RAG work?",
        "rewritten_query" : "",
        "documents"       : [],
        "grade"           : "ambiguous",
        "document_grades" : [
            {
                "chunk_id"  : "2401.15884_chunk_0",
                "grade"     : "ambiguous",
                "reasoning" : (
                    "Document discusses RAG in general but does not explain "
                    "the specific retrieval mechanism or architecture details."
                ),
            },
            {
                "chunk_id"  : "2312.10003_chunk_5",
                "grade"     : "irrelevant",
                "reasoning" : (
                    "Document is about CNN image classification, "
                    "unrelated to retrieval augmented generation."
                ),
            },
        ],
        "generation"             : "",
        "hallucination"          : False,
        "hallucination_reasoning": "",
        "retry_count"            : 0,
        "web_search_used"        : False,
        "source"                 : "faiss",
        "error"                  : "",
    }

    print("\nRunning rewriter node test...")
    result = rewriter_node(test_state)

    print(f"\nOriginal query  : {test_state['query']}")
    print(f"Rewritten query : {result['rewritten_query']}")
    print(f"Retry count     : {result['retry_count']}")