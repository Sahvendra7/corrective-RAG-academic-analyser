"""
nodes/generator.py
-------------------
The answer generation node in the CRAG pipeline.

Receives the final set of RELEVANT documents (from FAISS and/or web search)
and generates a grounded, cited answer to the user's query.

Key responsibilities:
    1. Build a structured context from the filtered relevant docs
    2. Generate an answer with inline citations using Gemini 1.5 Flash
    3. Format citations pointing back to source papers/URLs

This node always runs last in the main pipeline path,
right before the hallucination checker.
"""

import logging
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.pipeline.state import CRAGState, Document
from src.pipeline.llm_factory import get_generator_llm

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MAX_CONTEXT_DOCS = 5     # Max documents to include in context


# ── Prompt Templates ──────────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """You are an expert research assistant specialising in \
machine learning and AI. Your job is to answer questions about academic papers \
using ONLY the provided document context.

Rules:
1. Base your answer ENTIRELY on the provided context — do not use outside knowledge
2. Cite sources inline using [Doc N] format where N is the document number
3. If multiple documents support a claim, cite all of them e.g. [Doc 1][Doc 2]
4. If the context does not contain enough information to fully answer the question,
   say so clearly rather than guessing
5. Structure your answer clearly — use paragraphs, not bullet points
6. Be precise and technical — your audience are ML researchers
7. End with a "Sources" section listing all cited documents with their titles and URLs

Format:
<answer>
Your detailed answer here with inline [Doc N] citations...

Sources:
[Doc 1] Title — URL
[Doc 2] Title — URL
</answer>"""


GENERATOR_HUMAN_PROMPT = """Question: {query}

Context documents:
{context}

Answer the question using ONLY the context above. Include inline citations."""


# ── Context Builder ───────────────────────────────────────────────────────────

def build_context(documents: list[Document]) -> str:
    """
    Build a structured context string from documents for the LLM prompt.

    Each document is numbered [Doc N] so the LLM can cite them inline.
    """
    context_parts = []

    for i, doc in enumerate(documents, start=1):
        # Build source line depending on doc type
        if doc.get("arxiv_id"):
            authors = ", ".join(doc.get("authors", [])[:3])
            if len(doc.get("authors", [])) > 3:
                authors += " et al."
            source_line = (
                f"Paper: {doc.get('title', 'Unknown')}\n"
                f"Authors: {authors}\n"
                f"Published: {doc.get('published', 'unknown')}\n"
                f"URL: {doc.get('url', 'N/A')}"
            )
        else:
            # Web result
            source_line = (
                f"Web Source: {doc.get('title', 'Web Result')}\n"
                f"URL: {doc.get('url', 'N/A')}\n"
                f"Published: {doc.get('published', 'unknown')}"
            )

        # Truncate text to avoid token overflow
        text = doc.get("text", "")
        if len(text) > 1200:
            text = text[:1200] + "..."

        context_parts.append(
            f"[Doc {i}]\n"
            f"{source_line}\n"
            f"Content:\n{text}"
        )

    return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(context_parts) + "\n"


# ── Generation ────────────────────────────────────────────────────────────────

def generate_answer(query: str, context: str, llm) -> str:
    """
    Generate a cited answer using the LLM.
    """
    messages = [
        SystemMessage(content=GENERATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=GENERATOR_HUMAN_PROMPT.format(
                query=query,
                context=context,
            )
        ),
    ]

    try:
        # Safely invoke the LLM
        response = llm.invoke(messages)
        raw_content = response.content
        
        # Google Gemini safe-extraction: If it's a list of blocks, join them into a string
        if isinstance(raw_content, list):
            answer = "".join(block.get("text", "") for block in raw_content if isinstance(block, dict))
        else:
            answer = raw_content
            
        # Now it is guaranteed to be a string, so we can strip it
        answer = answer.strip()

        # Strip <answer> tags if LLM included them
        if answer.startswith("<answer>"):
            answer = answer[8:]
        if answer.endswith("</answer>"):
            answer = answer[:-9]

        answer = answer.strip()
        logger.info(f"[GENERATOR] Generated answer ({len(answer)} chars)")
        return answer

    except Exception as e:
        logger.error("[GENERATOR] Answer generation failed", exc_info=True)
        return (
            "I was unable to generate an answer due to an internal error. "
            "Please try again or rephrase your question."
        )


# ── Node Function ─────────────────────────────────────────────────────────────

def generator_node(state: CRAGState) -> dict:
    """
    LangGraph node: generate a cited answer from retrieved documents.
    """
    query  = state["query"]
    source = state.get("source", "faiss")
    
    # CRITICAL ARCHITECTURAL UPDATE: 
    # Directly use relevant_documents. The grader and web search nodes
    # already curated this list, so we don't need to re-filter anything.
    documents = state.get("relevant_documents", [])

    logger.info(
        f"[GENERATOR] Generating answer for: '{query[:60]}' "
        f"(source={source}, relevant docs={len(documents)})"
    )

    # Handle edge case — no documents at all
    if not documents:
        logger.warning("[GENERATOR] No relevant documents in state — generating fallback response")
        return {
            "generation": (
                "I could not find relevant information to answer your question. "
                "The local knowledge base and web search both returned no useful results. "
                "Please try rephrasing your question or check if the topic is covered "
                "in the indexed papers."
            )
        }

    llm = get_generator_llm()

    # Step 1: Cap to MAX_CONTEXT_DOCS
    useful_docs = documents[:MAX_CONTEXT_DOCS]

    # Step 2: Build context string
    context = build_context(useful_docs)
    logger.info(
        f"[GENERATOR] Context built — "
        f"{len(useful_docs)} docs, {len(context)} chars"
    )

    # Step 3: Generate answer
    answer = generate_answer(query, context, llm)

    # Step 4: Add metadata footer
    source_label = {
        "faiss" : "Local paper database",
        "web"   : "Web search (Tavily)",
        "mixed" : "Local paper database + Web search",
    }.get(source, source)

    footer = f"\n\n---\n*Answer generated from: {source_label}*"
    final_answer = answer + footer

    logger.info("[GENERATOR] Answer generation complete")

    return {
        "generation": final_answer,
    }


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_state: CRAGState = {
        "query": "How does Corrective RAG decide when to use web search?",
        "rewritten_query": "",
        "documents": [], 
        "relevant_documents": [ # The test now populates the correct filtered state key
            {
                "chunk_id"  : "2401.15884_chunk_0",
                "arxiv_id"  : "2401.15884",
                "text"      : (
                    "Corrective Retrieval Augmented Generation (CRAG) introduces "
                    "a lightweight retrieval evaluator to assess the quality of "
                    "retrieved documents. The evaluator assigns one of three grades: "
                    "correct, incorrect, or ambiguous. When documents are graded as "
                    "incorrect, CRAG triggers a web search via external APIs to "
                    "supplement the retrieved context."
                ),
                "score"     : 0.93,
                "title"     : "Corrective Retrieval Augmented Generation",
                "authors"   : ["Shi-Qi Yan", "Jia-Chen Gu", "Yun Leng", "Zhen Li"],
                "published" : "2024-01-29",
                "url"       : "http://arxiv.org/abs/2401.15884",
                "abstract"  : "CRAG paper abstract",
                "grade"     : "relevant",
                "source"    : "faiss",
            }
        ],
        "grade"                  : "relevant",
        "document_grades"        : [],
        "generation"             : "",
        "hallucination"          : False,
        "hallucination_reasoning": "",
        "retry_count"            : 0,
        "web_search_used"        : False,
        "source"                 : "faiss",
        "error"                  : "",
    }

    print("\nRunning generator node test...")
    result = generator_node(test_state)

    print("\n" + "="*60)
    print("GENERATED ANSWER:")
    print("="*60)
    print(result["generation"])