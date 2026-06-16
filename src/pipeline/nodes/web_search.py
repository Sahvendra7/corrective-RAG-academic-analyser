"""
nodes/web_search.py
--------------------
The web search fallback node in the CRAG pipeline.

Triggered when the grader decides ALL retrieved documents
are IRRELEVANT — meaning the local FAISS knowledge base
doesn't have useful information for this query.

Uses the Tavily Search API to fetch fresh, relevant web results
and converts them into Document objects compatible with the
rest of the pipeline (grader, generator, hallucination checker).

The retrieved web documents are APPENDED to state["documents"]
alongside any FAISS documents (operator.add in state.py handles this).

Flow:
    CRAGState.query              (original question)
    CRAGState.rewritten_query    (used if available — often better)
        ↓
    Tavily Search API
        ↓
    CRAGState.documents          (web docs appended to existing docs)
    CRAGState.web_search_used    (set to True)
    CRAGState.source             (updated to "web" or "mixed")
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.pipeline.state import (
    CRAGState,
    Document,
    SOURCE_FAISS,
    SOURCE_WEB,
    SOURCE_MIXED,
)

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MAX_WEB_RESULTS  = 3     # Tavily results to fetch — keep low to reduce noise
MAX_CONTENT_LEN  = 1500  # Truncate web content to this many chars per result


# ── Tavily Client Singleton ───────────────────────────────────────────────────

_tavily: TavilyClient | None = None

def get_tavily() -> TavilyClient:
    """Return singleton Tavily client."""
    global _tavily
    if _tavily is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY not found in environment. Check your .env file."
            )
        _tavily = TavilyClient(api_key=api_key)
        logger.info("[WEB_SEARCH] Tavily client initialised")
    return _tavily


# ── Query Selection ───────────────────────────────────────────────────────────

def select_search_query(state: CRAGState) -> str:
    """
    Decide which query to send to Tavily.

    Prefers rewritten_query if available since it's more
    specific and likely to return better web results.
    Falls back to original query.

    Args:
        state: Current CRAGState

    Returns:
        Query string to send to Tavily
    """
    rewritten = state.get("rewritten_query", "").strip()
    original  = state["query"]

    if rewritten and rewritten != original:
        logger.info(f"[WEB_SEARCH] Using rewritten query: '{rewritten[:80]}'")
        return rewritten

    logger.info(f"[WEB_SEARCH] Using original query: '{original[:80]}'")
    return original


# ── Convert Tavily Results to Documents ──────────────────────────────────────

def tavily_result_to_document(result: dict, index: int) -> Document:
    """
    Convert a single Tavily search result into a Document dict
    compatible with the rest of the CRAG pipeline.

    Tavily result keys:
        title, url, content, score, published_date (optional)

    Args:
        result: Single Tavily result dict
        index: Result index (used to generate chunk_id)

    Returns:
        Document dict
    """
    # Truncate content to avoid overwhelming the generator
    content = result.get("content", "").strip()
    if len(content) > MAX_CONTENT_LEN:
        content = content[:MAX_CONTENT_LEN] + "..."

    # Build a fake chunk_id for web results so they work
    # with the same pipeline logic as FAISS chunks
    url      = result.get("url", "")
    chunk_id = f"web_result_{index}_{url[-30:].replace('/', '_')}"

    # Parse published date if available
    published = result.get("published_date", "")
    if published:
        published = published[:10]  # Keep only YYYY-MM-DD
    else:
        published = "unknown"

    doc: Document = {
        "chunk_id"  : chunk_id,
        "arxiv_id"  : "",            # No arxiv ID for web results
        "text"      : content,
        "score"     : float(result.get("score", 0.0)),
        "title"     : result.get("title", "Web Result"),
        "authors"   : [],            # No authors for web results
        "published" : published,
        "url"       : url,
        "abstract"  : "",            # No abstract for web results
        "grade"     : "",            # Will be graded if needed
        "source"    : SOURCE_WEB,
    }

    return doc


# ── Core Web Search ───────────────────────────────────────────────────────────

def search_web(query: str) -> list[Document]:
    """
    Run a Tavily search and return results as Document dicts.

    Uses Tavily's "advanced" search for better quality results.
    Filters to include_answer=False since we want raw content,
    not Tavily's own answer generation.

    Args:
        query: Search query string

    Returns:
        List of Document dicts from web search results
    """
    client = get_tavily()

    try:
        logger.info(f"[WEB_SEARCH] Searching Tavily: '{query[:80]}'")

        response = client.search(
            query=query,
            search_depth="advanced",   # Higher quality than "basic"
            max_results=MAX_WEB_RESULTS,
            include_answer=False,      # We generate our own answer
            include_raw_content=False, # Processed content is cleaner
        )

        raw_results = response.get("results", [])
        logger.info(f"[WEB_SEARCH] Tavily returned {len(raw_results)} results")

        if not raw_results:
            logger.warning("[WEB_SEARCH] No results from Tavily")
            return []

        # Convert to Document dicts
        documents = []
        for i, result in enumerate(raw_results):
            doc = tavily_result_to_document(result, i)

            # Skip results with no content
            if not doc["text"].strip():
                logger.warning(f"[WEB_SEARCH] Skipping result {i} — empty content")
                continue

            documents.append(doc)
            logger.info(
                f"[WEB_SEARCH] Result {i+1}: "
                f"score={doc['score']:.3f} | "
                f"{doc['title'][:50]} | "
                f"{doc['url'][:60]}"
            )

        logger.info(
            f"[WEB_SEARCH] {len(documents)} usable web documents retrieved"
        )
        return documents

    except Exception as e:
        logger.error(f"[WEB_SEARCH] Tavily search failed: {e}")
        return []


# ── Determine Updated Source ──────────────────────────────────────────────────

def determine_source(existing_docs: list[Document], web_docs: list[Document]) -> str:
    """
    Determine the source label for state["source"] after web search.

    Args:
        existing_docs: Documents already in state (from FAISS)
        web_docs: New documents from web search

    Returns:
        "web"   — if only web docs exist
        "mixed" — if both FAISS and web docs exist
        "faiss" — if no web docs (shouldn't happen here but safe fallback)
    """
    has_faiss = any(d["source"] == SOURCE_FAISS for d in existing_docs)
    has_web   = len(web_docs) > 0

    if has_faiss and has_web:
        return SOURCE_MIXED
    elif has_web:
        return SOURCE_WEB
    else:
        return SOURCE_FAISS


# ── Node Function ─────────────────────────────────────────────────────────────

def web_search_node(state: CRAGState) -> dict:
    """
    LangGraph node: fetch web results as fallback when FAISS fails.

    Triggered when grader grades overall retrieval as "irrelevant".
    Searches Tavily and appends results to state["documents"].
    Because documents uses operator.add in state.py, these new
    docs are appended to existing docs automatically by LangGraph.

    Args:
        state: Current CRAGState

    Returns:
        Dict with keys to update in state:
            - documents: list of new web Document dicts (appended by LangGraph)
            - web_search_used: True
            - source: "web" or "mixed"
    """
    logger.info("[WEB_SEARCH] Triggering web search fallback...")

    # Select best query
    query = select_search_query(state)

    # Run Tavily search
    web_documents = search_web(query)

    if not web_documents:
        logger.warning(
            "[WEB_SEARCH] No web results found — "
            "pipeline will attempt generation with existing docs"
        )
        return {
            "documents"      : [],
            "web_search_used": True,
            "source"         : SOURCE_WEB,
            "error"          : "Web search returned no results",
        }

    # Determine source label
    existing_docs = state.get("documents", [])
    source        = determine_source(existing_docs, web_documents)

    logger.info(
        f"[WEB_SEARCH] Complete — "
        f"{len(web_documents)} web docs added | source={source}"
    )

    return {
            "documents": web_documents,          # Appends to the raw log (operator.add)
            "relevant_documents": web_documents, # Overwrites the empty list left by the grader
            "web_search_used": True,
            "source": source,
        }


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_state: CRAGState = {
        "query"                  : "What is Corrective RAG and how does it improve retrieval?",
        "rewritten_query"        : "Corrective Retrieval Augmented Generation CRAG self-correction mechanism",
        "documents"              : [],
        "grade"                  : "irrelevant",
        "document_grades"        : [],
        "generation"             : "",
        "hallucination"          : False,
        "hallucination_reasoning": "",
        "retry_count"            : 0,
        "web_search_used"        : False,
        "source"                 : "faiss",
        "error"                  : "",
    }

    print("\nRunning web search node test...")
    result = web_search_node(test_state)

    print(f"\nWeb search used : {result['web_search_used']}")
    print(f"Source          : {result['source']}")
    print(f"Documents found : {len(result['documents'])}")
    print()

    for i, doc in enumerate(result["documents"]):
        print(f"--- Web Result {i+1} ---")
        print(f"  Title   : {doc['title'][:60]}")
        print(f"  URL     : {doc['url'][:70]}")
        print(f"  Score   : {doc['score']:.3f}")
        print(f"  Content : {doc['text'][:150]}...")
        print()
