"""
nodes/retriever.py
-------------------
The first node in the CRAG pipeline.

Receives the user query (or rewritten query on retry) from state,
searches the FAISS vector store for the most relevant chunks,
and returns them as a list of Document objects back into state.

This node is called:
    1. At the start of every query with the original query
    2. Again after the rewriter node if retrieval was AMBIGUOUS
       (using the rewritten_query instead)

Flow:
    CRAGState.query / CRAGState.rewritten_query
        ↓
    FAISSStore.search()
        ↓
    CRAGState.documents  (list of Document dicts)
    CRAGState.source     ("faiss")
"""

import logging
import threading


from src.pipeline.state import (
    CRAGState,
    Document,
    SOURCE_FAISS,
    MAX_RETRIES,
)
from src.vectorstore.faiss_store import FAISSStore

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TOP_K = 2   # Number of chunks to retrieve per query

# ── FAISS Store Singleton ─────────────────────────────────────────────────────
# We load the FAISS store once at module level so it isn't
# reloaded on every node call — loading takes a few seconds.

_store: FAISSStore | None = None
_store_lock = threading.Lock()

def get_store() -> FAISSStore:
    """
    Return the singleton FAISSStore instance.
    Loads from disk on first call, reuses after that.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                logger.info("Loading FAISS store (first call)...")
                _store = FAISSStore(load_existing=True)
    return _store


# ── Node Function ─────────────────────────────────────────────────────────────

def retriever_node(state: CRAGState) -> dict:
    """
    LangGraph node: retrieve relevant chunks from FAISS.

    Decides which query to use:
        - On first attempt  → use state["query"]
        - On retry attempts → use state["rewritten_query"]

    Args:
        state: Current CRAGState

    Returns:
        Dict with keys to update in state:
            - documents: list of Document dicts
            - source: "faiss"
    """
    retry_count = state.get("retry_count", 0)

    # On retry, use the rewritten query if available
    rewritten = state.get("rewritten_query", "").strip()
    if retry_count > 0 and rewritten:
        query = rewritten
        logger.info(f"[RETRIEVER] Retry {retry_count}/{MAX_RETRIES} — using rewritten query")
    else:
        query = state["query"]
        logger.info(f"[RETRIEVER] First attempt — using original query")

    logger.info(f"[RETRIEVER] Query: '{query[:80]}'")

    # Guard against empty query
    if not query:
        logger.warning("[RETRIEVER] Empty query provided")
        return {
            "documents": state.get("documents", []),
            "source": SOURCE_FAISS,
            "error": "Empty query received by retriever",
        }

    try:
        store = get_store()

        # Search FAISS for top-k most similar chunks
        import time
        start_time = time.perf_counter()
        raw_results = store.search(query=query, top_k=TOP_K)
        retrieval_ms = (time.perf_counter() - start_time) * 1000

        if not raw_results:
            logger.warning("[RETRIEVER] No results returned from FAISS")
            return {
                "documents": state.get("documents", []),
                "source": SOURCE_FAISS,
                "retrieval_ms": state.get("retrieval_ms", 0.0) + retrieval_ms,
            }

        # Convert raw search results into Document dicts
        # Also tag each doc with its source and an empty grade
        # (grade will be filled in by the grader node)
        documents: list[Document] = []
        for result in raw_results:
            doc: Document = {
                "chunk_id"  : result["chunk_id"],
                "arxiv_id"  : result["arxiv_id"],
                "text"      : result["text"],
                "score"     : result["score"],
                "title"     : result.get("title", ""),
                "authors"   : result.get("authors", []),
                "published" : result.get("published", ""),
                "url"       : result.get("url", ""),
                "abstract"  : result.get("abstract", ""),
                "grade"     : "",        # Filled in by grader node
                "source"    : SOURCE_FAISS,
            }
            documents.append(doc)

        logger.info(f"[RETRIEVER] Found {len(documents)} context chunks")
        
        return {
            "documents": state.get("documents", []) + documents,
            "source"   : SOURCE_FAISS,
            "retrieval_ms": state.get("retrieval_ms", 0.0) + retrieval_ms,
        }

    except Exception as e:
        logger.error(f"[RETRIEVER] Error during FAISS search: {e}")
        return {
            "documents" : state.get("documents", []),
            "source"    : SOURCE_FAISS,
            "error"     : f"Retriever error: {str(e)}",
            "retrieval_ms": state.get("retrieval_ms", 0.0),
        }



# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate a state dict as LangGraph would pass it
    test_state: CRAGState = {
        "query"           : "How does corrective RAG handle irrelevant documents?",
        "rewritten_query" : "",
        "documents"       : [],
        "grade"           : "",
        "document_grades" : [],
        "generation"      : "",
        "hallucination"   : False,
        "hallucination_reasoning": "",
        "retry_count"     : 0,
        "web_search_used" : False,
        "source"          : "",
        "error"           : "",
    }

    print("\nRunning retriever node test...")
    result = retriever_node(test_state)

    print(f"\nReturned {len(result['documents'])} documents:")
    for i, doc in enumerate(result["documents"]):
        print(f"\n--- Doc {i+1} ---")
        print(f"  Score   : {doc['score']}")
        print(f"  Title   : {doc['title'][:60]}")
        print(f"  Chunk   : {doc['chunk_id']}")
        print(f"  Text    : {doc['text'][:150]}...")
