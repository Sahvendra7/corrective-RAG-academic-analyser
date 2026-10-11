"""
Retriever node: dense search over the local FAISS index.

* Uses ``rewritten_query`` after a rewrite, the original question otherwise.
* Passes the chunk ids already in state as ``exclude_ids`` so a retry fetches
  *new* passages instead of re-appending duplicates (v0.1 appended duplicates,
  which were graded again and sent to the generator twice).
* Honours the paper filter chosen by the user.
* An unavailable index is reported as a warning, not hidden behind the web
  fallback.
"""

from __future__ import annotations

import logging
import threading
import time

from src.config import get_settings
from src.pipeline.state import SOURCE_FAISS, CRAGState, Document, trace_entry
from src.vectorstore.faiss_store import FAISSStore, IndexConsistencyError

logger = logging.getLogger(__name__)

INDEX_UNAVAILABLE = "Local paper index is not available; answers cannot use the indexed papers."

_store: FAISSStore | None = None
_store_lock = threading.Lock()


def get_store() -> FAISSStore:
    """Process-wide index, loaded on first use."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = FAISSStore(load_existing=True)
    return _store


def set_store(store: FAISSStore | None) -> None:
    """Replace the process-wide store (tests, or after rebuilding the index)."""
    global _store
    with _store_lock:
        _store = store


def to_document(result: dict) -> Document:
    return Document(
        chunk_id=result["chunk_id"],
        arxiv_id=result.get("arxiv_id", ""),
        text=result["text"],
        score=float(result.get("score", 0.0)),
        title=result.get("title", ""),
        authors=list(result.get("authors") or []),
        published=result.get("published", ""),
        url=result.get("url", ""),
        section=result.get("section", ""),
        heading=result.get("heading", ""),
        page_start=result.get("page_start"),
        page_end=result.get("page_end"),
        grade="",
        grade_reason="",
        source=SOURCE_FAISS,
    )


def retriever_node(state: CRAGState) -> dict:
    started = time.perf_counter()
    settings = get_settings()
    rewritten = state.get("rewritten_query", "").strip()
    query = rewritten if state.get("retry_count", 0) > 0 and rewritten else state["query"]
    existing = list(state.get("documents", []))
    warnings = list(state.get("warnings", []))

    try:
        store = get_store()
        if not store.is_ready:
            raise IndexConsistencyError("index not loaded")
        results = store.search(
            query,
            top_k=settings.top_k,
            paper_ids=state.get("paper_filter") or None,
            exclude_ids={d["chunk_id"] for d in existing},
        )
    except Exception as exc:  # missing/corrupt index, embedding failure
        logger.error("[RETRIEVER] %s", exc)
        if INDEX_UNAVAILABLE not in warnings:
            warnings.append(INDEX_UNAVAILABLE)
        results = []

    new_docs = [to_document(r) for r in results]
    elapsed = (time.perf_counter() - started) * 1000
    logger.info("[RETRIEVER] %d new passages for %r", len(new_docs), query[:80])
    return {
        "documents": existing + new_docs,
        "warnings": warnings,
        "retrieval_ms": state.get("retrieval_ms", 0.0) + elapsed,
        "trace": trace_entry(state, "retriever", started, f"{len(new_docs)} new passages"),
    }
