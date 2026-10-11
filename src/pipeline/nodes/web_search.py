"""
Web-search fallback (Tavily), used when local retrieval is irrelevant, or
still ambiguous after the rewrite budget is spent.

* Skipped (with a visible warning) when disabled in settings, disabled for the
  request, or when ``TAVILY_API_KEY`` is missing.
* Results are **graded** with the same evaluator as local passages and
  irrelevant ones are dropped. v0.1 added every web result to the evidence
  without grading.
* Web results are a separate source class (``source="web"``); the generator
  labels them and the UI shows them apart from paper passages.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time

from src.config import get_settings
from src.pipeline.evidence import is_web
from src.pipeline.nodes.grader import grade_pending
from src.pipeline.state import SOURCE_WEB, CRAGState, Document, trace_entry

logger = logging.getLogger(__name__)

MAX_CONTENT_LEN = 1500
MAX_QUERY_LEN = 395  # Tavily rejects queries over 400 characters

WEB_DISABLED = "Web search fallback is disabled; the answer is limited to the indexed papers."
WEB_NO_KEY = "Web search fallback is unavailable (TAVILY_API_KEY not set)."
WEB_FAILED = "Web search failed; the answer is limited to the indexed papers."

_tavily = None
_tavily_lock = threading.Lock()


def get_tavily():
    global _tavily
    if _tavily is None:
        with _tavily_lock:
            if _tavily is None:
                from tavily import TavilyClient

                key = get_settings().tavily_key()
                if not key:
                    raise ValueError("TAVILY_API_KEY not set")
                _tavily = TavilyClient(api_key=key)
    return _tavily


def select_search_query(state: CRAGState) -> str:
    return (state.get("rewritten_query") or state["query"]).strip()


def tavily_result_to_document(result: dict, index: int) -> Document:
    url = result.get("url", "")
    content = " ".join(result.get("content", "").split())
    if len(content) > MAX_CONTENT_LEN:
        content = content[:MAX_CONTENT_LEN] + " …"
    return Document(
        chunk_id=f"web_{hashlib.sha1(url.encode()).hexdigest()[:12]}_{index}",
        arxiv_id="",
        text=content,
        score=float(result.get("score", 0.0)),
        title=result.get("title") or "Web result",
        authors=[],
        published=(result.get("published_date") or "")[:10],
        url=url,
        section="",
        heading="",
        page_start=None,
        page_end=None,
        grade="",
        grade_reason="",
        source=SOURCE_WEB,
    )


def search_web(query: str) -> list[Document]:
    """Run Tavily; raises on transport/API errors."""
    settings = get_settings()
    response = get_tavily().search(
        query=query[:MAX_QUERY_LEN],
        search_depth="advanced",
        max_results=settings.web_max_results,
        include_answer=False,
        include_raw_content=False,
        timeout=settings.web_timeout_s,
    )
    docs = [tavily_result_to_document(r, i) for i, r in enumerate(response.get("results", []))]
    return [d for d in docs if d["text"].strip()]


def web_search_node(state: CRAGState) -> dict:
    started = time.perf_counter()
    settings = get_settings()
    warnings = list(state.get("warnings", []))
    docs = list(state.get("documents", []))

    def skip(reason: str) -> dict:
        if reason not in warnings:
            warnings.append(reason)
        return {"warnings": warnings, "trace": trace_entry(state, "web_search", started, reason)}

    if not (settings.web_search_enabled and state.get("allow_web_search", True)):
        return skip(WEB_DISABLED)
    if not settings.tavily_key():
        return skip(WEB_NO_KEY)
    try:
        web_docs = search_web(select_search_query(state))
    except Exception as exc:
        logger.error("[WEB_SEARCH] %s", exc)
        return skip(WEB_FAILED)

    seen = {d.get("url") for d in docs if is_web(d)}
    docs += [d for d in web_docs if d["url"] not in seen]
    docs, records, _failed, calls = grade_pending(state, docs)
    kept = [d for d in docs if is_web(d) and d["grade"] != "irrelevant"]
    logger.info("[WEB_SEARCH] %d results, %d kept after grading", len(web_docs), len(kept))
    return {
        "documents": docs,
        "web_search_used": True,
        "document_grades": records,
        "warnings": warnings,
        "llm_calls": state.get("llm_calls", 0) + calls,
        "trace": trace_entry(state, "web_search", started, f"{len(web_docs)} results, {len(kept)} kept"),
    }
