"""
Streamlit front end for the CRAG Academic Analyzer.

Talks to the FastAPI backend (``CRAG_API_URL``, default http://localhost:8000):
``/ready`` for status, ``/papers`` for the paper filter and ``/chat/stream``
for answers. Everything shown is data returned by the backend — sources,
pages, grades, warnings and timings — never model reasoning text.

Run::

    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, Iterator

import requests
import streamlit as st

API_URL = os.getenv("CRAG_API_URL", "http://localhost:8000").rstrip("/")
STREAM_TIMEOUT = (5, 200)  # (connect, read) seconds; the server sends keepalives every 15 s

GRADE_LABEL = {"relevant": "relevant", "ambiguous": "partly relevant", "ungraded": "not checked", "irrelevant": "irrelevant"}


# ── Pure helpers (unit-tested) ────────────────────────────────────────────────


def parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield JSON payloads from ``data:`` lines; ignore comments/keepalives and bad JSON."""
    for line in lines:
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        try:
            yield json.loads(line[len("data:"):].strip())
        except json.JSONDecodeError:
            continue


def md_escape(text: str) -> str:
    """Escape Markdown control characters in untrusted metadata (titles, sections)."""
    return re.sub(r"([\\`*_{}\[\]()#+\-!|>~<])", r"\\\1", text or "")


def safe_url(url: str) -> str | None:
    return url if isinstance(url, str) and re.match(r"^https?://[^\s<>\"']+$", url) else None


def citation_line(c: dict[str, Any]) -> str:
    badge = "🌐 Web" if c.get("source") == "web" else "📄 Paper"
    title = md_escape(c.get("title") or "Untitled")
    url = safe_url(c.get("url", ""))
    title_md = f"[{title}]({url})" if url else title
    where = " · ".join(md_escape(x) for x in (c.get("section", ""), c.get("pages", "")) if x)
    ident = f"arXiv:{md_escape(c['arxiv_id'])}" if c.get("arxiv_id") else ""
    details = " · ".join(x for x in (ident, where) if x)
    return f"**[{c.get('n')}]** {badge} — {title_md}" + (f"  \n:gray[{details}]" if details else "")


def grounding_summary(result: dict[str, Any]) -> tuple[str, str]:
    """(level, message) for the grounding check: level ∈ success | warning | info."""
    g = result.get("grounding") or {}
    if not result.get("evidence"):
        return "info", "No evidence was found, so no answer was generated."
    if g.get("verifier_failed"):
        return "warning", "The grounding check could not run — the answer is unverified."
    if not g.get("checked"):
        return "info", "Grounding check disabled."
    if g.get("unsupported_claims") or g.get("unsupported_numbers"):
        return "warning", "Some statements may not be supported by the cited sources (details below the answer)."
    return "success", "Grounding check found no unsupported claims (this does not prove the answer correct)."


# ── Backend calls ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=15, show_spinner=False)
def fetch_ready() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_URL}/ready", timeout=2)
        return r.json()
    except (requests.RequestException, ValueError):
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_papers() -> list[dict[str, Any]]:
    try:
        r = requests.get(f"{API_URL}/papers", timeout=5)
        r.raise_for_status()
        return r.json().get("papers", [])
    except (requests.RequestException, ValueError):
        return []


def http_error_message(response: requests.Response) -> str:
    if response.status_code == 422:
        return "The question was rejected: it is empty, too long, or the paper filter is invalid."
    if response.status_code == 429:
        return "Rate limit reached. Please wait a minute and try again."
    if response.status_code == 503:
        return "The server is busy or not ready. Please retry in a few seconds."
    return f"The server returned an error ({response.status_code}). Please try again."


# ── Rendering ─────────────────────────────────────────────────────────────────


def render_result(result: dict[str, Any]) -> None:
    st.markdown(result.get("answer_text") or result.get("answer", ""))

    for warning in result.get("warnings", []):
        st.warning(warning, icon="⚠️")
    level, message = grounding_summary(result)
    getattr(st, level)(message)

    grounding = result.get("grounding") or {}
    if grounding.get("unsupported_claims") or grounding.get("unsupported_numbers"):
        with st.expander("Statements to verify"):
            for claim in grounding.get("unsupported_claims", []):
                st.markdown(f"- {md_escape(claim)}")
            if grounding.get("unsupported_numbers"):
                st.markdown("Numbers not found in any source: " + ", ".join(md_escape(n) for n in grounding["unsupported_numbers"]))

    citations = result.get("citations", [])
    if citations:
        st.markdown("**Sources**")
        for c in citations:
            st.markdown(citation_line(c))

    evidence = result.get("evidence", [])
    if evidence:
        with st.expander(f"Evidence passages ({len(evidence)})"):
            for e in evidence:
                cited = "cited" if e.get("cited") else "not cited"
                score = f"{e['score']:.2f}" if isinstance(e.get("score"), (int, float)) else "–"
                st.markdown(
                    citation_line(e)
                    + f"  \n:gray[{GRADE_LABEL.get(e.get('grade', ''), e.get('grade', ''))} · similarity {score} · {cited}]",
                )
                st.text(e.get("text", ""))

    with st.expander("How this answer was produced"):
        cols = st.columns(4)
        cols[0].metric("Retrieval decision", result.get("grade") or "–")
        cols[1].metric("Query rewrites", result.get("retry_count", 0))
        cols[2].metric("Web search", "used" if result.get("web_search_used") else "not used")
        cols[3].metric("LLM calls", result.get("llm_calls", 0))
        if result.get("rewritten_query"):
            st.caption(f"Last rewritten query: {result['rewritten_query']}")
        trace = result.get("trace", [])
        if trace:
            st.dataframe(
                [{"step": t.get("node"), "ms": t.get("ms"), "detail": t.get("detail")} for t in trace],
                hide_index=True,
                use_container_width=True,
            )


def render_sidebar() -> tuple[list[str], bool]:
    with st.sidebar:
        st.header("Research settings")
        ready = fetch_ready()
        if ready is None:
            st.error("Backend offline — start it with `uvicorn src.api.server:app`.")
        elif ready.get("ready"):
            idx = ready.get("index", {})
            st.success(f"Ready · {idx.get('papers', 0)} papers · {idx.get('vectors', 0):,} passages")
        else:
            reasons = []
            if not ready.get("index", {}).get("ready"):
                reasons.append("index not built")
            if not ready.get("llm_configured"):
                reasons.append("GEMINI_API_KEY missing")
            st.warning("Backend not ready: " + ", ".join(reasons or ["unknown"]))

        papers = fetch_papers()
        labels = {p["arxiv_id"]: f"{p['title'][:70]} ({p['arxiv_id']})" for p in papers}
        selected = st.multiselect(
            "Limit to papers (optional)",
            options=list(labels),
            format_func=lambda pid: labels.get(pid, pid),
            help="Leave empty to search all indexed papers.",
            disabled=not papers,
        )
        web_ok = bool(ready and ready.get("web_search_configured"))
        allow_web = st.toggle(
            "Allow web search fallback",
            value=web_ok,
            disabled=not web_ok,
            help="Used only when the papers do not answer the question. Web sources are labelled 🌐."
            + ("" if web_ok else " (TAVILY_API_KEY not configured on the server)"),
        )
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("Each question is answered independently; earlier turns are not used as context.")
    return selected, allow_web


def ask(question: str, paper_ids: list[str], allow_web: bool) -> dict[str, Any] | None:
    """Stream one answer from the backend, showing step status. Returns the result payload."""
    status = st.status("Working on it…", expanded=False)
    payload = {"query": question, "paper_ids": paper_ids, "allow_web_search": allow_web}
    try:
        with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=STREAM_TIMEOUT) as response:
            if response.status_code != 200:
                status.update(label="Request failed", state="error")
                st.error(http_error_message(response))
                return None
            for event in parse_sse(response.iter_lines(decode_unicode=True)):
                kind = event.get("event")
                if kind == "node_update":
                    status.write(f"{event.get('status', event.get('node'))}" + (f" — {event['detail']}" if event.get("detail") else ""))
                    status.update(label=event.get("status", "Working…"))
                elif kind == "result":
                    status.update(label="Done", state="complete")
                    return event
                elif kind == "error":
                    status.update(label="Failed", state="error")
                    st.error(event.get("message") or "The backend reported an error.")
                    return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to the backend at {API_URL}.")
    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again.")
    except requests.RequestException:
        st.error("Unexpected network error. Please try again.")
    status.update(label="No answer", state="error")
    return None


def main() -> None:
    st.set_page_config(page_title="CRAG Academic Analyzer", page_icon="📚", layout="wide")
    st.title("📚 CRAG Academic Analyzer")
    st.caption(
        "Ask questions about the indexed arXiv papers. Answers cite passages with paper, section and page; "
        "if the papers do not contain the answer the assistant says so."
    )
    paper_ids, allow_web = render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("result"):
                render_result(message["result"])
            else:
                st.markdown(md_escape(message["content"]) if message["role"] == "user" else message["content"])

    if question := st.chat_input("Ask a question about the papers…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md_escape(question))
        with st.chat_message("assistant"):
            result = ask(question, paper_ids, allow_web)
            if result is not None:
                render_result(result)
                st.session_state.messages.append({"role": "assistant", "content": result.get("answer", ""), "result": result})


if __name__ == "__main__":  # Streamlit runs the script as __main__
    main()
