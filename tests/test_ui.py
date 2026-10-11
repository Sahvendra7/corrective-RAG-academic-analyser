"""Streamlit UI: pure helpers and an AppTest run against a faked backend."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "ui" / "app.py"


@pytest.fixture(scope="module")
def ui():
    spec = importlib.util.spec_from_file_location("crag_ui", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is guarded, so nothing renders
    return module


RESULT = {
    "event": "result",
    "answer_text": "CRAG uses a lightweight retrieval evaluator [1].",
    "citations": [
        {"n": 1, "source": "paper", "title": "Corrective [RAG]", "arxiv_id": "2401.15884", "url": "http://arxiv.org/abs/2401.15884", "section": "Method", "pages": "p. 3"},
    ],
    "evidence": [
        {"n": 1, "cited": True, "source": "paper", "title": "Corrective [RAG]", "arxiv_id": "2401.15884", "url": "javascript:alert(1)", "section": "Method", "pages": "p. 3", "grade": "relevant", "score": 0.61, "text": "The evaluator scores documents."},
    ],
    "grade": "relevant",
    "retry_count": 0,
    "web_search_used": False,
    "grounding": {"checked": True, "unsupported_claims": [], "unsupported_numbers": []},
    "warnings": ["Web search fallback is unavailable (TAVILY_API_KEY not set)."],
    "trace": [{"node": "retriever", "ms": 3.2, "detail": "5 new passages"}],
    "llm_calls": 3,
}


def test_parse_sse_skips_keepalives_and_garbage(ui):
    lines = [": keepalive", "", "data: {\"event\": \"node_update\"}", "data: {broken", "event: x", "data: {\"event\": \"done\"}"]
    assert [e["event"] for e in ui.parse_sse(lines)] == ["node_update", "done"]


def test_untrusted_metadata_is_escaped_and_urls_checked(ui):
    line = ui.citation_line(RESULT["citations"][0])
    assert "Corrective \\[RAG\\]" in line and "(http://arxiv.org/abs/2401.15884)" in line
    assert "p. 3" in line and "📄 Paper" in line
    assert ui.safe_url("javascript:alert(1)") is None
    assert "javascript" not in ui.citation_line(RESULT["evidence"][0])
    assert "🌐 Web" in ui.citation_line({"n": 2, "source": "web", "title": "Blog", "url": "https://x.org"})


def test_grounding_summary_levels(ui):
    assert ui.grounding_summary(RESULT)[0] == "success"
    assert ui.grounding_summary({**RESULT, "grounding": {"verifier_failed": True}})[0] == "warning"
    assert ui.grounding_summary({**RESULT, "grounding": {"checked": True, "unsupported_numbers": ["42"]}})[0] == "warning"
    assert ui.grounding_summary({**RESULT, "evidence": []})[0] == "info"


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    import streamlit as st

    st.cache_data.clear()  # cached /ready and /papers responses persist in-process
    yield


class _Resp:
    def __init__(self, status=200, body=None, lines=()):
        self.status_code, self._body, self._lines = status, body, lines

    def json(self):
        return self._body

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_app_renders_answer_sources_and_status(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import requests

    ready = {"ready": True, "index": {"ready": True, "papers": 2, "vectors": 5}, "llm_configured": True, "web_search_configured": False}
    papers = {"papers": [{"arxiv_id": "2401.15884", "title": "Corrective RAG"}]}
    monkeypatch.setattr(requests, "get", lambda url, **kw: _Resp(body=ready if url.endswith("/ready") else papers))
    stream = [
        'data: {"event": "node_update", "node": "retriever", "status": "Searching the indexed papers", "detail": "5 new passages"}',
        ": keepalive",
        "data: " + json.dumps(RESULT),
        'data: {"event": "done"}',
    ]
    monkeypatch.setattr(requests, "post", lambda url, **kw: _Resp(lines=stream))

    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "Ready · 2 papers" in at.sidebar.success[0].value
    at.chat_input[0].set_value("What does the evaluator do?").run()
    assert not at.exception
    markdown = " ".join(m.value for m in at.markdown)
    assert "lightweight retrieval evaluator [1]" in markdown
    assert "arXiv:2401.15884" in markdown and "p. 3" in markdown
    assert any("TAVILY_API_KEY" in w.value for w in at.warning)
    assert any("no unsupported claims" in s.value for s in at.success)


def test_app_shows_offline_and_http_errors(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import requests

    def offline(*a, **k):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(requests, "get", offline)
    monkeypatch.setattr(requests, "post", lambda url, **kw: _Resp(status=429))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert "Backend offline" in at.sidebar.error[0].value
    at.chat_input[0].set_value("hello").run()
    assert any("Rate limit" in e.value for e in at.error)
