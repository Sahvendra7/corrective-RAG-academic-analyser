"""FastAPI endpoints and SSE framing (real graph + tiny index + fake LLM; fake graphs for failures)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.config import get_settings
from tests.conftest import keyword_grader


def events(response) -> list[dict]:
    out = []
    for block in response.text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            out.append(json.loads(block[len("data: "):]))
        elif block:
            assert block.startswith(":"), f"malformed SSE block: {block!r}"
    return out


@pytest.fixture
def client(tiny_store, fake_llm):
    fake_llm.grade_fn = keyword_grader({"retrieval evaluator": "relevant"})
    with TestClient(create_app()) as c:
        yield c


def test_health_ready_papers(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["index"]["vectors"] == 5
    assert ready.json()["web_search_configured"] is False
    assert [p["arxiv_id"] for p in client.get("/papers").json()["papers"]] == ["2401.00001", "2402.00002"]


def test_ready_reports_missing_llm_key(tiny_store, settings_env):
    with TestClient(create_app()) as c:
        r = c.get("/ready")
    assert r.status_code == 503 and r.json()["llm_configured"] is False


def test_stream_event_sequence_and_result_payload(client):
    r = client.post("/chat/stream", json={"query": "What does the retrieval evaluator do?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = events(r)
    kinds = [e["event"] for e in evs]
    assert kinds[-2:] == ["result", "done"] and kinds.count("result") == 1
    assert [e["node"] for e in evs if e["event"] == "node_update"] == [
        "retriever",
        "grader",
        "generator",
        "hallucination_checker",
    ]
    result = evs[-2]
    assert result["citations"][0]["arxiv_id"] == "2401.00001" and result["citations"][0]["pages"] == "p. 3"
    assert result["evidence"][0]["cited"] is True and result["evidence"][0]["source"] == "faiss"
    assert result["grade"] == "relevant" and result["llm_calls"] == 3
    assert len({e["request_id"] for e in evs}) == 1


def test_non_streaming_chat_and_paper_filter(client):
    r = client.post("/chat", json={"query": "retrieval evaluator", "paper_ids": ["2402.00002"], "allow_web_search": False})
    assert r.status_code == 200
    body = r.json()
    assert all(e["arxiv_id"] == "2402.00002" for e in body["evidence"])


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "    "},
        {"query": "x" * 5000},
        {"query": "ok", "paper_ids": ["../../etc/passwd"]},
        {"query": "ok", "paper_ids": ["2401.00001"] * 60},
        {},
    ],
)
def test_request_validation(client, payload):
    assert client.post("/chat/stream", json=payload).status_code == 422


class _FailingGraph:
    async def astream(self, *args, **kwargs):
        yield ("updates", {"retriever": {"trace": [{"detail": "ok"}]}})
        raise RuntimeError("secret internal detail sk-12345")

    def invoke(self, *args, **kwargs):
        raise RuntimeError("secret internal detail sk-12345")


def test_errors_are_generic_and_stream_still_terminates(tiny_store, fake_llm):
    with TestClient(create_app(graph=_FailingGraph())) as c:
        evs = events(c.post("/chat/stream", json={"query": "q"}))
        assert [e["event"] for e in evs] == ["node_update", "error", "done"]
        assert "sk-12345" not in json.dumps(evs)
        r = c.post("/chat", json={"query": "q"})
        assert r.status_code == 500 and "sk-12345" not in r.text


class _SlowGraph:
    async def astream(self, *args, **kwargs):
        await asyncio.sleep(5)
        yield ("values", {})

    def invoke(self, *args, **kwargs):
        time.sleep(5)


def test_timeout_produces_error_event(tiny_store, fake_llm, settings_env):
    settings_env.setenv("CRAG_REQUEST_TIMEOUT_S", "0.3")
    get_settings.cache_clear()
    with TestClient(create_app(graph=_SlowGraph())) as c:
        started = time.monotonic()
        evs = events(c.post("/chat/stream", json={"query": "q"}))
        assert time.monotonic() - started < 3
    assert [e["event"] for e in evs] == ["error", "done"] and evs[0]["code"] == "timeout"


def test_busy_server_rejects_instead_of_queueing(tiny_store, fake_llm, settings_env):
    settings_env.setenv("CRAG_MAX_CONCURRENT_REQUESTS", "1")
    get_settings.cache_clear()
    app = create_app(graph=_SlowGraph())
    with TestClient(app) as c:
        asyncio.run(app.state.slots.acquire())  # simulate one request in flight
        r = c.post("/chat/stream", json={"query": "q"})
    assert r.status_code == 503 and r.headers["retry-after"] == "5"


def test_rate_limit(tiny_store, fake_llm, settings_env):
    settings_env.setenv("CRAG_RATE_LIMIT", "2/minute")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        codes = [c.post("/chat", json={"query": "retrieval evaluator"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_missing_llm_key_is_reported_cleanly(tiny_store, settings_env):
    from src.pipeline import llm_factory

    llm_factory.reset_llms()
    with TestClient(create_app()) as c:
        evs = events(c.post("/chat/stream", json={"query": "retrieval evaluator"}))
    kinds = [e["event"] for e in evs]
    assert kinds[-2:] == ["error", "done"] and evs[-2]["code"] == "llm_not_configured"


def test_oversized_body_rejected_before_parsing(client):
    body = json.dumps({"query": "x", "padding": "y" * 100_000})
    r = client.post("/chat/stream", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413
