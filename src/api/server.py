"""
FastAPI service for the CRAG pipeline.

Endpoints
---------
GET  /health        liveness (process is up)
GET  /ready         readiness: index loaded, LLM key present (503 otherwise)
GET  /papers        papers in the index (for the UI's paper filter)
POST /chat          run the pipeline, return the result as JSON
POST /chat/stream   Server-Sent Events: ``node_update`` per pipeline step, then
                    ``result`` (answer, citations, evidence, grounding, trace)
                    and ``done``; or ``error`` then ``done``. Comment lines
                    (``: keepalive``) are sent while a step runs > 15 s.

Limits (all from :mod:`src.config`): query length, paper filter size, a
per-client rate limit (in-memory, per process), a cap on concurrent pipeline
runs, and an overall per-request deadline. Error responses never contain
exception text, stack traces or secrets.

Note: graph nodes are synchronous and run in worker threads. On client
disconnect or timeout the stream stops and no further node starts, but a
node already running (e.g. an LLM call) completes in its thread.

Run::

    uvicorn src.api.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from src.config import get_settings
from src.pipeline import llm_factory
from src.pipeline.evidence import pages_label, passage_text
from src.pipeline.graph import LLM_NOT_CONFIGURED, build_graph, graph_config
from src.pipeline.llm_factory import LLMNotConfiguredError
from src.pipeline.nodes.retriever import get_store
from src.pipeline.state import create_initial_state

logger = logging.getLogger(__name__)

KEEPALIVE_S = 15.0
MAX_BODY_BYTES = 64 * 1024
NODE_STATUS = {
    "retriever": "Searching the indexed papers",
    "grader": "Checking which passages are relevant",
    "rewriter": "Rewriting the query to retrieve better passages",
    "web_search": "Searching the web",
    "generator": "Writing the answer from the selected evidence",
    "hallucination_checker": "Checking the answer against its sources",
}
# arXiv ids as stored in metadata ("2401.15884", old-style "quant-ph_0412125").
ARXIV_ID = r"^[A-Za-z0-9][A-Za-z0-9_\-]*(\.[A-Za-z0-9_\-]+)*$"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["How does Corrective RAG handle irrelevant documents?"])
    paper_ids: list[str] = Field(default_factory=list, max_length=50, description="Restrict retrieval to these papers")
    allow_web_search: bool = Field(True, description="Allow the Tavily fallback for this request")
    thread_id: str | None = Field(None, description="Accepted for v0.1 compatibility; conversations are stateless")

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        limit = get_settings().max_query_chars
        if len(value) > limit:
            raise ValueError(f"query longer than {limit} characters")
        return value

    @field_validator("paper_ids")
    @classmethod
    def _paper_ids(cls, value: list[str]) -> list[str]:
        import re

        bad = [p for p in value if len(p) > 40 or not re.fullmatch(ARXIV_ID, p)]
        if bad:
            raise ValueError("invalid paper id")
        return list(dict.fromkeys(value))


# ── Result serialisation ──────────────────────────────────────────────────────


def serialise_result(state: dict[str, Any]) -> dict[str, Any]:
    """Public JSON view of a final pipeline state (no raw prompts, no secrets)."""
    cited = {c["n"] for c in state.get("citations", [])}
    evidence = [
        {
            "n": n,
            "cited": n in cited,
            "source": doc.get("source", ""),
            "title": doc.get("title", ""),
            "arxiv_id": doc.get("arxiv_id", ""),
            "url": doc.get("url", ""),
            "section": doc.get("heading") or doc.get("section", ""),
            "pages": pages_label(doc),
            "grade": doc.get("grade", ""),
            "score": doc.get("score"),
            "text": passage_text(doc, 700),
        }
        for n, doc in enumerate(state.get("evidence", []), start=1)
    ]
    return {
        "answer": state.get("generation", ""),
        "answer_text": state.get("answer_text", ""),
        "citations": state.get("citations", []),
        "evidence": evidence,
        "grade": state.get("grade", ""),
        "source": state.get("source", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "retry_count": state.get("retry_count", 0),
        "web_search_used": state.get("web_search_used", False),
        "hallucination": state.get("hallucination", False),
        "grounding": state.get("grounding", {}),
        "warnings": state.get("warnings", []),
        "trace": state.get("trace", []),
        "llm_calls": state.get("llm_calls", 0),
        "retrieval_ms": round(state.get("retrieval_ms", 0.0), 1),
        "generation_ms": round(state.get("generation_ms", 0.0), 1),
    }


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(graph: Any = None) -> FastAPI:
    """Build the app. ``graph`` lets tests inject a fake compiled graph."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.graph = graph if graph is not None else build_graph()
        try:
            await run_in_threadpool(get_store)  # load index + embedding model once
        except Exception:
            logger.exception("Index failed to load; /ready will report it")
        yield

    app = FastAPI(
        title="CRAG Academic Analyzer API",
        description="Corrective RAG over arXiv papers with cited, grounded answers",
        version="0.2.0",
        lifespan=lifespan,
    )
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.state.slots = asyncio.Semaphore(settings.max_concurrent_requests)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        # Query length is validated after JSON parsing; refuse oversized bodies before reading them.
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        if request.method == "POST" and length is None:
            return JSONResponse({"detail": "Content-Length required"}, status_code=411)
        return await call_next(request)

    def _graph(request: Request) -> Any:
        g = getattr(request.app.state, "graph", None)
        if g is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialised")
        return g

    def _admit(request: Request) -> asyncio.Semaphore:
        slots: asyncio.Semaphore = request.app.state.slots
        if slots.locked():
            raise HTTPException(status_code=503, detail="Server busy, retry shortly", headers={"Retry-After": "5"})
        return slots

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        try:
            # A failed startup load is retried here; keep that I/O off the event loop.
            index = (await run_in_threadpool(get_store)).stats()
        except Exception:
            index = {"ready": False}
        body = {
            "ready": bool(index.get("ready")) and llm_factory.is_configured(),
            "index": index,
            "llm_configured": llm_factory.is_configured(),
            "web_search_configured": bool(get_settings().web_search_enabled and get_settings().tavily_key()),
        }
        return JSONResponse(body, status_code=200 if body["ready"] else 503)

    @app.get("/papers")
    async def papers() -> dict[str, Any]:
        try:
            store = await run_in_threadpool(get_store)
        except Exception:
            raise HTTPException(status_code=503, detail="Index not available")
        return {"papers": store.papers() if store.is_ready else []}

    @app.post("/chat")
    @limiter.limit(lambda: get_settings().rate_limit)
    async def chat(request: Request, body: ChatRequest) -> dict[str, Any]:
        g, slots = _graph(request), _admit(request)
        state = create_initial_state(body.query, body.paper_ids, body.allow_web_search)
        async with slots:
            try:
                final = await asyncio.wait_for(
                    run_in_threadpool(g.invoke, state, graph_config()), timeout=get_settings().request_timeout_s
                )
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="The pipeline did not finish in time")
            except LLMNotConfiguredError:
                raise HTTPException(status_code=503, detail=LLM_NOT_CONFIGURED)
            except Exception:
                logger.exception("pipeline error")
                raise HTTPException(status_code=500, detail="Internal error while answering")
        return serialise_result(final)

    @app.post("/chat/stream")
    @limiter.limit(lambda: get_settings().rate_limit)
    async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
        g, slots = _graph(request), _admit(request)
        state = create_initial_state(body.query, body.paper_ids, body.allow_web_search)
        return StreamingResponse(
            stream_events(g, state, slots, request_id=uuid.uuid4().hex[:12]),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


async def stream_events(
    graph: Any, state: dict[str, Any], slots: asyncio.Semaphore, request_id: str
) -> AsyncIterator[str]:
    """SSE generator. Always ends with a ``done`` event unless the client disconnects."""
    deadline = time.monotonic() + get_settings().request_timeout_s
    final: dict[str, Any] | None = None
    async with slots:
        stream = graph.astream(state, config=graph_config(), stream_mode=["updates", "values"])
        pending: asyncio.Task | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(stream.__anext__())
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                done, _ = await asyncio.wait({pending}, timeout=min(KEEPALIVE_S, remaining))
                if not done:
                    yield ": keepalive\n\n"
                    continue
                task, pending = pending, None
                try:
                    mode, chunk = task.result()
                except StopAsyncIteration:
                    break
                if mode == "values":
                    final = chunk
                    continue
                for node, update in chunk.items():
                    trace = (update or {}).get("trace") or [{}]
                    yield sse(
                        {
                            "event": "node_update",
                            "node": node,
                            "status": NODE_STATUS.get(node, node),
                            "detail": trace[-1].get("detail", ""),
                            "request_id": request_id,
                        }
                    )
            if final is None:
                raise RuntimeError("pipeline produced no state")
            yield sse({"event": "result", "request_id": request_id, **serialise_result(final)})
        except asyncio.TimeoutError:
            logger.warning("request %s timed out", request_id)
            yield sse({"event": "error", "code": "timeout", "message": "The pipeline did not finish in time."})
        except LLMNotConfiguredError:
            yield sse({"event": "error", "code": "llm_not_configured", "message": LLM_NOT_CONFIGURED})
        except asyncio.CancelledError:
            logger.info("request %s: client disconnected", request_id)
            raise
        except Exception:
            logger.exception("request %s failed", request_id)
            yield sse({"event": "error", "code": "internal", "message": "An internal error occurred. Please try again."})
        finally:
            if pending is not None:
                pending.cancel()
            await _aclose(stream)
    yield sse({"event": "done", "request_id": request_id})


async def _aclose(stream: Any) -> None:
    close = getattr(stream, "aclose", None)
    if close is not None:
        try:
            await close()
        except BaseException:  # noqa: BLE001 - best effort cleanup of a cancelled generator
            pass


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=get_settings().log_level)
    uvicorn.run(app, host="127.0.0.1", port=8000)
