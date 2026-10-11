"""
Shared fixtures for the hermetic test suite.

Nothing here touches the network, real API keys, the user's ``data/`` folder
or a downloaded embedding model:

* ``FakeLLM`` stands in for Gemini at the LangChain boundary (``invoke`` and
  ``with_structured_output``) and records every call.
* ``HashingEmbedder`` replaces sentence-transformers.
* ``make_pdf`` builds small academic-looking PDFs with PyMuPDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest
from langchain_core.messages import AIMessage

import src.config as config
from src.pipeline import llm_factory
from src.pipeline.nodes import retriever as retriever_mod
from src.pipeline.nodes import web_search as web_mod

_SOURCE = re.compile(r'<source id="(\d+)"([^>]*)>\n(.*?)\n</source>', re.DOTALL)


def parse_prompt(messages) -> tuple[str, list[str]]:
    """Return (question, [source texts]) from a pipeline prompt."""
    human = messages[-1].content
    question = re.search(r"Question: (.*)", human)
    return (question.group(1).strip() if question else ""), [m.group(3) for m in _SOURCE.finditer(human)]


def keyword_grader(keywords: dict[str, str]) -> Callable[[str, str], str]:
    """Grade a passage by the first keyword it contains; irrelevant otherwise."""

    def grade(question: str, text: str) -> str:
        for word, value in keywords.items():
            if word.lower() in text.lower():
                return value
        return "irrelevant"

    return grade


@dataclass
class FakeLLM:
    grade_fn: Callable[[str, str], str] = field(default_factory=lambda: (lambda q, t: "relevant"))
    answer_fn: Callable[[str, list[str]], str] = field(default_factory=lambda: (lambda q, s: "The answer is X [1]."))
    rewrite_fn: Callable[[str], str] = field(default_factory=lambda: (lambda q: f"{q} expanded"))
    unsupported: list[str] = field(default_factory=list)
    fail: set[str] = field(default_factory=set)  # kinds that raise: grade, generate, rewrite, verify
    calls: list[str] = field(default_factory=list)

    def _maybe_fail(self, kind: str) -> None:
        self.calls.append(kind)
        if kind in self.fail:
            raise RuntimeError(f"simulated {kind} failure (429)")

    def invoke(self, messages):
        system = messages[0].content
        if "rewrite questions" in system:
            self._maybe_fail("rewrite")
            question, _ = parse_prompt(messages)
            return AIMessage(content=self.rewrite_fn(question))
        self._maybe_fail("generate")
        question, sources = parse_prompt(messages)
        return AIMessage(content=self.answer_fn(question, sources))

    def with_structured_output(self, schema):
        llm = self

        class _Structured:
            def invoke(self, messages):
                question, sources = parse_prompt(messages)
                if schema.__name__ == "BatchGrades":
                    llm._maybe_fail("grade")
                    return schema(
                        grades=[
                            {"id": i, "grade": llm.grade_fn(question, text), "reason": "fake"}
                            for i, text in enumerate(sources, start=1)
                        ]
                    )
                if schema.__name__ == "HallucinationGrade":
                    llm._maybe_fail("verify")
                    return schema(
                        hallucination=bool(llm.unsupported), reasoning="fake", unsupported_claims=llm.unsupported
                    )
                raise AssertionError(f"unexpected schema {schema}")

        return _Structured()


@pytest.fixture
def settings_env(monkeypatch, tmp_path):
    """Isolated settings: hashing embedder, tmp index dir, no keys unless set by the test."""
    for key in ("GEMINI_API_KEY", "TAVILY_API_KEY", "CRAG_GEMINI_API_KEY", "CRAG_TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CRAG_EMBEDDING_MODEL", "hashing")
    monkeypatch.setenv("CRAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.chdir(tmp_path)  # keep a developer's .env out of the test
    config.get_settings.cache_clear()
    yield monkeypatch
    config.get_settings.cache_clear()


@pytest.fixture
def fake_llm():
    llm = FakeLLM()
    with llm_factory.override_llms(llm):
        yield llm


PAPERS: dict[str, dict[str, Any]] = {
    "2401.00001": {
        "title": "Corrective Retrieval Augmented Generation",
        "authors": ["A. Author"],
        "published": "2024-01-29",
        "url": "http://arxiv.org/abs/2401.00001v1",
        "abstract": "We propose CRAG, a retrieval evaluator that triggers web search when retrieval is wrong.",
    },
    "2402.00002": {
        "title": "Graph Neural Networks for Molecules",
        "authors": ["B. Author"],
        "published": "2024-02-10",
        "url": "http://arxiv.org/abs/2402.00002v1",
        "abstract": "Message passing networks predict molecular properties.",
    },
}

CHUNK_TEXTS = {
    "2401.00001": [
        ("method", "Retrieval Evaluator", 3, "The lightweight retrieval evaluator scores each retrieved document and assigns correct, incorrect or ambiguous confidence."),
        ("method", "Knowledge Refinement", 4, "Knowledge refinement decomposes documents into strips and filters irrelevant strips before generation."),
        ("results", "Results", 6, "CRAG improves accuracy on PopQA by 19.0 points over standard RAG with the same retriever."),
    ],
    "2402.00002": [
        ("method", "Message Passing", 2, "Message passing neural networks aggregate neighbour features to predict molecular energy."),
        ("experiments", "Datasets", 5, "We evaluate on QM9 with 134,000 small organic molecules."),
    ],
}


def make_chunks() -> list[dict[str, Any]]:
    import hashlib

    chunks = []
    for pid, rows in CHUNK_TEXTS.items():
        for i, (section, heading, page, text) in enumerate(rows):
            chunks.append(
                {
                    "chunk_id": f"{pid}_chunk_{i}",
                    "arxiv_id": pid,
                    "chunk_index": i,
                    "text": text,
                    "section": section,
                    "heading": heading,
                    "page_start": page,
                    "page_end": page,
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "content_hash": hashlib.sha1(text.encode()).hexdigest()[:16],
                    "source_format": "pdf",
                    **{k: PAPERS[pid][k] for k in ("title", "authors", "published", "url")},
                }
            )
    return chunks


@pytest.fixture
def tiny_store(settings_env, tmp_path):
    """A real FAISS index over five synthetic chunks, installed as the pipeline's store."""
    from src.vectorstore.embeddings import HashingEmbedder
    from src.vectorstore.faiss_store import FAISSStore, build_index

    build_index(make_chunks(), tmp_path / "index", embedder=HashingEmbedder())
    store = FAISSStore(index_dir=tmp_path / "index", embedder=HashingEmbedder())
    retriever_mod.set_store(store)
    yield store
    retriever_mod.set_store(None)


@pytest.fixture
def fake_web(settings_env):
    """Pretend Tavily is configured; returns a list the test can fill with results."""
    settings_env.setenv("TAVILY_API_KEY", "test-key")
    config.get_settings.cache_clear()
    results: list[dict[str, Any]] = []
    calls: list[str] = []

    class _Client:
        def search(self, query, **kwargs):
            calls.append(query)
            return {"results": list(results)}

    web_mod._tavily = _Client()
    yield results, calls
    web_mod._tavily = None


# ── Synthetic PDFs ────────────────────────────────────────────────────────────


@dataclass
class TextItem:
    text: str
    page: int = 1
    x: float = 72
    y: float = 100
    size: float = 10
    bold: bool = False


def make_pdf(path: Path, items: list[TextItem], pages: int | None = None, width: float = 612, height: float = 792) -> Path:
    """Write a PDF where each item is an independent text block at (x, y)."""
    import pymupdf

    doc = pymupdf.open()
    n_pages = pages or max(i.page for i in items)
    for _ in range(n_pages):
        doc.new_page(width=width, height=height)
    for item in items:
        page = doc[item.page - 1]
        font = "hebo" if item.bold else "helv"
        page.insert_text((item.x, item.y), item.text, fontsize=item.size, fontname=font)
    doc.save(str(path))
    doc.close()
    return path
