"""
End-to-end integration on a real paper (opt-in).

    pytest -m network tests/test_pipeline.py     # arXiv download + real embedding model
    pytest -m llm tests/test_pipeline.py         # additionally calls Gemini (needs GEMINI_API_KEY)

Everything is written under pytest's ``tmp_path``. The v0.1 version of this
file wrote into ``data/processed`` (overwriting tracked files) and returned
early — i.e. *passed* — when the API key was missing.
"""

from __future__ import annotations

import os

import pytest

from src.config import Settings
from src.ingestion.arxiv_downloader import fetch_pdf
from src.ingestion.chunker import chunk_parsed_document
from src.ingestion.pdf_parser import parse_pdf

TEST_ARXIV_ID = "2401.15884"  # "Corrective Retrieval Augmented Generation"
TEST_QUERY = "How does Corrective RAG handle irrelevant retrieved documents?"
PAPER = {"title": "Corrective Retrieval Augmented Generation", "url": f"http://arxiv.org/abs/{TEST_ARXIV_ID}v1"}


@pytest.fixture(scope="module")
def crag_index(tmp_path_factory):
    from src.vectorstore.embeddings import SentenceTransformerEmbedder
    from src.vectorstore.faiss_store import FAISSStore, build_index

    root = tmp_path_factory.mktemp("crag")
    pdf = root / f"{TEST_ARXIV_ID}.pdf"
    fetch_pdf(f"https://arxiv.org/pdf/{TEST_ARXIV_ID}v1", pdf)
    doc = parse_pdf(pdf, doc_id=TEST_ARXIV_ID)
    settings = Settings()
    chunks = [c.to_dict() for c in chunk_parsed_document(doc, PAPER, settings)]
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    build_index(chunks, root / "index", embedder=embedder, settings=settings)
    return doc, chunks, FAISSStore(index_dir=root / "index", embedder=embedder)


@pytest.mark.network
def test_real_paper_structure(crag_index):
    doc, chunks, _ = crag_index
    sections = {p.section for p in doc.paragraphs}
    assert {"abstract", "introduction", "references"} <= sections
    assert any(p.page > 1 for p in doc.paragraphs if p.section == "method")
    assert "abili- ties" not in doc.text
    assert all(c["word_count"] <= Settings().chunk_max_words + Settings().chunk_min_words for c in chunks)
    assert not any(c["section"] == "references" for c in chunks)


@pytest.mark.network
def test_real_retrieval_finds_the_evaluator(crag_index):
    _, _, store = crag_index
    results = store.search("lightweight retrieval evaluator confidence correct incorrect ambiguous", top_k=3)
    assert results and results[0]["page_start"] is not None
    assert any("evaluator" in r["text"].lower() for r in results)


@pytest.mark.network
@pytest.mark.llm
def test_full_pipeline_with_gemini(crag_index, monkeypatch):
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    from src.pipeline.graph import run_query
    from src.pipeline.nodes.retriever import set_store

    set_store(crag_index[2])
    try:
        state = run_query(TEST_QUERY, allow_web_search=False)
    finally:
        set_store(None)
    assert not state.get("error"), state.get("error")
    assert state["citations"], "answer has no citations"
    assert all(c["arxiv_id"] == TEST_ARXIV_ID for c in state["citations"])
