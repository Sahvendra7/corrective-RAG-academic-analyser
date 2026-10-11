"""Index build/load consistency and retrieval behaviour (hashing embedder, no downloads)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.vectorstore.embeddings import HashingEmbedder
from src.vectorstore.faiss_store import FAISSStore, IndexConsistencyError, build_index
from tests.conftest import make_chunks


@pytest.fixture
def index_dir(settings_env, tmp_path):
    build_index(make_chunks(), tmp_path / "index", embedder=HashingEmbedder())
    return tmp_path / "index"


def load(index_dir):
    return FAISSStore(index_dir=index_dir, embedder=HashingEmbedder())


def test_embedder_vectors_are_normalised_and_deterministic():
    emb = HashingEmbedder(dim=64)
    a, b = emb.encode(["retrieval evaluator", "retrieval evaluator"]), emb.encode(["retrieval evaluator"])
    assert a.dtype == np.float32 and a.shape == (2, 64)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a[0], b[0])


def test_roundtrip_and_manifest(index_dir):
    store = load(index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text())
    assert store.is_ready and store.index.ntotal == len(make_chunks()) == manifest["num_vectors"]
    assert manifest["embedding_model"] == "hashing" and manifest["num_papers"] == 2
    assert store.get_chunk_by_id("2401.00001_chunk_2")["page_start"] == 6


def test_search_returns_provenance_and_cosine_scores(index_dir):
    results = load(index_dir).search("retrieval evaluator assigns confidence", top_k=2)
    assert results[0]["chunk_id"] == "2401.00001_chunk_0"
    assert {"title", "section", "page_start", "url", "score"} <= set(results[0])
    assert -1.0 <= results[-1]["score"] <= results[0]["score"] <= 1.0


def test_paper_filter_is_exact_not_post_filtered(index_dir):
    results = load(index_dir).search("retrieval evaluator", top_k=5, paper_ids=["2402.00002"])
    assert results and {r["arxiv_id"] for r in results} == {"2402.00002"}
    assert load(index_dir).search("anything", paper_ids=["9999.99999"]) == []


def test_exclude_ids_returns_next_ranked(index_dir):
    store = load(index_dir)
    first = store.search("retrieval evaluator", top_k=2)
    second = store.search("retrieval evaluator", top_k=2, exclude_ids={r["chunk_id"] for r in first})
    assert not {r["chunk_id"] for r in first} & {r["chunk_id"] for r in second}


def test_duplicate_text_is_deduplicated(settings_env, tmp_path):
    chunks = make_chunks()
    dup = dict(chunks[0], chunk_id="2401.00001_chunk_99")  # same content_hash
    build_index(chunks + [dup], tmp_path / "dup", embedder=HashingEmbedder())
    results = load(tmp_path / "dup").search(chunks[0]["text"], top_k=3)
    assert len({r["content_hash"] for r in results}) == len(results)


def test_empty_or_missing_index(settings_env, tmp_path):
    store = FAISSStore(index_dir=tmp_path / "missing", embedder=HashingEmbedder())
    assert not store.is_ready and store.search("x") == [] and store.stats() == {"ready": False}
    with pytest.raises(ValueError):
        build_index([], tmp_path / "empty", embedder=HashingEmbedder())


def test_blank_query_returns_nothing(index_dir):
    assert load(index_dir).search("   ") == []


@pytest.mark.parametrize(
    "tamper, match",
    [
        (lambda d: (d / "chunks.jsonl").write_text((d / "chunks.jsonl").read_text().split("\n", 1)[1]), "row mismatch"),
        (lambda d: (d / "faiss.index").unlink(), "missing"),
        (lambda d: (d / "faiss.index").write_bytes(b"garbage"), "cannot read"),
        (
            lambda d: (d / "chunks.jsonl").write_text((d / "chunks.jsonl").read_text().replace("chunk_0", "chunk_X", 1)),
            "checksum",
        ),
    ],
)
def test_corruption_is_detected(index_dir, tamper, match):
    tamper(index_dir)
    with pytest.raises(IndexConsistencyError, match=match):
        load(index_dir)


def test_embedder_mismatch_is_refused(index_dir):
    with pytest.raises(IndexConsistencyError, match="query embedder"):
        FAISSStore(index_dir=index_dir, embedder=HashingEmbedder(dim=128))


def test_rebuild_replaces_atomically(index_dir):
    chunks = make_chunks()[:2]
    build_index(chunks, index_dir, embedder=HashingEmbedder())
    assert load(index_dir).index.ntotal == 2
    assert not list(index_dir.parent.glob(".index.*"))  # no temp/old dirs left behind


def test_duplicate_chunk_ids_rejected(settings_env, tmp_path):
    chunks = make_chunks()
    with pytest.raises(ValueError, match="duplicate"):
        build_index(chunks + [chunks[0]], tmp_path / "x", embedder=HashingEmbedder())


def test_papers_listing(index_dir):
    papers = load(index_dir).papers()
    assert [p["arxiv_id"] for p in papers] == ["2401.00001", "2402.00002"]
    assert papers[0]["num_chunks"] == 3
