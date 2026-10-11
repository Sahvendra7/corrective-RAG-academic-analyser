"""Retrieval metric definitions used by src/evaluation/retrieval_benchmark.py."""

from __future__ import annotations

import pytest

from src.evaluation.retrieval_benchmark import evaluate, evidence_recall, looks_like_bibliography, reciprocal_rank


def test_evidence_recall_is_normalised_and_union_based():
    gold = "the retrieval evaluator assigns a confidence score to every retrieved document"
    assert evidence_recall(gold, ["The retrieval evalu- ator assigns a confidence score to every retrieved document."]) == 1.0
    half = ["the retrieval evaluator assigns a confidence", "score to every retrieved document"]
    assert 0 < evidence_recall(gold, half[:1]) < evidence_recall(gold, half) <= 1.0
    assert evidence_recall(gold, ["unrelated text about molecules"]) == 0.0


def test_reciprocal_rank_and_bibliography_proxy():
    assert reciprocal_rank("b", ["a", "b", "b"]) == 0.5
    assert reciprocal_rank("z", ["a"]) == 0.0
    assert looks_like_bibliography("A et al. B et al. C et al. arXiv preprint arXiv:1")
    assert not looks_like_bibliography("Yan et al. propose CRAG.")


def test_evaluate_aggregates_per_k():
    dataset = [{"question": "q", "arxiv_id": "p1", "context": "alpha beta gamma delta epsilon zeta"}]
    hits = [
        {"arxiv_id": "p2", "text": "nothing relevant here"},
        {"arxiv_id": "p1", "text": "alpha beta gamma delta epsilon zeta"},
    ]
    s = evaluate(dataset, lambda q, k: hits[:k], delivered_chars=1000)["summary"]
    assert s["paper_hit@1"] == 0.0 and s["paper_hit@2"] == 1.0 and s["paper_mrr"] == 0.5
    assert s["evidence_recall@1"] == 0.0 and s["evidence_recall@2"] == pytest.approx(1.0)
    assert s["words@2"] == 9
