"""Corrective control flow end to end: real graph + real FAISS index, fake LLM and fake Tavily."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import get_settings
from src.pipeline.graph import recursion_limit, route_after_grader, run_query
from src.pipeline.nodes.generator import ABSTAIN_MESSAGE
from src.pipeline.nodes.grader import GRADER_FAILED
from src.pipeline.nodes.retriever import INDEX_UNAVAILABLE, set_store
from src.pipeline.nodes.web_search import WEB_DISABLED, WEB_NO_KEY
from tests.conftest import keyword_grader

WEB_RESULT = {"title": "CRAG explained", "url": "https://example.com/crag", "content": "Corrective RAG uses a retrieval evaluator.", "score": 0.9}


def nodes(state):
    return [t["node"] for t in state["trace"]]


def test_relevant_path_generates_cited_answer(tiny_store, fake_llm):
    fake_llm.grade_fn = keyword_grader({"retrieval evaluator": "relevant"})
    state = run_query("What does the retrieval evaluator do?")
    assert nodes(state) == ["retriever", "grader", "generator", "hallucination_checker"]
    assert state["grade"] == "relevant" and state["source"] == "faiss"
    assert [d["grade"] for d in state["evidence"]] == ["relevant"]  # only relevant passages
    assert state["citations"][0]["chunk_id"] == "2401.00001_chunk_0"
    assert "**Sources**" in state["generation"] and "p. 3" in state["generation"]
    assert fake_llm.calls == ["grade", "generate", "verify"]  # one batched grading call
    assert state["llm_calls"] == 3


def test_ambiguous_loop_is_bounded_and_never_duplicates(tiny_store, fake_llm, settings_env):
    settings_env.setenv("CRAG_MAX_REWRITES", "2")
    settings_env.setenv("CRAG_TOP_K", "2")
    get_settings.cache_clear()
    fake_llm.grade_fn = lambda q, t: "ambiguous"
    state = run_query("Something only loosely covered")
    assert state["retry_count"] == 2
    assert nodes(state).count("rewriter") == 2
    ids = [d["chunk_id"] for d in state["documents"]]
    assert len(ids) == len(set(ids)) == 5  # retries fetched new passages only
    assert WEB_NO_KEY in state["warnings"]
    assert state["evidence"] and all(d["grade"] == "ambiguous" for d in state["evidence"])


@pytest.mark.parametrize("max_rewrites", [0, 1, 3, 5])
def test_termination_within_recursion_limit(tiny_store, fake_llm, settings_env, max_rewrites):
    settings_env.setenv("CRAG_MAX_REWRITES", str(max_rewrites))
    settings_env.setenv("CRAG_TOP_K", "1")
    get_settings.cache_clear()
    fake_llm.grade_fn = lambda q, t: "ambiguous"
    state = run_query("never good enough")
    assert state["retry_count"] == max_rewrites
    assert len(state["trace"]) <= recursion_limit()
    assert not state.get("error")


def test_irrelevant_goes_to_web_and_web_results_are_graded(tiny_store, fake_llm, fake_web):
    results, calls = fake_web
    results += [WEB_RESULT, {"title": "Spam", "url": "https://spam.example", "content": "Buy cheap watches.", "score": 0.5}]
    fake_llm.grade_fn = keyword_grader({"corrective rag": "relevant"})
    state = run_query("What is corrective RAG?")
    assert nodes(state)[:3] == ["retriever", "grader", "web_search"]
    assert state["retry_count"] == 0 and state["web_search_used"] and calls == ["What is corrective RAG?"]
    assert [d["url"] for d in state["evidence"]] == ["https://example.com/crag"]  # spam graded out
    assert state["source"] == "web" and state["citations"][0]["source"] == "web"


def test_no_evidence_abstains_without_generation_call(tiny_store, fake_llm):
    fake_llm.grade_fn = lambda q, t: "irrelevant"
    state = run_query("Who won the 1998 football world cup?", allow_web_search=False)
    assert state["generation"] == ABSTAIN_MESSAGE.format(web="")
    assert "generate" not in fake_llm.calls and "verify" not in fake_llm.calls
    assert WEB_DISABLED in state["warnings"] and state["source"] == "none"


def test_irrelevant_without_web_tries_rewrites_first(tiny_store, fake_llm, settings_env):
    settings_env.setenv("CRAG_MAX_REWRITES", "1")
    get_settings.cache_clear()
    fake_llm.grade_fn = lambda q, t: "irrelevant"
    state = run_query("unknown topic")
    assert "rewriter" in nodes(state) and state["retry_count"] == 1


def test_grader_failure_is_visible_and_does_not_loop(tiny_store, fake_llm):
    fake_llm.fail = {"grade"}
    state = run_query("What does the retrieval evaluator do?")
    assert "rewriter" not in nodes(state)
    assert GRADER_FAILED in state["warnings"]
    assert state["evidence"] and all(d["grade"] == "ungraded" for d in state["evidence"])


def test_verifier_failure_marks_answer_unverified(tiny_store, fake_llm):
    fake_llm.fail = {"verify"}
    state = run_query("What does the retrieval evaluator do?")
    assert state["hallucination"] is True and state["grounding"]["checked"] is False
    assert "Unverified" in state["generation"]


def test_unsupported_claims_and_numbers_are_annotated(tiny_store, fake_llm):
    fake_llm.grade_fn = keyword_grader({"popqa": "relevant"})
    fake_llm.answer_fn = lambda q, s: "CRAG improves PopQA by 42.7 points [1]."
    fake_llm.unsupported = ["improves PopQA by 42.7 points"]
    state = run_query("How much does CRAG improve PopQA?")
    assert state["hallucination"] is True
    assert state["grounding"]["unsupported_numbers"] == ["42.7"]
    assert "Possibly unsupported" in state["generation"] and "42.7" in state["generation"]


def test_fabricated_citations_are_removed(tiny_store, fake_llm):
    fake_llm.grade_fn = keyword_grader({"retrieval evaluator": "relevant"})
    fake_llm.answer_fn = lambda q, s: "It scores documents [1][4]. Sources: [4] Invented Paper, 2099"
    state = run_query("What does the retrieval evaluator do?")
    assert state["answer_text"] == "It scores documents [1]."
    assert state["grounding"]["invalid_citations"] == [4]
    assert "Invented Paper" not in state["generation"]


def test_generation_failure_is_reported(tiny_store, fake_llm):
    fake_llm.fail = {"generate"}
    state = run_query("What does the retrieval evaluator do?")
    assert "could not be generated" in state["generation"] and state["error"]
    assert "hallucination_checker" not in nodes(state)


def test_missing_index_is_reported_not_hidden(settings_env, fake_llm):
    from src.vectorstore.embeddings import HashingEmbedder
    from src.vectorstore.faiss_store import FAISSStore

    set_store(FAISSStore(index_dir=Path("/nonexistent"), embedder=HashingEmbedder()))
    try:
        state = run_query("anything", allow_web_search=False)
    finally:
        set_store(None)
    assert INDEX_UNAVAILABLE in state["warnings"]
    assert state["source"] == "none"


def test_paper_filter_restricts_retrieval(tiny_store, fake_llm):
    state = run_query("retrieval evaluator", paper_ids=["2402.00002"])
    assert {d["arxiv_id"] for d in state["documents"]} == {"2402.00002"}


def test_routing_table():
    base = {"retry_count": 0, "allow_web_search": True}
    assert route_after_grader({**base, "grade": "relevant"}) == "generator"
    assert route_after_grader({**base, "grade": "ambiguous"}) == "rewriter"
    assert route_after_grader({**base, "grade": "ambiguous", "grader_failed": True}) == "generator"
    assert route_after_grader({**base, "grade": "ambiguous", "retry_count": 99}) == "web_search"


def test_missing_api_key_gives_clean_error(tiny_store, settings_env):
    from src.pipeline import llm_factory
    from src.pipeline.graph import LLM_NOT_CONFIGURED

    llm_factory.reset_llms()
    state = run_query("What does the retrieval evaluator do?")
    assert state["error"] == "LLMNotConfiguredError"
    assert state["generation"] == LLM_NOT_CONFIGURED
