"""Latency fields recorded by the retriever and generator nodes."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.pipeline.nodes.generator import generator_node
from src.pipeline.nodes.retriever import retriever_node
from src.pipeline.state import create_initial_state


def test_retriever_latency_timing(settings_env):
    state = create_initial_state("test query")
    store = MagicMock(is_ready=True)
    store.search.side_effect = lambda *a, **k: (
        time.sleep(0.1),
        [{"chunk_id": "c1", "arxiv_id": "1234.56789", "text": "mock text", "score": 0.9}],
    )[1]
    with patch("src.pipeline.nodes.retriever.get_store", return_value=store):
        result = retriever_node(state)
    assert 95 <= result["retrieval_ms"] <= 1000
    assert result["trace"][-1]["node"] == "retriever"


def test_generator_latency_timing(settings_env):
    state = create_initial_state("test query")
    state["documents"] = [
        {
            "chunk_id": "c1",
            "arxiv_id": "1234.56789",
            "text": "mock text",
            "score": 0.9,
            "title": "Mock Title",
            "url": "http://arxiv.org/abs/1234.56789",
            "grade": "relevant",
            "source": "faiss",
        }
    ]

    def slow_answer(query, evidence, llm):
        time.sleep(0.1)
        return "Mock response answer [1]."

    with patch("src.pipeline.nodes.generator.get_generator_llm", return_value=MagicMock()), patch(
        "src.pipeline.nodes.generator.generate_answer", side_effect=slow_answer
    ):
        result = generator_node(state)
    assert 95 <= result["generation_ms"] <= 1000
