import pytest
import time
from unittest.mock import patch, MagicMock

from src.pipeline.nodes.retriever import retriever_node
from src.pipeline.nodes.generator import generator_node
from src.pipeline.state import create_initial_state

def test_retriever_latency_timing():
    # Setup initial state
    state = create_initial_state("test query")
    
    # Mock get_store and store.search to have an artificial delay of 100ms
    mock_store = MagicMock()
    mock_store.search.side_effect = lambda *args, **kwargs: (time.sleep(0.1), [{"chunk_id": "test_chunk", "arxiv_id": "1234.56789", "text": "mock text", "score": 0.9}])[1]
    
    with patch("src.pipeline.nodes.retriever.get_store", return_value=mock_store):
        result = retriever_node(state)
        
        # Verify retrieval_ms exists and is approximately 100ms (allowing some tolerance)
        assert "retrieval_ms" in result
        assert 95 <= result["retrieval_ms"] <= 300

def test_generator_latency_timing():
    # Setup initial state
    state = create_initial_state("test query")
    state["relevant_documents"] = [{
        "chunk_id": "test_chunk",
        "arxiv_id": "1234.56789",
        "text": "mock text",
        "score": 0.9,
        "title": "Mock Title",
        "authors": ["Author 1"],
        "published": "2024-01-01",
        "url": "http://arxiv.org/abs/1234.56789",
        "abstract": "Mock abstract",
        "grade": "relevant",
        "source": "faiss"
    }]
    
    # Mock get_generator_llm and generate_answer with artificial delay of 100ms
    mock_llm = MagicMock()
    def mock_generate_answer(query, context, llm):
        time.sleep(0.1)
        return "Mock response answer."
        
    with patch("src.pipeline.nodes.generator.get_generator_llm", return_value=mock_llm), \
         patch("src.pipeline.nodes.generator.generate_answer", side_effect=mock_generate_answer):
        result = generator_node(state)
        
        # Verify generation_ms exists and is approximately 100ms
        assert "generation_ms" in result
        assert 95 <= result["generation_ms"] <= 300
