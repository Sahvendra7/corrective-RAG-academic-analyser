import os
import pytest
import asyncio
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import context_precision
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

def test_context_precision_calculation():
    # Ensure Gemini API key is set
    assert "GEMINI_API_KEY" in os.environ, "GEMINI_API_KEY environment variable is not set"

    # Setup the asyncio loop for Ragas under Windows
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception:
        pass

    # Initialize model wrappers
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.0,
        google_api_key=os.environ["GEMINI_API_KEY"]
    )
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    ragas_judge = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    # Hand-crafted mock dataset:
    # Row 0: First context is relevant, second is irrelevant. Expected CP = 1.0
    # Row 1: First context is irrelevant, second is relevant. Expected CP = 0.5
    # Row 2: Both contexts are irrelevant. Expected CP = 0.0
    data_dict = {
        "question": [
            "What is the capital of France?",
            "What is the capital of France?",
            "What is the capital of France?"
        ],
        "contexts": [
            ["Paris is the capital of France.", "Apples are delicious fruits."],
            ["Apples are delicious fruits.", "Paris is the capital of France."],
            ["Apples are delicious fruits.", "Bananas are yellow."]
        ],
        "ground_truth": [
            "The capital of France is Paris.",
            "The capital of France is Paris.",
            "The capital of France is Paris."
        ],
        "answer": [
            "Paris",
            "Paris",
            "Paris"
        ]
    }
    dataset = Dataset.from_dict(data_dict)

    # Run Ragas evaluation for context_precision
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision],
        llm=ragas_judge,
        embeddings=ragas_embeddings,
        raise_exceptions=True
    )

    df_res = result.to_pandas()
    scores = df_res["context_precision"].tolist()

    print("\nEvaluated Context Precision Scores:", scores)

    # Assertions based on hand-calculated expected ranges (allowing minor LLM reasoning variations)
    # Row 0 (Relevant first): should be close to 1.0 (strict: >= 0.8)
    assert scores[0] >= 0.8, f"Row 0 context precision expected to be high, got {scores[0]}"

    # Row 1 (Irrelevant first): should be close to 0.5 (strict: 0.3 <= x <= 0.7)
    assert 0.3 <= scores[1] <= 0.7, f"Row 1 context precision expected to be moderate (around 0.5), got {scores[1]}"

    # Row 2 (None relevant): should be close to 0.0 (strict: <= 0.2)
    assert scores[2] <= 0.2, f"Row 2 context precision expected to be 0.0, got {scores[2]}"
