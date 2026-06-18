"""
pipeline/llm_factory.py
------------------------
Centralized LLM factory for the CRAG pipeline.

Provides singleton instances of the Google Gemini LLM at different
temperature settings for use across all pipeline nodes:

    get_llm()           → temperature=0.0 (grader, hallucination checker)
    get_creative_llm()  → temperature=0.3 (rewriter)
    get_generator_llm() → temperature=0.2 (generator)

Uses the free Gemini 1.5 Flash model via langchain-google-genai.
"""

import logging
import os
import threading
from langchain_google_genai import ChatGoogleGenerativeAI

import src.config as config

logger = logging.getLogger(__name__)

# ── Model Config ──────────────────────────────────────────────────────────────

GEMINI_MODEL = config.GEMINI_MODEL

# ── Thread-Safe Singletons ────────────────────────────────────────────────────

_lock = threading.Lock()
_llm: ChatGoogleGenerativeAI | None = None
_creative_llm: ChatGoogleGenerativeAI | None = None
_generator_llm: ChatGoogleGenerativeAI | None = None


def _get_api_key() -> str:
    """Load and validate the Gemini API key from the environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment. "
            "Add it to your .env file.\n"
            "Get a free API key at: https://aistudio.google.com/app/apikey"
        )
    return api_key


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return a singleton LLM instance with temperature=0.0.

    Used by: grader, hallucination checker.
    """
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                api_key = _get_api_key()
                _llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    temperature=0.0,
                    google_api_key=api_key,
                )
                logger.info(f"[LLM_FACTORY] LLM loaded: {GEMINI_MODEL} (temp=0.0)")
    return _llm


def get_creative_llm() -> ChatGoogleGenerativeAI:
    """
    Return a singleton LLM instance with temperature=0.3.

    Used by: rewriter (needs slight creativity for diverse rewrites).
    """
    global _creative_llm
    if _creative_llm is None:
        with _lock:
            if _creative_llm is None:
                api_key = _get_api_key()
                _creative_llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    temperature=0.3,
                    google_api_key=api_key,
                )
                logger.info(f"[LLM_FACTORY] Creative LLM loaded: {GEMINI_MODEL} (temp=0.3)")
    return _creative_llm


def get_generator_llm() -> ChatGoogleGenerativeAI:
    """
    Return a singleton LLM instance with temperature=0.2.

    Used by: generator (low temp for factual, grounded answers).
    """
    global _generator_llm
    if _generator_llm is None:
        with _lock:
            if _generator_llm is None:
                api_key = _get_api_key()
                _generator_llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    temperature=0.2,
                    google_api_key=api_key,
                )
                logger.info(f"[LLM_FACTORY] Generator LLM loaded: {GEMINI_MODEL} (temp=0.2)")
    return _generator_llm


def reset_llms():
    """Reset all singleton LLM instances. Useful for testing."""
    global _llm, _creative_llm, _generator_llm
    with _lock:
        _llm = None
        _creative_llm = None
        _generator_llm = None
        logger.info("[LLM_FACTORY] All LLM instances reset")
