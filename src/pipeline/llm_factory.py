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

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logger = logging.getLogger(__name__)

# ── Model Config ──────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-3.1-flash-lite"
API_TIMEOUT = 60.0  # Hard network timeout in seconds
MAX_RETRIES = 2     # Stop trying after 2 failed network requests

# ── Singletons ────────────────────────────────────────────────────────────────

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


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Return a singleton LLM instance with temperature=0.0.

    Used by: grader, hallucination checker.
    """
    global _llm
    if _llm is None:
        api_key = _get_api_key()
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=temperature,
            google_api_key=api_key,
            timeout=API_TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        logger.info(f"[LLM_FACTORY] LLM loaded: {GEMINI_MODEL} (temp={temperature})")
    return _llm


def get_creative_llm() -> ChatGoogleGenerativeAI:
    """
    Return a singleton LLM instance with temperature=0.3.

    Used by: rewriter (needs slight creativity for diverse rewrites).
    """
    global _creative_llm
    if _creative_llm is None:
        api_key = _get_api_key()
        _creative_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.3,
            google_api_key=api_key,
            timeout=API_TIMEOUT,
            max_retries=MAX_RETRIES,
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
        api_key = _get_api_key()
        _generator_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.2,
            google_api_key=api_key,
            timeout=API_TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        logger.info(f"[LLM_FACTORY] Generator LLM loaded: {GEMINI_MODEL} (temp=0.2)")
    return _generator_llm
