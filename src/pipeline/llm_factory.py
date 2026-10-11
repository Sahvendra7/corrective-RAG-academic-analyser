"""
LLM factory for the CRAG pipeline.

    get_llm()           temperature = Settings.llm_temperature   (grader, rewriter, verifier)
    get_creative_llm()  alias of get_llm() kept for v0.1 imports
    get_generator_llm() temperature = Settings.generator_temperature

Model, timeout and retry count come from :mod:`src.config`. v0.1 used
``max_retries=0``, so a single transient 429 failed a node and silently turned
into an "ambiguous" grade or a "hallucinated" verdict.

``override_llms(fake)`` is the seam tests (and offline demos) use to run the
whole graph without network access.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import src.config as config

logger = logging.getLogger(__name__)

GEMINI_MODEL = config.GEMINI_MODEL

_lock = threading.Lock()
_instances: dict[str, Any] = {}
_override: Any | None = None


class LLMNotConfiguredError(RuntimeError):
    """Raised when no API key is configured for the LLM provider."""


def _build(temperature: float) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = config.get_settings()
    api_key = settings.gemini_key()
    if not api_key:
        raise LLMNotConfiguredError(
            "GEMINI_API_KEY is not set. Add it to .env (https://aistudio.google.com/app/apikey)."
        )
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=temperature,
        google_api_key=api_key,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )
    logger.info("LLM loaded: %s (temperature=%s)", settings.llm_model, temperature)
    return llm


def _get(kind: str, temperature: float) -> Any:
    if _override is not None:
        return _override
    if kind not in _instances:
        with _lock:
            if kind not in _instances:
                _instances[kind] = _build(temperature)
    return _instances[kind]


def get_llm() -> Any:
    return _get("default", config.get_settings().llm_temperature)


def get_creative_llm() -> Any:
    return get_llm()


def get_generator_llm() -> Any:
    return _get("generator", config.get_settings().generator_temperature)


def is_configured() -> bool:
    return _override is not None or bool(config.get_settings().gemini_key())


def reset_llms() -> None:
    """Drop cached clients (e.g. after changing settings)."""
    with _lock:
        _instances.clear()


@contextmanager
def override_llms(llm: Any) -> Iterator[Any]:
    """Route every ``get_*llm()`` call to ``llm`` inside the ``with`` block."""
    global _override
    previous, _override = _override, llm
    try:
        yield llm
    finally:
        _override = previous


def response_text(response: Any) -> str:
    """Text of a chat-model response; Gemini may return a list of content blocks."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content).strip()
    return str(content or "").strip()
