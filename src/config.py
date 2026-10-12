"""
Central configuration.

All tunable values live in :class:`Settings` and can be overridden through
environment variables (prefix ``CRAG_``) or a ``.env`` file, e.g.::

    CRAG_TOP_K=8 CRAG_WEB_SEARCH_ENABLED=false uvicorn src.api.server:app

API keys keep their conventional unprefixed names (``GEMINI_API_KEY``,
``TAVILY_API_KEY``).

The module-level path constants and legacy names (``TOP_K``, ``CHUNK_SIZE``…)
are kept for backwards compatibility with scripts written against v0.1.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEXT_DIR = PROCESSED_DIR / "texts"
CHUNK_DIR = PROCESSED_DIR / "chunks"
STRUCTURED_DIR = PROCESSED_DIR / "structured"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"  # legacy v0.1 artefacts
INDEX_DIR = DATA_DIR / "index"
REPORTS_DIR = PROJECT_ROOT / "reports"
META_FILE = PROCESSED_DIR / "metadata.json"


class Settings(BaseSettings):
    """Runtime configuration. Every field maps to ``CRAG_<FIELD_NAME>``.

    Only settings that are actually read by the code live here; anything
    listed is wired up (see the "Configuration" table in README.md).
    """

    model_config = SettingsConfigDict(
        env_prefix="CRAG_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    # ── Secrets ───────────────────────────────────────────────────────────────
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "CRAG_GEMINI_API_KEY")
    )
    tavily_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("TAVILY_API_KEY", "CRAG_TAVILY_API_KEY")
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model: str = "gemini-3.1-flash-lite"
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)  # grader, rewriter, verifier
    generator_temperature: float = Field(0.2, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(60.0, gt=0)
    llm_max_retries: int = Field(2, ge=0, le=10)  # transient 429/5xx retries per call

    # ── Embeddings ────────────────────────────────────────────────────────────
    # "hashing" selects a deterministic offline embedder (tests / CI only).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(64, ge=1)

    # ── Chunking ──────────────────────────────────────────────────────────────
    # all-MiniLM-L6-v2 truncates input at 256 word-pieces (~190 English words),
    # so chunks are sized to fit inside that window.
    chunk_max_words: int = Field(170, ge=20)
    chunk_overlap_words: int = Field(30, ge=0)
    chunk_min_words: int = Field(25, ge=1)
    index_references: bool = False  # bibliography text is retrieval noise
    # Embed "Title | Section" together with the chunk. Off by default: on the
    # retrieval benchmark it did not help (reports/retrieval_benchmark.md).
    contextual_headers: bool = False

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(5, ge=1, le=50)  # passages retrieved per attempt

    # ── Corrective policy ─────────────────────────────────────────────────────
    max_rewrites: int = Field(2, ge=0, le=5)  # query rewrites before web fallback
    web_search_enabled: bool = True
    web_max_results: int = Field(3, ge=1, le=10)
    web_timeout_s: float = Field(15.0, gt=0)

    # ── Generation / grounding ────────────────────────────────────────────────
    max_evidence_docs: int = Field(6, ge=1, le=20)
    max_evidence_chars: int = Field(1500, ge=200)  # per passage, same for generator and verifier
    verify_grounding: bool = True

    # ── API ───────────────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    rate_limit: str = "20/minute"
    request_timeout_s: float = Field(180.0, gt=0)
    max_concurrent_requests: int = Field(4, ge=1)
    max_query_chars: int = Field(1000, ge=10)

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Paths ─────────────────────────────────────────────────────────────────
    index_dir: Path = INDEX_DIR
    metadata_file: Path = META_FILE

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap_words >= self.chunk_max_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_max_words")
        if self.chunk_min_words > self.chunk_max_words:
            raise ValueError("chunk_min_words must not exceed chunk_max_words")
        return self

    # Convenience accessors that never leak the secret value into reprs/logs.
    def gemini_key(self) -> str | None:
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else None

    def tavily_key(self) -> str | None:
        return self.tavily_api_key.get_secret_value() if self.tavily_api_key else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()


def setup_dirs() -> None:
    """Ensure data directories exist. Must be called explicitly by entry points."""
    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, TEXT_DIR, CHUNK_DIR, STRUCTURED_DIR, INDEX_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# ── Backwards-compatible constants (v0.1 API) ────────────────────────────────
_defaults = Settings.model_construct()
EMBEDDING_MODEL = _defaults.embedding_model
GEMINI_MODEL = _defaults.llm_model
TOP_K = _defaults.top_k
MAX_RETRIES = _defaults.max_rewrites
CHUNK_SIZE = _defaults.chunk_max_words
CHUNK_OVERLAP = _defaults.chunk_overlap_words
MIN_CHUNK_SIZE = _defaults.chunk_min_words
