"""
Central configuration.

All tunable values live in :class:`Settings` and can be overridden through
environment variables (prefix ``CRAG_``) or a ``.env`` file, e.g.::

    CRAG_TOP_K=8 CRAG_RERANK_ENABLED=false uvicorn src.api.server:app

API keys keep their conventional unprefixed names (``GEMINI_API_KEY``,
``TAVILY_API_KEY``).

The module-level path constants and legacy names (``TOP_K``, ``CHUNK_SIZE``…)
are kept for backwards compatibility with scripts written against v0.1.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
EVAL_DIR = DATA_DIR / "eval"
REPORTS_DIR = PROJECT_ROOT / "reports"
META_FILE = PROCESSED_DIR / "metadata.json"

RetrievalStrategy = Literal["dense", "bm25", "hybrid"]
GraderMode = Literal["llm", "cross_encoder", "hybrid"]


class Settings(BaseSettings):
    """Runtime configuration. Every field maps to ``CRAG_<FIELD_NAME>``."""

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
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    generator_temperature: float = Field(0.2, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(60.0, gt=0)
    llm_max_retries: int = Field(2, ge=0, le=10)
    judge_model: str | None = None  # evaluation judge; defaults to llm_model

    # ── Embeddings / reranker ─────────────────────────────────────────────────
    # "hashing" selects a deterministic offline embedder (tests / CI only).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_query_prefix: str = ""
    embedding_batch_size: int = Field(64, ge=1)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_max_words: int = Field(170, ge=20)
    chunk_overlap_words: int = Field(30, ge=0)
    chunk_min_words: int = Field(25, ge=1)
    index_references: bool = False  # bibliography text is retrieval noise
    contextual_headers: bool = True  # prepend "Title | Section" before embedding

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_strategy: RetrievalStrategy = "hybrid"
    top_k: int = Field(5, ge=1, le=50)  # evidence passages handed to the generator
    candidate_k: int = Field(30, ge=1, le=500)  # candidates per retriever before fusion
    rrf_k: int = Field(60, ge=1)
    rerank_enabled: bool = True
    rerank_candidates: int = Field(20, ge=1, le=200)
    min_dense_similarity: float = Field(0.0, ge=-1.0, le=1.0)
    max_chunks_per_paper: int = Field(3, ge=1)  # diversity cap in the final evidence set
    section_prior_weight: float = Field(0.15, ge=0.0, le=1.0)
    retrieval_cache_size: int = Field(512, ge=0)

    # ── Corrective policy ─────────────────────────────────────────────────────
    grader_mode: GraderMode = "hybrid"
    # Cross-encoder logit thresholds (ms-marco scale). Calibrated on the golden
    # set — see reports/retrieval_benchmark.md.
    evaluator_upper_threshold: float = 0.0
    evaluator_lower_threshold: float = -6.0
    max_rewrites: int = Field(1, ge=0, le=5)
    max_regenerations: int = Field(1, ge=0, le=3)
    refine_evidence: bool = True
    refine_max_sentences: int = Field(6, ge=1)
    web_search_enabled: bool = True
    web_search_on_ambiguous: bool = False
    web_max_results: int = Field(3, ge=1, le=10)
    web_timeout_s: float = Field(15.0, gt=0)
    web_cache_ttl_s: int = Field(3600, ge=0)
    web_allowed_domains: list[str] = Field(default_factory=list)

    # ── Generation / grounding ────────────────────────────────────────────────
    max_evidence_chars: int = Field(1600, ge=200)
    min_supported_ratio: float = Field(0.6, ge=0.0, le=1.0)
    verify_grounding: bool = True

    # ── API ───────────────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])
    rate_limit: str = "20/minute"
    request_timeout_s: float = Field(120.0, gt=0)
    max_concurrent_requests: int = Field(4, ge=1)
    max_query_chars: int = Field(1000, ge=10)
    max_body_bytes: int = Field(16_384, ge=1024)

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = True
    log_query_text: bool = True  # set false to keep user queries out of logs

    # ── Paths ─────────────────────────────────────────────────────────────────
    index_dir: Path = INDEX_DIR
    metadata_file: Path = META_FILE

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.evaluator_lower_threshold > self.evaluator_upper_threshold:
            raise ValueError("evaluator_lower_threshold must be <= evaluator_upper_threshold")
        if self.chunk_overlap_words >= self.chunk_max_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_max_words")
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
