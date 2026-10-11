"""Settings validation and heading classification."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings
from src.ingestion.sections import Section, classify_heading, numbering_depth, strip_numbering


def test_env_overrides_and_secret_masking(settings_env):
    settings_env.setenv("CRAG_TOP_K", "9")
    settings_env.setenv("GEMINI_API_KEY", "sk-very-secret")
    get_settings.cache_clear()
    s = get_settings()
    assert s.top_k == 9
    assert s.gemini_key() == "sk-very-secret"
    assert "sk-very-secret" not in repr(s) and "sk-very-secret" not in str(s.model_dump())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"max_rewrites": 99},
        {"chunk_max_words": 50, "chunk_overlap_words": 50},
        {"chunk_max_words": 50, "chunk_min_words": 60},
        {"llm_temperature": -1},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValidationError):
        Settings(**kwargs)


def test_legacy_constants_match_settings_defaults():
    import src.config as config
    from src.pipeline import state

    defaults = Settings.model_construct()
    assert config.TOP_K == defaults.top_k
    assert config.CHUNK_SIZE == defaults.chunk_max_words
    assert state.MAX_RETRIES == get_settings().max_rewrites  # single source of truth


@pytest.mark.parametrize(
    "heading, expected",
    [
        ("Abstract", Section.ABSTRACT),
        ("1 Introduction", Section.INTRODUCTION),
        ("IV. EXPERIMENTAL SETUP", Section.EXPERIMENTS),
        ("2 Related Work", Section.RELATED_WORK),
        ("5.3 Ablation Study", Section.RESULTS),
        ("Limitations and Future Work", Section.LIMITATIONS),
        ("6 Conclusion", Section.CONCLUSION),
        ("References", Section.REFERENCES),
        ("Acknowledgments", Section.ACKNOWLEDGEMENTS),
        ("Appendix A", Section.APPENDIX),
        ("3 Our Approach", Section.METHOD),
        ("Quantum Gravity", Section.OTHER),
    ],
)
def test_classify_heading(heading, expected):
    assert classify_heading(heading) == expected


def test_numbering_helpers():
    assert strip_numbering("3.2 Training Details") == "Training Details"
    assert strip_numbering("IV. RESULTS") == "RESULTS"
    assert numbering_depth("3.2 Training") == 2
    assert numbering_depth("Results") == 0
