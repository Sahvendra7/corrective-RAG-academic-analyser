"""
Canonical academic-paper section labels.

Headings in real papers vary ("3 Our Approach", "IV. EXPERIMENTAL SETUP",
"Limitations and Future Work"). Retrieval and summarisation need a small,
stable vocabulary, so every heading is mapped to one canonical label.
"""

from __future__ import annotations

import re
from enum import Enum


class Section(str, Enum):
    FRONT_MATTER = "front_matter"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHOD = "method"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    LIMITATIONS = "limitations"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGEMENTS = "acknowledgements"
    APPENDIX = "appendix"
    OTHER = "other"


# Ordered: the first matching rule wins, so more specific rules come first.
_RULES: list[tuple[Section, re.Pattern[str]]] = [
    (Section.ABSTRACT, re.compile(r"^abstract\b")),
    (Section.REFERENCES, re.compile(r"^(references?|bibliography|works cited)\b")),
    (Section.ACKNOWLEDGEMENTS, re.compile(r"^acknowledge?ments?\b")),
    (Section.APPENDIX, re.compile(r"^(appendix|appendices|supplementary|supplemental)\b")),
    (Section.LIMITATIONS, re.compile(r"\blimitations?\b|\bthreats to validity\b|\bethic")),
    (Section.CONCLUSION, re.compile(r"\bconclu|\bfuture (work|directions)\b|\bconcluding\b|^summary\b")),
    (Section.RELATED_WORK, re.compile(r"\brelated (work|literature|research)\b|\bliterature review\b|\bprior work\b|\bprevious work\b")),
    (Section.INTRODUCTION, re.compile(r"^introduction\b|^motivation\b")),
    (Section.BACKGROUND, re.compile(r"^(background|preliminar|problem (formulation|definition|statement|setting)|notation)")),
    (Section.RESULTS, re.compile(r"\bresults?\b|\bablation|\banalysis\b|\bfindings\b|\bcomparison\b|\bcase stud")),
    (Section.EXPERIMENTS, re.compile(r"\bexperiment|\bevaluation\b|\bsetup\b|\bimplementation\b|\bdatasets?\b|\bbenchmarks?\b|\bbaselines?\b|\bmetrics?\b|\btraining details\b")),
    (Section.DISCUSSION, re.compile(r"\bdiscussion\b|\binsights?\b|\bimplications?\b")),
    (Section.METHOD, re.compile(r"\bmethod|\bapproach\b|\bmodel\b|\barchitecture\b|\bframework\b|\bproposed\b|\balgorithm\b|\bdesign\b|\bsystem\b|\bformulation\b|\btraining\b|\bobjective\b|\bloss\b")),
]

_NUMBERING = re.compile(r"^\s*((\d+(\.\d+)*)|([ivxlc]+)|([a-z]))[\.\):]?\s+", re.IGNORECASE)

# Sections whose text should not be indexed by default.
NON_CONTENT_SECTIONS = {Section.REFERENCES, Section.ACKNOWLEDGEMENTS, Section.FRONT_MATTER}

# Sections a query intent should prefer (used as a soft retrieval prior).
INTENT_SECTION_PRIORS: dict[str, set[Section]] = {
    "limitations": {Section.LIMITATIONS, Section.DISCUSSION, Section.CONCLUSION},
    "datasets": {Section.EXPERIMENTS, Section.RESULTS},
    "baselines": {Section.EXPERIMENTS, Section.RESULTS},
    "results": {Section.RESULTS, Section.EXPERIMENTS, Section.ABSTRACT},
    "method": {Section.METHOD, Section.ABSTRACT, Section.INTRODUCTION},
    "contribution": {Section.ABSTRACT, Section.INTRODUCTION, Section.CONCLUSION},
    "related_work": {Section.RELATED_WORK, Section.INTRODUCTION},
    "future_work": {Section.CONCLUSION, Section.LIMITATIONS, Section.DISCUSSION},
}


def strip_numbering(heading: str) -> str:
    """Remove leading section numbering such as ``3.2``, ``IV.`` or ``A.``."""
    return _NUMBERING.sub("", heading.strip(), count=1).strip()


def numbering_depth(heading: str) -> int:
    """Return nesting depth implied by numbering (``3`` → 1, ``3.2`` → 2), 0 if none."""
    match = re.match(r"^\s*(\d+(?:\.\d+)*)[\.\)]?\s+\S", heading)
    if not match:
        return 0
    return match.group(1).count(".") + 1


def classify_heading(heading: str) -> Section:
    """Map a raw heading string to a canonical :class:`Section`."""
    text = strip_numbering(heading).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return Section.OTHER
    for section, pattern in _RULES:
        if pattern.search(text):
            return section
    return Section.OTHER


KNOWN_TOP_LEVEL = re.compile(
    r"^\s*((\d+|[IVX]+)[\.\)]?\s+)?(abstract|introduction|related work|background|preliminaries|"
    r"method(s|ology)?|approach|experiments?|experimental (setup|results)|evaluation|results|"
    r"discussion|limitations|conclusions?|references|bibliography|acknowledge?ments?|appendix)\s*$",
    re.IGNORECASE,
)


def looks_like_known_heading(line: str) -> bool:
    """True for a standalone line that is exactly a common section heading."""
    return bool(KNOWN_TOP_LEVEL.match(line.strip()))
