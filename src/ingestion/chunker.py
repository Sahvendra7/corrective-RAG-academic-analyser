"""
Section-aware chunking of academic papers.

Input is either a :class:`~src.ingestion.pdf_parser.ParsedDocument` (preferred:
carries page numbers and section labels) or, for papers whose PDF is not
available, the legacy flat ``.txt`` produced by the v0.1 parser.

Why this replaced the v0.1 chunker
----------------------------------
* v0.1 split on ``(?<=[.!?])\\s+(?=[A-Z])``, which rarely matches PDF text with
  hard line breaks, so chunks averaged 535 words. ``all-MiniLM-L6-v2`` only
  reads the first 256 word-pieces (~190 words): most text was never embedded.
* Bibliographies were chunked and indexed (~20% of chunks).
* Chunks carried no page or section, so citations could not point anywhere.
* ``chunk_all_papers`` crashed (``save_metadata`` called without a path).

Chunks here never cross a section boundary, are capped at
``Settings.chunk_max_words`` words, overlap by whole sentences, and carry
``page_start``/``page_end``/``section``/``heading`` provenance. Output is
deterministic for a given input and settings, so chunk ids are stable.

Usage::

    python -m src.ingestion.chunker            # chunk the corpus → JSONL
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import src.config as config
from src.ingestion.pdf_parser import ParsedDocument, join_lines
from src.ingestion.sections import (
    NON_CONTENT_SECTIONS,
    Section,
    classify_heading,
    looks_like_known_heading,
    strip_numbering,
)

logger = logging.getLogger(__name__)

# Abbreviations after which a period does not end a sentence.
_ABBREVIATIONS = re.compile(
    r"(?:\b(?:e\.g|i\.e|et al|etc|vs|cf|resp|approx|Fig|Figs|Eq|Eqs|Sec|Secs|Tab|Ref|No|Dr|Mr|Ms|Prof|Inc)|\b[A-Z])\.$"
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"“])")


@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    chunk_index: int
    text: str
    section: str
    heading: str
    page_start: int | None
    page_end: int | None
    word_count: int
    char_count: int
    content_hash: str
    source_format: str  # "pdf" | "text" | "metadata"
    title: str = ""
    authors: list[str] | None = None
    published: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Unit:
    """A sentence (or a word window of an over-long sentence) with provenance."""

    text: str
    page: int | None

    @property
    def words(self) -> int:
        return len(self.text.split())


# ── Sentence handling ─────────────────────────────────────────────────────────


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking at "e.g.", "et al.", "Fig. 3"."""
    pieces = [p.strip() for p in _SENTENCE_BOUNDARY.split(text) if p.strip()]
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ABBREVIATIONS.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _units(text: str, page: int | None, max_words: int) -> list[_Unit]:
    out: list[_Unit] = []
    for sentence in split_into_sentences(text):
        words = sentence.split()
        if len(words) <= max_words:
            out.append(_Unit(sentence, page))
            continue
        # Tables, equations and run-on extraction artefacts: hard word windows.
        for start in range(0, len(words), max_words):
            out.append(_Unit(" ".join(words[start : start + max_words]), page))
    return out


# ── Packing ───────────────────────────────────────────────────────────────────


def pack_units(units: list[_Unit], max_words: int, overlap_words: int, min_words: int) -> list[list[_Unit]]:
    """Greedily pack sentences into windows of at most ``max_words`` words.

    Consecutive windows share trailing whole sentences totalling at most
    ``overlap_words`` words. A final window shorter than ``min_words`` is merged
    into the previous one rather than dropped, so no text is lost.
    """
    windows: list[list[_Unit]] = []
    current: list[_Unit] = []
    count = 0
    shared = 0  # number of leading units in `current` carried over as overlap
    for unit in units:
        if current and count + unit.words > max_words:
            windows.append(current)
            overlap: list[_Unit] = []
            carried = 0
            for prev in reversed(current):
                if carried + prev.words > overlap_words or carried + prev.words + unit.words > max_words:
                    break
                overlap.insert(0, prev)
                carried += prev.words
            current, count, shared = overlap, carried, len(overlap)
        current.append(unit)
        count += unit.words
    if len(current) > shared:
        if windows and count < min_words:
            windows[-1] = windows[-1] + current[shared:]
        else:
            windows.append(current)
    return windows


# ── Grouping ──────────────────────────────────────────────────────────────────


@dataclass
class _Group:
    section: str
    heading: str
    units: list[_Unit]


def _groups_from_parsed(doc: ParsedDocument, max_words: int, min_words: int, include: set[str]) -> list[_Group]:
    groups: list[_Group] = []
    keys: list[tuple[str, str, str]] = []
    for para in doc.paragraphs:
        if para.section not in include:
            continue
        key = (para.section, para.heading, para.subheading)
        if not keys or keys[-1] != key:
            label = f"{para.heading} › {para.subheading}" if para.subheading else para.heading
            groups.append(_Group(para.section, label, []))
            keys.append(key)
        groups[-1].units.extend(_units(para.text, para.page, max_words))
    # A tiny group (a heading followed by one short sentence) joins the next
    # group of the same section instead of becoming a near-empty chunk.
    merged: list[_Group] = []
    for i, group in enumerate(groups):
        nxt = groups[i + 1] if i + 1 < len(groups) else None
        if nxt is not None and nxt.section == group.section and sum(u.words for u in group.units) < min_words:
            nxt.units[:0] = group.units
            continue
        merged.append(group)
    return merged


def _groups_from_text(text: str, max_words: int, include: set[str]) -> list[_Group]:
    """Recover coarse structure from a legacy line-wrapped ``.txt`` file."""
    lines = [ln.strip() for ln in text.splitlines()]
    groups: list[_Group] = []
    section, heading = Section.OTHER.value, ""
    buffer: list[str] = []

    def flush() -> None:
        if buffer and section in include:
            body = join_lines(buffer)
            if body:
                groups.append(_Group(section, heading, _units(body, None, max_words)))
        buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        candidate = line
        # "4.2" on one line and "Retrieval Evaluator" on the next.
        if re.fullmatch(r"\d+(\.\d+)*\.?", line) and i + 1 < len(lines):
            candidate = f"{line} {lines[i + 1]}"
        if looks_like_known_heading(candidate) or looks_like_known_heading(line):
            title = candidate if looks_like_known_heading(candidate) else line
            flush()
            canonical = classify_heading(title)
            if section == Section.REFERENCES.value and canonical == Section.OTHER:
                canonical = Section.APPENDIX
            section, heading = canonical.value, strip_numbering(title)
            i += 2 if title == candidate and candidate != line else 1
            continue
        if line:
            buffer.append(line)
        i += 1
    flush()
    return groups


# ── Public API ────────────────────────────────────────────────────────────────


def _included_sections(index_references: bool) -> set[str]:
    excluded = {s.value for s in NON_CONTENT_SECTIONS}
    if index_references:
        excluded.discard(Section.REFERENCES.value)
    return {s.value for s in Section} - excluded


def _make_chunks(
    groups: Iterable[_Group],
    arxiv_id: str,
    paper: dict[str, Any],
    source_format: str,
    settings: config.Settings,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for group in groups:
        windows = pack_units(
            group.units, settings.chunk_max_words, settings.chunk_overlap_words, settings.chunk_min_words
        )
        for window in windows:
            text = " ".join(u.text for u in window).strip()
            if not text:
                continue
            pages = [u.page for u in window if u.page is not None]
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{arxiv_id}_chunk_{index}",
                    arxiv_id=arxiv_id,
                    chunk_index=index,
                    text=text,
                    section=group.section,
                    heading=group.heading,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    word_count=len(text.split()),
                    char_count=len(text),
                    content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
                    source_format=source_format,
                    title=paper.get("title", ""),
                    authors=list(paper.get("authors", [])),
                    published=paper.get("published", ""),
                    url=paper.get("url", ""),
                )
            )
    return chunks


def _with_metadata_abstract(groups: list[_Group], paper: dict[str, Any], max_words: int) -> list[_Group]:
    """Guarantee the abstract is indexed: use arXiv's abstract if parsing found none."""
    if any(g.section == Section.ABSTRACT.value for g in groups):
        return groups
    abstract = re.sub(r"\s+", " ", paper.get("abstract", "")).strip()
    if not abstract:
        return groups
    return [_Group(Section.ABSTRACT.value, "Abstract (arXiv metadata)", _units(abstract, None, max_words)), *groups]


def chunk_parsed_document(
    doc: ParsedDocument, paper: dict[str, Any] | None = None, settings: config.Settings | None = None
) -> list[Chunk]:
    """Chunk a structured document; references/front matter excluded by default."""
    settings = settings or config.get_settings()
    paper = paper or {}
    groups = _groups_from_parsed(doc, settings.chunk_max_words, settings.chunk_min_words, _included_sections(settings.index_references))
    groups = _with_metadata_abstract(groups, paper, settings.chunk_max_words)
    chunks = _make_chunks(groups, doc.doc_id, paper, "pdf", settings)
    # The metadata abstract has no page; everything else came from the PDF.
    for chunk in chunks:
        if chunk.heading == "Abstract (arXiv metadata)":
            chunk.source_format = "metadata"
    return chunks


def chunk_text(
    text: str, arxiv_id: str, paper: dict[str, Any] | None = None, settings: config.Settings | None = None
) -> list[dict[str, Any]]:
    """Chunk a legacy flat text file (no page numbers). Returns chunk dicts."""
    settings = settings or config.get_settings()
    paper = paper or {}
    groups = _groups_from_text(text, settings.chunk_max_words, _included_sections(settings.index_references))
    groups = _with_metadata_abstract(groups, paper, settings.chunk_max_words)
    chunks = _make_chunks(groups, arxiv_id, paper, "text", settings)
    for chunk in chunks:
        if chunk.heading == "Abstract (arXiv metadata)":
            chunk.source_format = "metadata"
    return [c.to_dict() for c in chunks]


def embedding_text(chunk: dict[str, Any], contextual_headers: bool = True) -> str:
    """Text that is embedded for a chunk: optionally prefixed with title and section.

    The prefix lets a passage that never names its paper ("our method …") be
    found by queries that do. The stored/displayed text is unchanged.
    """
    if not contextual_headers:
        return chunk["text"]
    header = " | ".join(x for x in (chunk.get("title", ""), chunk.get("heading", "")) if x)
    return f"{header}\n{chunk['text']}" if header else chunk["text"]


# ── Corpus ────────────────────────────────────────────────────────────────────


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def chunk_corpus(
    metadata: dict[str, dict[str, Any]],
    settings: config.Settings | None = None,
    prefer_pdf: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Chunk every paper in ``metadata``.

    Uses the structured PDF parse when the PDF is in ``data/raw`` (cached in
    ``data/processed/structured``), otherwise the legacy ``.txt``.
    """
    from src.ingestion.pdf_parser import PDFParseError, find_pdf, load_structured, parse_pdf, structured_path
    from src.utils.metadata_utils import save_metadata

    settings = settings or config.get_settings()
    stats: Counter[str] = Counter()
    chunks: list[dict[str, Any]] = []
    for arxiv_id in sorted(metadata):
        paper = metadata[arxiv_id]
        doc = load_structured(arxiv_id) if prefer_pdf else None
        if doc is None and prefer_pdf:
            pdf = find_pdf(arxiv_id, paper)
            if pdf is not None:
                try:
                    doc = parse_pdf(pdf, doc_id=arxiv_id)
                    save_metadata(doc.to_dict(), structured_path(arxiv_id))
                except PDFParseError as exc:
                    logger.warning("parse failed for %s, falling back to text: %s", arxiv_id, exc)
                    stats["parse_failed"] += 1
        if doc is not None:
            paper_chunks = [c.to_dict() for c in chunk_parsed_document(doc, paper, settings)]
            stats["from_pdf"] += 1
        else:
            txt = _resolve(paper.get("txt_path")) or config.TEXT_DIR / f"{arxiv_id}.txt"
            if not txt.exists():
                stats["missing"] += 1
                continue
            paper_chunks = chunk_text(txt.read_text(encoding="utf-8"), arxiv_id, paper, settings)
            stats["from_text"] += 1
        if not paper_chunks:
            stats["empty"] += 1
            continue
        chunks.extend(paper_chunks)
    stats["chunks"] = len(chunks)
    return chunks, dict(stats)


def write_jsonl(chunks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    import argparse

    from src.utils.metadata_utils import load_metadata

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Chunk the corpus into a JSONL file")
    ap.add_argument("--out", type=Path, default=config.PROCESSED_DIR / "corpus_chunks.jsonl")
    ap.add_argument("--text-only", action="store_true", help="ignore PDFs; chunk legacy .txt files")
    args = ap.parse_args()
    metadata = load_metadata(config.META_FILE)
    chunks, stats = chunk_corpus(metadata, prefer_pdf=not args.text_only)
    write_jsonl(chunks, args.out)
    logger.info("Wrote %d chunks to %s — %s", len(chunks), args.out, stats)


if __name__ == "__main__":
    main()
