"""
Structure-aware PDF parsing for academic papers.

Produces a :class:`ParsedDocument`: an ordered list of paragraphs, each carrying
its page number, the heading it belongs to and the canonical section label.
This replaces the v0.1 parser, which concatenated pages (losing page numbers)
and sorted blocks by ``(y, x)`` — an order that interleaves the two columns of
most conference papers.

Key steps
---------
1. Read text blocks in the PDF's content-stream order (correct for two-column
   LaTeX output in practice), keeping font size and bold flags per line.
2. Remove running headers/footers: short lines in the top/bottom margin whose
   digit-normalised text repeats on many pages, plus bare page numbers.
3. Drop figure internals and footnotes (blocks set well below body font size),
   while keeping "Figure n:" / "Table n:" captions.
4. Detect headings (bold or larger than body text, short, not sentence-like, or
   a standalone well-known section name); merge "1" + "Introduction" splits.
5. Repair hyphenation across line breaks and join wrapped lines.

Usage::

    python -m src.ingestion.pdf_parser            # parse data/raw/*.pdf
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import src.config as config
from src.ingestion.sections import (
    Section,
    classify_heading,
    looks_like_known_heading,
    numbering_depth,
    strip_numbering,
)
from src.utils.metadata_utils import load_metadata, save_metadata

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 500  # below this the PDF is probably scanned / image-only
MAX_PDF_BYTES = 50 * 1024 * 1024  # refuse pathological inputs
MAX_PAGES = 200
MARGIN_FRACTION = 0.08
REPEAT_FRACTION = 0.3
SMALL_FONT_RATIO = 0.82
HEADING_MAX_WORDS = 14

_CAPTION = re.compile(r"^(fig(ure)?|table|tab\.|algorithm)\s*\d+", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
_INLINE_ABSTRACT = re.compile(r"^abstract\s*[—–\-:.]\s*", re.IGNORECASE)


class PDFParseError(Exception):
    """Raised for unreadable, encrypted, oversized or empty PDFs."""


@dataclass
class Paragraph:
    text: str
    page: int  # 1-based
    heading: str  # nearest heading text ("" before the first heading)
    subheading: str
    section: str  # canonical Section value
    kind: str = "body"  # body | caption


@dataclass
class ParsedDocument:
    doc_id: str
    num_pages: int
    paragraphs: list[Paragraph] = field(default_factory=list)
    headings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "num_pages": self.num_pages,
            "headings": self.headings,
            "paragraphs": [asdict(p) for p in self.paragraphs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedDocument:
        return cls(
            doc_id=data["doc_id"],
            num_pages=data["num_pages"],
            headings=data.get("headings", []),
            paragraphs=[Paragraph(**p) for p in data["paragraphs"]],
        )


# ── Text helpers ──────────────────────────────────────────────────────────────


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines, repairing end-of-line hyphenation (``abili-`` + ``ties``)."""
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith("-") and line[:1].islower() and not out.endswith(" -"):
            out = out[:-1] + line
        else:
            out = out + " " + line
    return re.sub(r"\s{2,}", " ", out).strip()


def clean_text(raw_text: str) -> str:
    """Normalise extracted text (kept for backwards compatibility with v0.1)."""
    text = raw_text.replace("\x0c", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\S+@\S+\.\w+", "", text)  # e-mail addresses
    text = re.sub(r"^[-_=]{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalise_for_repeat(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _is_sentence_like(text: str) -> bool:
    return text.rstrip().endswith((".", ",", ";", ":")) and not re.match(r"^\d+(\.\d+)*\.?$", text.strip())


# ── Low-level extraction ──────────────────────────────────────────────────────


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float


@dataclass
class _Block:
    page: int
    lines: list[_Line]
    page_height: float

    @property
    def text(self) -> str:
        return join_lines([ln.text for ln in self.lines])

    @property
    def mean_size(self) -> float:
        total = sum(len(ln.text) for ln in self.lines) or 1
        return sum(ln.size * len(ln.text) for ln in self.lines) / total


def _read_blocks(pdf_path: Path) -> tuple[list[_Block], int]:
    import pymupdf  # local import keeps module import cheap

    if not pdf_path.is_file():
        raise PDFParseError(f"Not a file: {pdf_path.name}")
    if pdf_path.stat().st_size > MAX_PDF_BYTES:
        raise PDFParseError(f"PDF too large: {pdf_path.name}")
    with open(pdf_path, "rb") as fh:
        if fh.read(5) != b"%PDF-":
            raise PDFParseError(f"Not a PDF file: {pdf_path.name}")

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:  # pymupdf raises several exception types
        raise PDFParseError(f"Cannot open PDF {pdf_path.name}: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise PDFParseError(f"Encrypted PDF: {pdf_path.name}")
        num_pages = len(doc)
        blocks: list[_Block] = []
        for page_index in range(min(num_pages, MAX_PAGES)):
            page = doc[page_index]
            height = float(page.rect.height) or 1.0
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue  # image block
                lines: list[_Line] = []
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans)
                    chars = sum(len(s["text"]) for s in spans) or 1
                    size = sum(s["size"] * len(s["text"]) for s in spans) / chars
                    bold = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
                    bbox = line.get("bbox", block["bbox"])
                    lines.append(_Line(text=text, size=size, bold=bold, y0=bbox[1], y1=bbox[3]))
                if lines:
                    blocks.append(_Block(page=page_index + 1, lines=lines, page_height=height))
    return blocks, num_pages


def _body_font_size(blocks: list[_Block]) -> float:
    sizes: Counter[float] = Counter()
    for b in blocks:
        for ln in b.lines:
            sizes[round(ln.size)] += len(ln.text)
    return float(sizes.most_common(1)[0][0]) if sizes else 10.0


def _repeated_margin_lines(blocks: list[_Block], num_pages: int) -> set[str]:
    counts: Counter[str] = Counter()
    for b in blocks:
        seen_on_page: set[str] = set()
        for ln in b.lines:
            in_margin = ln.y1 < b.page_height * MARGIN_FRACTION or ln.y0 > b.page_height * (1 - MARGIN_FRACTION)
            if in_margin and len(ln.text.split()) <= 15:
                key = _normalise_for_repeat(ln.text)
                if key not in seen_on_page:
                    counts[key] += 1
                    seen_on_page.add(key)
    threshold = max(2, int(num_pages * REPEAT_FRACTION))
    return {k for k, c in counts.items() if c >= threshold}


# ── Structure reconstruction ──────────────────────────────────────────────────


def _is_heading(block: _Block, body_size: float) -> bool:
    text = block.text
    words = text.split()
    if not words or len(words) > HEADING_MAX_WORDS or len(block.lines) > 3:
        return False
    if _CAPTION.match(text) or _is_sentence_like(text) or not re.search(r"[A-Za-z]{3}", text):
        return False
    if looks_like_known_heading(text):
        return True
    all_bold = all(ln.bold for ln in block.lines)
    larger = block.mean_size >= body_size * 1.12
    numbered = numbering_depth(text) > 0 or bool(re.match(r"^[IVX]+\.\s+[A-Z]", text))
    if (all_bold and block.mean_size >= body_size * 0.95) or (larger and len(words) <= 10):
        return True
    return numbered and all_bold


def parse_pdf(pdf_path: Path, doc_id: str | None = None) -> ParsedDocument:
    """Parse a PDF into structured paragraphs. Raises :class:`PDFParseError`."""
    doc_id = doc_id or pdf_path.stem
    blocks, num_pages = _read_blocks(pdf_path)
    if not blocks:
        raise PDFParseError(f"No extractable text in {pdf_path.name} (scanned PDF?)")

    body_size = _body_font_size(blocks)
    repeated = _repeated_margin_lines(blocks, num_pages)

    parsed = ParsedDocument(doc_id=doc_id, num_pages=num_pages)
    heading, subheading = "", ""
    section = Section.FRONT_MATTER
    pending_number = ""

    for block in blocks:
        # Drop running headers/footers and bare page numbers line by line.
        kept = [
            ln
            for ln in block.lines
            if _normalise_for_repeat(ln.text) not in repeated and not _PAGE_NUMBER.match(ln.text.strip())
        ]
        if not kept:
            continue
        block = _Block(page=block.page, lines=kept, page_height=block.page_height)
        text = clean_text(block.text)
        if not text:
            continue

        # "1" on its own line followed by "Introduction": remember the number.
        if re.fullmatch(r"\d+(\.\d+)*\.?|[IVX]+\.?", text) and all(ln.bold for ln in block.lines):
            pending_number = text.rstrip(".")
            continue
        if pending_number and len(block.lines) >= 2 and re.fullmatch(r"\d+(\.\d+)*\.?", block.lines[0].text.strip()):
            pending_number = ""

        # Number and title in one block on separate lines ("1" / "Introduction").
        first = block.lines[0].text.strip()
        if len(block.lines) == 2 and re.fullmatch(r"\d+(\.\d+)*\.?", first) and all(ln.bold for ln in block.lines):
            text = f"{first.rstrip('.')} {block.lines[1].text.strip()}"
            block = _Block(page=block.page, lines=[block.lines[1]], page_height=block.page_height)
            if _is_heading(block, body_size):
                block = _Block(page=block.page, lines=[_Line(text, block.lines[0].size, True, 0, 0)], page_height=0)

        is_caption = bool(_CAPTION.match(text))
        if not is_caption and block.mean_size < body_size * SMALL_FONT_RATIO and section != Section.REFERENCES:
            continue  # figure internals, footnotes

        if not is_caption and _is_heading(block, body_size):
            title = f"{pending_number} {text}".strip() if pending_number else text
            pending_number = ""
            canonical = classify_heading(title)
            depth = numbering_depth(title)
            is_sub = depth >= 2 or (canonical == Section.OTHER and section not in (Section.FRONT_MATTER,) and depth == 0 and not looks_like_known_heading(title))
            if is_sub and heading:
                subheading = strip_numbering(title)
            else:
                heading, subheading = strip_numbering(title), ""
                if canonical != Section.OTHER:
                    section = canonical
                elif section in (Section.FRONT_MATTER, Section.ABSTRACT):
                    section = Section.OTHER
                elif section == Section.REFERENCES:
                    section = Section.APPENDIX  # content after the bibliography
            parsed.headings.append({"text": title, "page": block.page, "section": section.value, "level": 2 if subheading else 1})
            continue
        pending_number = ""

        # Inline "Abstract—Accurate uncertainty ..." (IEEE style).
        if section == Section.FRONT_MATTER and _INLINE_ABSTRACT.match(text):
            section, heading, subheading = Section.ABSTRACT, "Abstract", ""
            text = _INLINE_ABSTRACT.sub("", text)

        parsed.paragraphs.append(
            Paragraph(
                text=text,
                page=block.page,
                heading=heading,
                subheading=subheading,
                section=section.value,
                kind="caption" if is_caption else "body",
            )
        )

    if len(parsed.text) < MIN_TEXT_LENGTH:
        raise PDFParseError(f"Too little text in {pdf_path.name} ({len(parsed.text)} chars)")
    return parsed


def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """v0.1-compatible helper: return the document text or ``None`` on failure."""
    try:
        return parse_pdf(pdf_path).text
    except PDFParseError as exc:
        logger.error("Failed to extract text from %s: %s", pdf_path.name, exc)
        return None


# ── Batch CLI ─────────────────────────────────────────────────────────────────


def structured_path(doc_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._\-]", "_", doc_id)
    return config.STRUCTURED_DIR / f"{safe}.json"


def find_pdf(doc_id: str, paper: dict[str, Any]) -> Path | None:
    """Locate the PDF for a paper inside ``data/raw`` (never outside it)."""
    raw_dir = config.RAW_DIR.resolve()
    candidates = [config.RAW_DIR / f"{doc_id}.pdf"]
    if paper.get("pdf_path"):
        candidates.append(config.PROJECT_ROOT / paper["pdf_path"])
    candidates.extend(sorted(config.RAW_DIR.glob(f"{doc_id}_*.pdf")))
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file() and raw_dir in resolved.parents:
            return resolved
    return None


def parse_all(force: bool = False) -> dict[str, int]:
    """Parse every paper listed in ``metadata.json`` that has a PDF in ``data/raw``."""
    config.setup_dirs()
    metadata = load_metadata(config.META_FILE)
    stats = Counter[str]()
    for doc_id, paper in metadata.items():
        out = structured_path(doc_id)
        if out.exists() and not force:
            stats["skipped"] += 1
            continue
        pdf = find_pdf(doc_id, paper)
        if pdf is None:
            stats["missing_pdf"] += 1
            continue
        try:
            parsed = parse_pdf(pdf, doc_id=doc_id)
        except PDFParseError as exc:
            logger.warning("parse failed for %s: %s", doc_id, exc)
            stats["failed"] += 1
            continue
        save_metadata(parsed.to_dict(), out)
        stats["parsed"] += 1
    logger.info("PDF parsing finished: %s", dict(stats))
    return dict(stats)


def load_structured(doc_id: str) -> ParsedDocument | None:
    path = structured_path(doc_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return ParsedDocument.from_dict(json.load(fh))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Parse PDFs in data/raw into structured JSON")
    ap.add_argument("--force", action="store_true", help="re-parse already parsed papers")
    print(parse_all(force=ap.parse_args().force))
