"""Section-aware chunking: sizes, boundaries, provenance, determinism."""

from __future__ import annotations

from src.config import Settings
from src.ingestion.chunker import (
    _Unit,
    chunk_parsed_document,
    chunk_text,
    embedding_text,
    pack_units,
    split_into_sentences,
)
from src.ingestion.pdf_parser import Paragraph, ParsedDocument

SETTINGS = Settings(chunk_max_words=40, chunk_overlap_words=10, chunk_min_words=8)
PAPER = {"title": "Test Paper", "authors": ["A"], "published": "2024-01-01", "url": "http://x", "abstract": "Meta abstract text."}


def sentence(i: int, words: int = 9) -> str:
    return f"Sentence {i} " + " ".join(f"w{i}x{j}" for j in range(words - 3)) + " end."


def para(text: str, section: str, page: int, heading: str = "H", sub: str = "") -> Paragraph:
    return Paragraph(text=text, page=page, heading=heading, subheading=sub, section=section)


def test_sentence_split_respects_abbreviations():
    text = "Yan et al. (2024) propose CRAG. See Fig. 2 and Eq. 3 for details. It works, e.g. on PopQA. Done."
    assert split_into_sentences(text) == [
        "Yan et al. (2024) propose CRAG.",
        "See Fig. 2 and Eq. 3 for details.",
        "It works, e.g. on PopQA.",
        "Done.",
    ]


def test_pack_units_respects_max_and_overlap_without_losing_text():
    units = [_Unit(sentence(i), page=1 + i // 5) for i in range(20)]
    windows = pack_units(units, max_words=40, overlap_words=10, min_words=8)
    assert all(sum(u.words for u in w) <= 40 for w in windows)
    covered = {u.text for w in windows for u in w}
    assert covered == {u.text for u in units}  # nothing dropped
    for prev, nxt in zip(windows, windows[1:]):
        assert prev[-1].text == nxt[0].text  # whole-sentence overlap


def test_short_tail_is_merged_not_dropped():
    units = [_Unit(sentence(i), 1) for i in range(4)] + [_Unit("Tail end.", 1)]
    windows = pack_units(units, max_words=36, overlap_words=0, min_words=8)
    assert windows[-1][-1].text == "Tail end."
    assert len(windows) == 1 or sum(u.words for u in windows[-1]) >= 8


def test_overlong_sentence_is_split_into_windows():
    text = " ".join(f"tok{i}" for i in range(100))  # a table row with no punctuation
    doc = ParsedDocument("d1", 1, [para(text, "results", 1)])
    chunks = chunk_parsed_document(doc, PAPER, SETTINGS)
    assert all(c.word_count <= 40 for c in chunks if c.source_format == "pdf")
    assert " ".join(c.text for c in chunks if c.source_format == "pdf").split() == text.split()


def make_doc() -> ParsedDocument:
    paras = [
        para("Front matter author line.", "front_matter", 1, ""),
        para(" ".join(sentence(i) for i in range(3)), "abstract", 1, "Abstract"),
        para(" ".join(sentence(i) for i in range(3, 12)), "method", 2, "Method"),
        para(" ".join(sentence(i) for i in range(12, 15)), "method", 3, "Method", "Training"),
        para(" ".join(sentence(i) for i in range(15, 20)), "results", 4, "Results"),
        para("[1] A. Author. Some paper. arXiv preprint, 2023. " * 5, "references", 5, "References"),
        para(" ".join(sentence(i) for i in range(20, 24)), "appendix", 6, "A Details"),
    ]
    return ParsedDocument("2401.00001", 6, paras)


def test_chunks_never_cross_sections_and_carry_provenance():
    chunks = chunk_parsed_document(make_doc(), PAPER, SETTINGS)
    by_section = {}
    for c in chunks:
        by_section.setdefault(c.section, []).append(c)
        assert c.page_start is not None and c.page_start <= c.page_end
        assert c.title == "Test Paper" and c.url == "http://x"
    assert set(by_section) == {"abstract", "method", "results", "appendix"}
    assert all("Sentence 1 " not in c.text for c in by_section["results"])  # abstract text stays out
    method_pages = {(c.page_start, c.page_end) for c in by_section["method"]}
    assert min(p[0] for p in method_pages) == 2 and max(p[1] for p in method_pages) == 3
    assert any("Training" in c.heading for c in by_section["method"])


def test_references_and_front_matter_excluded_by_default():
    chunks = chunk_parsed_document(make_doc(), PAPER, SETTINGS)
    assert not any("arXiv preprint" in c.text for c in chunks)
    assert not any("Front matter author" in c.text for c in chunks)
    with_refs = chunk_parsed_document(make_doc(), PAPER, Settings(**{**SETTINGS.model_dump(), "index_references": True}))
    assert any(c.section == "references" for c in with_refs)


def test_chunk_ids_are_stable_and_deterministic():
    a = [c.to_dict() for c in chunk_parsed_document(make_doc(), PAPER, SETTINGS)]
    b = [c.to_dict() for c in chunk_parsed_document(make_doc(), PAPER, SETTINGS)]
    assert a == b
    assert [c["chunk_id"] for c in a] == [f"2401.00001_chunk_{i}" for i in range(len(a))]
    assert len({c["content_hash"] for c in a}) == len(a)


def test_metadata_abstract_used_only_when_parse_found_none():
    doc = make_doc()
    assert not any(c.source_format == "metadata" for c in chunk_parsed_document(doc, PAPER, SETTINGS))
    doc.paragraphs = [p for p in doc.paragraphs if p.section != "abstract"]
    chunks = chunk_parsed_document(doc, PAPER, SETTINGS)
    assert chunks[0].source_format == "metadata" and chunks[0].text == "Meta abstract text."
    assert chunks[0].page_start is None  # no invented page number


def test_legacy_text_dehyphenates_and_drops_bibliography():
    lines = (
        ["Abstract", "We show the abili-", "ties of retrieval models are limited."]
        + ["1", "Introduction"]
        + [sentence(i) for i in range(10)]
        + ["References", "[1] A. Author. Paper. arXiv preprint, 2023.", "[2] B. Author. Paper. 2022."]
        + ["Appendix", "Extra ablation results are reported here."]
    )
    chunks = chunk_text("\n".join(lines), "2401.00001", PAPER, SETTINGS)
    text = " ".join(c["text"] for c in chunks)
    assert "abilities of retrieval" in text
    assert "arXiv preprint" not in text
    assert "Extra ablation results" in text
    assert {c["section"] for c in chunks} >= {"abstract", "introduction", "appendix"}
    assert all(c["page_start"] is None for c in chunks)  # legacy text has no pages


def test_empty_inputs_produce_no_chunks():
    assert chunk_text("", "x", {}, SETTINGS) == []
    assert chunk_parsed_document(ParsedDocument("x", 1, []), {}, SETTINGS) == []


def test_embedding_text_adds_context_header():
    chunk = {"text": "Our method wins.", "title": "CRAG", "heading": "Results"}
    assert embedding_text(chunk) == "CRAG | Results\nOur method wins."
    assert embedding_text(chunk, contextual_headers=False) == "Our method wins."
