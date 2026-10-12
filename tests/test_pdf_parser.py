"""Structure-aware PDF parsing on synthetic academic-style PDFs."""

from __future__ import annotations

import pytest

from src.ingestion.pdf_parser import PDFParseError, extract_text_from_pdf, join_lines, parse_pdf
from tests.conftest import TextItem, make_pdf

BODY = (
    "Retrieval augmented generation grounds language models in external documents and "
    "reduces hallucination on knowledge intensive tasks across many benchmarks."
)


def paper_items() -> list[TextItem]:
    """Three pages, two columns, running header, page numbers, references, appendix."""
    items: list[TextItem] = []
    for page in (1, 2, 3):
        items.append(TextItem("Preprint. Under review at ICLR 2025", page, x=200, y=30, size=8))
        items.append(TextItem(str(page), page, x=300, y=770, size=10))
    items += [
        TextItem("A Study of Corrective Retrieval", 1, x=150, y=80, size=16, bold=True),
        TextItem("Jane Doe, John Roe", 1, x=220, y=105, size=10),
        TextItem("Abstract", 1, x=72, y=150, size=12, bold=True),
        TextItem("We study corrective retrieval for question answering.\n" + BODY, 1, x=72, y=170),
        TextItem("1 Introduction", 1, x=72, y=260, size=12, bold=True),
        # Left column paragraph, then right column paragraph at an *earlier* y.
        TextItem("LEFTCOL the language model abili-\nties are limited by stale knowledge.\n" + BODY[:60], 1, x=72, y=290),
        TextItem("RIGHTCOL retrieval quality decides answer quality.\n" + BODY[:60], 1, x=330, y=280),
        TextItem("2 Method", 2, x=72, y=80, size=12, bold=True),
        TextItem("Our evaluator scores each passage on page two.\n" + BODY, 2, x=72, y=110),
        TextItem("References", 3, x=72, y=80, size=12, bold=True),
        TextItem("[1] A. Author. A paper about retrieval. arXiv preprint arXiv:2401.00001, 2024.", 3, x=72, y=110),
        TextItem("A Additional Experiments", 3, x=72, y=200, size=12, bold=True),
        TextItem("The appendix reports extra ablations on page three.\n" + BODY, 3, x=72, y=230),
    ]
    return items


@pytest.fixture
def paper_pdf(tmp_path):
    return make_pdf(tmp_path / "paper.pdf", paper_items())


def section_of(doc, needle: str) -> str:
    return next(p.section for p in doc.paragraphs if needle in p.text)


def test_sections_pages_and_headings(paper_pdf):
    doc = parse_pdf(paper_pdf, doc_id="2401.00001")
    assert doc.num_pages == 3
    assert section_of(doc, "We study corrective retrieval") == "abstract"
    assert section_of(doc, "LEFTCOL") == "introduction"
    assert section_of(doc, "Our evaluator") == "method"
    assert section_of(doc, "A paper about retrieval") == "references"
    # First heading after the bibliography is an appendix, not "other".
    assert section_of(doc, "extra ablations") == "appendix"
    pages = {p.text.split()[0]: p.page for p in doc.paragraphs}
    assert pages["Our"] == 2 and pages["The"] == 3
    assert [h["text"] for h in doc.headings][:3] == ["Abstract", "1 Introduction", "2 Method"]


def test_title_is_front_matter_not_a_section(paper_pdf):
    doc = parse_pdf(paper_pdf)
    # "A Study of Corrective Retrieval" contains no section keyword, and must
    # not open a section; the author line stays in front matter.
    assert section_of(doc, "Jane Doe") == "front_matter"
    assert all(h["text"] != "A Study of Corrective Retrieval" for h in doc.headings)


def test_running_headers_and_page_numbers_removed(paper_pdf):
    text = parse_pdf(paper_pdf).text
    assert "Preprint. Under review" not in text
    assert all(p.text.strip() not in {"1", "2", "3"} for p in parse_pdf(paper_pdf).paragraphs)


def test_content_stream_reading_order_keeps_columns_apart(paper_pdf):
    text = parse_pdf(paper_pdf).text
    # Sorting blocks by (y, x) — the v0.1 approach — would put RIGHTCOL first
    # because it starts higher on the page.
    assert text.index("LEFTCOL") < text.index("RIGHTCOL")


def test_dehyphenation_across_lines(paper_pdf):
    text = parse_pdf(paper_pdf).text
    assert "abilities are limited" in text
    assert "abili-" not in text


def test_join_lines_keeps_real_hyphens():
    assert join_lines(["state-of-", "the-art models"]) == "state-of-the-art models"
    assert join_lines(["pre-", "Training"]) == "pre- Training"  # capitalised: not a split word
    assert join_lines(["a range of 1 -", "5"]) == "a range of 1 - 5"


def test_run_in_heading_split_from_paragraph(tmp_path):
    items = [
        TextItem("A Paper Title Here", 1, x=150, y=80, size=16, bold=True),
        TextItem("Abstract", 1, x=72, y=120, size=12, bold=True),
        TextItem(BODY + "\n" + BODY, 1, x=72, y=140),
        # Heading and its paragraph inside one text block (IEEE style).
        TextItem("I. INTRODUCTION\nChatbots answer questions in natural language.\n" + BODY + "\n" + BODY, 1, x=72, y=220),
    ]
    doc = parse_pdf(make_pdf(tmp_path / "ieee.pdf", items))
    assert section_of(doc, "Chatbots answer") == "introduction"
    assert any(h["text"] == "I. INTRODUCTION" for h in doc.headings)
    assert not any(p.text.startswith("I. INTRODUCTION") for p in doc.paragraphs)


def test_long_front_matter_without_headings_is_kept_as_body(tmp_path):
    long_para = "\n".join([BODY] * 3)  # > 50 words, no heading anywhere
    items = [TextItem(long_para, 1, x=72, y=120), TextItem(long_para, 2, x=72, y=120)]
    doc = parse_pdf(make_pdf(tmp_path / "noheadings.pdf", items))
    assert {p.section for p in doc.paragraphs} == {"other"}


def test_non_latin_text_is_not_dropped(tmp_path):
    import pymupdf

    path = tmp_path / "korean.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    text = "이 연구에서는 대규모 언어 모델을 활용하여 부적절한 발언을 탐지한다. " * 20
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 700), text, fontname="korea", fontsize=10)
    doc.save(str(path))
    doc.close()
    parsed = parse_pdf(path)
    assert "연구" in parsed.text


def test_figure_fragments_dropped(tmp_path):
    items = [
        TextItem("Abstract", 1, x=72, y=100, size=12, bold=True),
        TextItem("\n".join([BODY] * 6), 1, x=72, y=120),
        TextItem("1-3", 1, x=72, y=400),
        TextItem("4-5", 1, x=150, y=400),
    ]
    doc = parse_pdf(make_pdf(tmp_path / "fig.pdf", items))
    assert all(p.text not in {"1-3", "4-5"} for p in doc.paragraphs)


@pytest.mark.parametrize(
    "content, message",
    [(b"not a pdf at all", "Not a PDF"), (b"%PDF-1.4\n%%EOF garbage", "Cannot open")],
)
def test_malformed_inputs_raise_parse_error(tmp_path, content, message):
    path = tmp_path / "bad.pdf"
    path.write_bytes(content)
    with pytest.raises(PDFParseError, match=message):
        parse_pdf(path)


def test_empty_and_scanned_pdfs_raise(tmp_path):
    import pymupdf

    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    with pytest.raises(PDFParseError, match="No extractable text"):
        parse_pdf(path)
    assert extract_text_from_pdf(path) is None  # v0.1 helper returns None instead of raising


def test_encrypted_pdf_raises(tmp_path):
    import pymupdf

    path = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), BODY)
    doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    with pytest.raises(PDFParseError, match="Encrypted"):
        parse_pdf(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(PDFParseError, match="Not a file"):
        parse_pdf(tmp_path / "nope.pdf")


def test_small_lowercase_figure_label_does_not_open_references(tmp_path):
    items = [
        TextItem("Abstract", 1, x=72, y=100, size=12, bold=True),
        TextItem("\n".join([BODY] * 3), 1, x=72, y=120),
        TextItem("1 Introduction", 1, x=72, y=200, size=12, bold=True),
        # A figure label in 7 pt that happens to read "references" (arXiv 1606.05112).
        TextItem("references\nelements of", 1, x=300, y=260, size=7),
        TextItem("Body text after the figure continues the introduction.\n" + BODY, 1, x=72, y=320),
        TextItem("References", 2, x=72, y=80, size=12, bold=True),
        TextItem("[1] A. Author. A paper. arXiv preprint, 2024.", 2, x=72, y=110),
    ]
    doc = parse_pdf(make_pdf(tmp_path / "figlabel.pdf", items))
    assert section_of(doc, "Body text after the figure") == "introduction"
    assert section_of(doc, "A paper. arXiv") == "references"


def test_structured_cache_is_versioned(tmp_path, monkeypatch, paper_pdf):
    import json

    import src.config as config
    from src.ingestion import pdf_parser

    monkeypatch.setattr(config, "STRUCTURED_DIR", tmp_path / "structured")
    doc = parse_pdf(paper_pdf, doc_id="2401.00001")
    path = pdf_parser.structured_path("2401.00001")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc.to_dict()))
    assert pdf_parser.load_structured("2401.00001").paragraphs == doc.paragraphs
    stale = dict(doc.to_dict(), parser_version=pdf_parser.PARSER_VERSION - 1)
    path.write_text(json.dumps(stale))
    assert pdf_parser.load_structured("2401.00001") is None  # forces a re-parse
    path.write_text("{corrupt")
    assert pdf_parser.load_structured("2401.00001") is None
