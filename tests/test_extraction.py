"""v0.1-compatible text extraction helpers (``extract_text_from_pdf``, ``clean_text``).

The previous version of this file re-implemented an extraction routine inline
and tested that copy, so it could not catch regressions in the project code.
"""

from __future__ import annotations

from src.ingestion.pdf_parser import clean_text, extract_text_from_pdf
from tests.conftest import TextItem, make_pdf

LINE = "This is a test block of text on the page with enough words to count as body text."


def test_extract_text_from_pdf_returns_body_text(tmp_path):
    pdf = make_pdf(tmp_path / "test.pdf", [TextItem("\n".join([LINE] * 8), 1, y=100)])
    text = extract_text_from_pdf(pdf)
    assert text is not None and text.count("test block of text") == 8


def test_extract_text_from_pdf_returns_none_for_invalid_file(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf")
    assert extract_text_from_pdf(bad) is None


def test_clean_text_normalises_whitespace_and_strips_emails():
    raw = "Hello\x0cworld  \x07 contact me@example.com\n\n\n\n----\nEnd"
    assert clean_text(raw) == "Hello world contact\n\nEnd"
