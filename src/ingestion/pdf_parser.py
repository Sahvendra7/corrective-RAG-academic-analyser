"""
pdf_parser.py
--------------
Reads PDFs from data/raw/, extracts and cleans text using PyMuPDF,
and saves individual .txt files to data/processed/texts/.
Also updates metadata.json with parsing status.

Usage:
    python src/ingestion/pdf_parser.py
"""

import fitz  # PyMuPDF
import json
import re
import logging
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
import src.config as config

RAW_DIR = config.RAW_DIR
TEXT_DIR = config.TEXT_DIR
META_FILE = config.META_FILE
PARSE_LOG = config.PROCESSED_DIR / "parse_log.json"

MIN_TEXT_LENGTH = 500  # Skip papers with less than 500 chars (likely scanned/empty)

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_dirs():
    """Create output directories if they don't exist."""
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Text output directory ready: {TEXT_DIR}")




# ── Text Extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """
    Extract raw text from a PDF using PyMuPDF block-level layout analysis.
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            pages_text = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get the page dimensions to calculate thresholds
                page_rect = page.rect
                header_threshold = page_rect.height * 0.08  # Top 8% of page
                footer_threshold = page_rect.height * 0.92  # Bottom 8% of page

                # Extract text organized by physical blocks
                blocks = page.get_text("blocks")
                
                # Sort blocks primarily by y0 (top to bottom), then by x0 (left to right)
                # This helps preserve reading order in complex layouts
                blocks.sort(key=lambda b: (b[1], b[0]))
                
                page_content = []
                for b in blocks:
                    # b[0]=x0, b[1]=y0, b[2]=x1, b[3]=y1, b[4]=text, b[5]=block_no, b[6]=block_type
                    y0 = b[1]
                    y1 = b[3]
                    block_type = b[6]

                    # Skip non-text blocks (like images, where type == 1)
                    if block_type != 0:
                        continue

                    # Filter out headers and footers based on y-coordinates
                    if y0 < header_threshold or y1 > footer_threshold:
                        continue
                    
                    # Clean up the block text and append
                    text = b[4].strip()
                    if text:
                        page_content.append(text)

                pages_text.append("\n\n".join(page_content))

            return "\n\n".join(pages_text)

    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
        return None


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(raw_text: str) -> str:
    """
    Clean extracted PDF text by removing noise and normalizing whitespace.

    Args:
        raw_text: Raw text extracted from PDF

    Returns:
        Cleaned text string
    """
    text = raw_text

    # Remove form feed characters (page breaks in PDFs)
    text = text.replace("\x0c", " ")

    # Remove other common non-printable/control characters
    text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f]", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove standalone numbers (page numbers, figure numbers etc.)
    #text = re.sub(r"(?<!\w)\d+(?!\w)", "", text)

    # Remove lines that are just dashes or underscores (dividers)
    text = re.sub(r"^[-_=]{3,}$", "", text, flags=re.MULTILINE)

    # Normalize multiple spaces into a single space
    text = re.sub(r" {2,}", " ", text)

    # Normalize multiple newlines into a maximum of two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty lines at start/end, keep paragraph breaks
    text = "\n".join(lines)

    # Final strip
    text = text.strip()

    return text


# ── Quality Check ─────────────────────────────────────────────────────────────

def is_valid_text(text: str, arxiv_id: str) -> bool:
    """
    Check if extracted text meets minimum quality threshold.

    Args:
        text: Cleaned text to validate
        arxiv_id: Paper ID for logging

    Returns:
        True if text is usable, False otherwise
    """
    if len(text) < MIN_TEXT_LENGTH:
        logger.warning(
            f"[SKIP] {arxiv_id} — text too short ({len(text)} chars). "
            f"Likely a scanned PDF with no extractable text."
        )
        return False
    return True


# ── Core Parsing Logic ────────────────────────────────────────────────────────

def parse_papers(metadata: dict, parse_log: dict) -> tuple[dict, dict]:
    """
    Parse all PDFs in metadata that haven't been parsed yet.

    Args:
        metadata: Paper metadata dictionary from arxiv_downloader
        parse_log: Dictionary tracking parse status per paper

    Returns:
        Updated (metadata, parse_log) tuple
    """
    total = len(metadata)
    parsed = 0
    skipped = 0
    failed = 0

    logger.info(f"Found {total} papers in metadata. Starting parsing...")

    for arxiv_id, paper in metadata.items():

        # Skip already parsed papers
        if arxiv_id in parse_log and parse_log[arxiv_id]["status"] == "success":
            logger.info(f"[SKIP] Already parsed: {arxiv_id}")
            skipped += 1
            continue

        raw_pdf_path = paper.get("pdf_path", "")
        if not raw_pdf_path:
            logger.warning(f"[MISS] No pdf_path for: {arxiv_id}")
            parse_log[arxiv_id] = {"status": "missing", "reason": "No pdf_path in metadata"}
            failed += 1
            continue
        pdf_path = Path(raw_pdf_path)

        # Skip if PDF file doesn't exist on disk
        if not pdf_path.exists():
            logger.warning(f"[MISS] PDF not found on disk: {pdf_path.name}")
            parse_log[arxiv_id] = {"status": "missing", "reason": "PDF file not found"}
            failed += 1
            continue

        logger.info(f"[PARSE] {arxiv_id} — {paper['title'][:60]}...")

        # Step 1: Extract raw text
        raw_text = extract_text_from_pdf(pdf_path)
        if raw_text is None:
            parse_log[arxiv_id] = {"status": "failed", "reason": "extraction error"}
            failed += 1
            continue

        # Step 2: Clean the text
        clean = clean_text(raw_text)

        # Step 3: Quality check
        if not is_valid_text(clean, arxiv_id):
            parse_log[arxiv_id] = {
                "status": "failed",
                "reason": f"text too short ({len(clean)} chars)"
            }
            failed += 1
            continue

        # Step 4: Save cleaned text as .txt file
        txt_path = TEXT_DIR / f"{arxiv_id}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean)

        # Step 5: Update metadata with text path and char count
        metadata[arxiv_id]["txt_path"] = str(txt_path)
        metadata[arxiv_id]["char_count"] = len(clean)
        metadata[arxiv_id]["word_count"] = len(clean.split())

        # Step 6: Log success
        parse_log[arxiv_id] = {
            "status": "success",
            "txt_path": str(txt_path),
            "char_count": len(clean),
            "word_count": len(clean.split()),
        }

        parsed += 1
        logger.info(
            f"[OK]   {arxiv_id} — "
            f"{len(clean):,} chars, {len(clean.split()):,} words"
        )

        # Save progress after every paper
        save_metadata(metadata)
        save_parse_log(parse_log)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Parsing complete.")
    logger.info(f"  Parsed     : {parsed}")
    logger.info(f"  Skipped    : {skipped}")
    logger.info(f"  Failed     : {failed}")
    logger.info(f"  Text files : {TEXT_DIR}")
    logger.info(f"{'='*60}")

    return metadata, parse_log


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    logger.info("Starting PDF parser...")
    setup_dirs()
    metadata = load_metadata()

    if not metadata:
        return

    parse_log = load_parse_log()
    metadata, parse_log = parse_papers(metadata, parse_log)
    save_metadata(metadata)
    save_parse_log(parse_log)
    logger.info("Done.")


if __name__ == "__main__":
    main()
