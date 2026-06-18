"""
chunker.py
-----------
Reads cleaned .txt files from data/processed/texts/,
splits them into overlapping chunks, and saves the chunks
to data/processed/chunks/ as JSON files.

One JSON file per paper. Each JSON contains a list of chunk dicts.

Usage:
    python src/ingestion/chunker.py
"""

import json
import logging
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

import src.config as config

TEXT_DIR   = config.TEXT_DIR
CHUNK_DIR  = config.CHUNK_DIR
META_FILE  = config.META_FILE
CHUNK_LOG  = config.PROCESSED_DIR / "chunk_log.json"

CHUNK_SIZE    = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
MIN_CHUNK_SIZE = config.MIN_CHUNK_SIZE

# Improved regex for sentence splitting (handles decimals, common abbreviations)
# Looks for sentence endings (.!?) followed by space and a capital letter,
# but tries to ignore common cases like e.g., i.e., et al., etc.
SENTENCE_PATTERN = re.compile(r'(?<!\b(?:e\.g|i\.e|et\sal|vs|etc|cf|Dr|Mr|Mrs|Ms|Prof|Inc|Ltd))\.\s+(?=[A-Z])|(?<=[!?])\s+(?=[A-Z])')

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_dirs():
    """Create output directories if they don't exist."""
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Chunk output directory ready: {CHUNK_DIR}")


def load_metadata() -> dict:
    """Load paper metadata from JSON file."""
    if not META_FILE.exists():
        logger.error(f"Metadata file not found: {META_FILE}")
        logger.error("Run arxiv_downloader.py and pdf_parser.py first.")
        return {}
    with open(META_FILE, "r") as f:
        return json.load(f)


def save_metadata(metadata: dict):
    """Save updated metadata back to JSON file."""
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def load_chunk_log() -> dict:
    """Load chunk log to track which papers have already been chunked."""
    if CHUNK_LOG.exists():
        with open(CHUNK_LOG, "r") as f:
            return json.load(f)
    return {}


def save_chunk_log(chunk_log: dict):
    """Save chunk log to disk."""
    with open(CHUNK_LOG, "w") as f:
        json.dump(chunk_log, f, indent=2)


# ── Sentence Splitting ────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.
    Tries to respect sentence boundaries rather than cutting mid-sentence.

    Args:
        text: Cleaned paper text

    Returns:
        List of sentence strings
    """
    # Split on period/exclamation/question mark followed by space and capital letter
    # This avoids splitting on "e.g." or "et al." or "Fig. 3"
    sentences = SENTENCE_PATTERN.split(text)

    # Clean up each sentence
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# ── Core Chunking Logic ───────────────────────────────────────────────────────

def chunk_text(text: str, arxiv_id: str) -> list[dict]:
    """
    Split a paper's text into overlapping word-based chunks.
    Respects sentence boundaries where possible.

    Strategy:
        1. Split text into sentences
        2. Accumulate sentences into a chunk until word count >= CHUNK_SIZE
        3. Start next chunk overlapping back by CHUNK_OVERLAP words
        4. Discard chunks below MIN_CHUNK_SIZE

    Args:
        text: Full cleaned text of a paper
        arxiv_id: Paper ID for chunk metadata

    Returns:
        List of chunk dicts, each with text, metadata, and position info
    """
    sentences = split_into_sentences(text)

    if not sentences:
        logger.warning(f"No sentences found for {arxiv_id}")
        return []

    chunks = []
    chunk_index = 0

    # We work at the word level for size tracking
    # but accumulate at the sentence level for clean boundaries
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        current_sentences.append(sentence)
        current_word_count += sentence_words

        # Once we hit the target chunk size, save this chunk
        if current_word_count >= CHUNK_SIZE:
            chunk_text_str = " ".join(current_sentences)

            # Only keep chunks above minimum size
            if len(chunk_text_str.split()) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "chunk_id": f"{arxiv_id}_chunk_{chunk_index}",
                    "arxiv_id": arxiv_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text_str,
                    "word_count": len(chunk_text_str.split()),
                    "char_count": len(chunk_text_str),
                })
                chunk_index += 1

            # Overlap: keep last CHUNK_OVERLAP words worth of sentences
            # for the next chunk so context isn't lost at boundaries
            overlap_sentences = []
            overlap_word_count = 0

            # Walk backwards through current sentences to build overlap
            for sent in reversed(current_sentences):
                sent_words = len(sent.split())
                overlap_sentences.insert(0, sent)
                overlap_word_count += sent_words
                
                # Break ONLY AFTER we have met or exceeded the overlap quota
                if overlap_word_count >= CHUNK_OVERLAP:
                    break

            # Start next chunk from the overlap sentences
            current_sentences = overlap_sentences
            current_word_count = overlap_word_count

    # Don't forget the last chunk (remaining sentences after the loop)
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        if len(chunk_text_str.split()) >= MIN_CHUNK_SIZE:
            chunks.append({
                "chunk_id": f"{arxiv_id}_chunk_{chunk_index}",
                "arxiv_id": arxiv_id,
                "chunk_index": chunk_index,
                "text": chunk_text_str,
                "word_count": len(chunk_text_str.split()),
                "char_count": len(chunk_text_str),
            })

    return chunks


# ── File I/O ──────────────────────────────────────────────────────────────────

def save_chunks(arxiv_id: str, chunks: list[dict]):
    """Save a paper's chunks to a JSON file in CHUNK_DIR."""
    output_path = CHUNK_DIR / f"{arxiv_id}_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    return output_path


def load_text(txt_path: Path) -> str | None:
    """Read a cleaned text file from disk."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read {txt_path}: {e}")
        return None


# ── Orchestrator ──────────────────────────────────────────────────────────────

def chunk_all_papers(metadata: dict, chunk_log: dict) -> tuple[dict, dict]:
    """
    Loop over all parsed papers and chunk their text files.

    Args:
        metadata: Paper metadata dict
        chunk_log: Tracks chunking status per paper

    Returns:
        Updated (metadata, chunk_log) tuple
    """
    total = len(metadata)
    chunked = 0
    skipped = 0
    failed = 0
    total_chunks = 0

    logger.info(f"Found {total} papers in metadata. Starting chunking...")

    for arxiv_id, paper in metadata.items():

        # Skip papers that haven't been parsed yet (no txt_path in metadata)
        if "txt_path" not in paper:
            logger.warning(f"[SKIP] No parsed text for: {arxiv_id} — run pdf_parser.py first")
            skipped += 1
            continue

        # Skip already chunked papers
        if arxiv_id in chunk_log and chunk_log[arxiv_id]["status"] == "success":
            logger.info(f"[SKIP] Already chunked: {arxiv_id}")
            skipped += 1
            continue

        txt_path = Path(paper["txt_path"])

        # Skip if text file doesn't exist on disk
        if not txt_path.exists():
            logger.warning(f"[MISS] Text file not found: {txt_path}")
            chunk_log[arxiv_id] = {"status": "missing", "reason": "txt file not found"}
            failed += 1
            continue

        logger.info(f"[CHUNK] {arxiv_id} — {paper['title'][:60]}...")

        # Load the cleaned text
        text = load_text(txt_path)
        if text is None:
            chunk_log[arxiv_id] = {"status": "failed", "reason": "could not read txt file"}
            failed += 1
            continue

        # Chunk the text
        chunks = chunk_text(text, arxiv_id)

        if not chunks:
            logger.warning(f"[WARN] No chunks produced for {arxiv_id}")
            chunk_log[arxiv_id] = {"status": "failed", "reason": "no chunks produced"}
            failed += 1
            continue

        # Save chunks to disk
        output_path = save_chunks(arxiv_id, chunks)

        # Update metadata
        metadata[arxiv_id]["chunks_path"] = str(output_path)
        metadata[arxiv_id]["num_chunks"] = len(chunks)

        # Update chunk log
        chunk_log[arxiv_id] = {
            "status": "success",
            "chunks_path": str(output_path),
            "num_chunks": len(chunks),
        }

        chunked += 1
        total_chunks += len(chunks)
        logger.info(f"[OK]   {arxiv_id} — {len(chunks)} chunks")

        # Save progress after every paper
        save_metadata(metadata)
        save_chunk_log(chunk_log)

    logger.info(f"\n{'='*60}")
    logger.info(f"Chunking complete.")
    logger.info(f"  Chunked      : {chunked} papers")
    logger.info(f"  Skipped      : {skipped} papers")
    logger.info(f"  Failed       : {failed} papers")
    logger.info(f"  Total chunks : {total_chunks}")
    logger.info(f"  Avg per paper: {total_chunks // max(chunked, 1)} chunks")
    logger.info(f"  Output dir   : {CHUNK_DIR}")
    logger.info(f"{'='*60}")

    return metadata, chunk_log


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    logger.info("Starting chunker...")
    setup_dirs()
    metadata = load_metadata()

    if not metadata:
        return

    chunk_log = load_chunk_log()
    metadata, chunk_log = chunk_all_papers(metadata, chunk_log)
    save_metadata(metadata)
    save_chunk_log(chunk_log)
    logger.info("Done.")


if __name__ == "__main__":
    main()
