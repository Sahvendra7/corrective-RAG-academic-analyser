"""
arxiv_downloader.py
--------------------
Downloads ML/AI papers from arXiv API and saves them as PDFs
into the data/raw/ directory with metadata saved as JSON.

Usage:
    python src/ingestion/arxiv_downloader.py
"""

# TODO: Add try/except block for 404s and fix pre-2007 ID parsing

import arxiv
import json
import time
import logging
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
import src.config as config

RAW_DIR = config.RAW_DIR
META_FILE = config.META_FILE

SEARCH_QUERIES = [
    "large language models",
    "retrieval augmented generation",
    "transformer architecture",
    "diffusion models",
    "reinforcement learning from human feedback",
    "graph neural networks",
    "vision transformers",
    "prompt engineering",
    "chain of thought reasoning",
    "neural architecture search",
]

PAPERS_PER_QUERY = 50       # 50 x 10 queries = 500 papers total
MAX_TOTAL_PAPERS = 500
DELAY_BETWEEN_DOWNLOADS = 2  # seconds — be polite to arXiv servers

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

from src.utils.metadata_utils import load_metadata, save_metadata

def setup_dirs():
    """Create required directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories ready: {RAW_DIR}, {META_FILE.parent}")


def sanitize_filename(title: str) -> str:
    """Convert paper title to a safe filename."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_")
    sanitized = "".join(c if c in keep else "_" for c in title)
    return sanitized[:80].strip().replace(" ", "_")


# ── Core Download Logic ───────────────────────────────────────────────────────

def download_papers(queries: list[str], papers_per_query: int, metadata: dict) -> dict:
    """
    Search arXiv for each query and download PDFs.

    Args:
        queries: List of search query strings
        papers_per_query: Max papers to fetch per query
        metadata: Existing metadata dict (to skip already-downloaded papers)

    Returns:
        Updated metadata dict
    """
    client = arxiv.Client(
        page_size=50,
        delay_seconds=3,
        num_retries=3,
    )

    total_downloaded = 0
    total_skipped = 0

    for query in queries:
        if total_downloaded >= MAX_TOTAL_PAPERS:
            logger.info(f"Reached max total papers ({MAX_TOTAL_PAPERS}). Stopping.")
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"Query: '{query}'")
        logger.info(f"{'='*60}")

        search = arxiv.Search(
            query=query,
            max_results=papers_per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in client.results(search):
            if total_downloaded >= MAX_TOTAL_PAPERS:
                break

            paper_id = result.entry_id.split("/")[-1]  # e.g. "2401.15884v2"
            arxiv_id = paper_id.split("v")[0]           # e.g. "2401.15884"
            # Sanitize pre-2007 IDs that contain '/' (e.g. "hep-th/9901001")
            arxiv_id = arxiv_id.replace("/", "_")

            # Skip if already downloaded
            if arxiv_id in metadata:
                logger.info(f"[SKIP] Already downloaded: {arxiv_id}")
                total_skipped += 1
                continue

            # Build safe filename
            safe_title = sanitize_filename(result.title)
            filename = f"{arxiv_id}_{safe_title}.pdf"
            pdf_path = RAW_DIR / filename

            try:
                # Download PDF
                logger.info(f"[DOWN] {arxiv_id} — {result.title[:60]}...")
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
                urllib.request.urlretrieve(pdf_url, str(pdf_path))

                # Validate downloaded file is actually a PDF
                with open(pdf_path, "rb") as pdf_check:
                    header = pdf_check.read(5)
                if header != b"%PDF-":
                    logger.warning(f"[WARN] Downloaded file is not a valid PDF: {arxiv_id}")
                    pdf_path.unlink(missing_ok=True)
                    continue

                # Save metadata entry
                metadata[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "abstract": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "categories": result.categories,
                    "pdf_path": str(pdf_path),
                    "query": query,
                    "url": result.entry_id,
                }

                total_downloaded += 1
                logger.info(f"[OK]   {total_downloaded}/{MAX_TOTAL_PAPERS} — {arxiv_id}")

                # Save metadata after every download (safe against crashes)
                save_metadata(metadata)

                # Polite delay
                time.sleep(DELAY_BETWEEN_DOWNLOADS)

            except Exception as e:
                logger.error(f"[ERR]  Failed to download {arxiv_id}: {e}")
                # Clean up partial/corrupt file on failure
                if pdf_path.exists():
                    pdf_path.unlink(missing_ok=True)
                time.sleep(5)  # Longer wait on error
                continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Download complete.")
    logger.info(f"  Downloaded : {total_downloaded}")
    logger.info(f"  Skipped    : {total_skipped}")
    logger.info(f"  Total in DB: {len(metadata)}")
    logger.info(f"  Metadata   : {META_FILE}")
    logger.info(f"{'='*60}")

    return metadata


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    logger.info("Starting arXiv downloader...")
    setup_dirs()
    metadata = load_existing_metadata()
    metadata = download_papers(SEARCH_QUERIES, PAPERS_PER_QUERY, metadata)
    save_metadata(metadata)
    logger.info("Done.")


if __name__ == "__main__":
    main()