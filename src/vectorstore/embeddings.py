"""
embeddings.py
--------------
Loads all chunks from data/processed/chunks/,
generates embeddings using sentence-transformers,
and saves them to data/embeddings/ as numpy arrays.

Also builds a chunk registry JSON mapping chunk_id -> metadata
for fast lookup during retrieval.

Usage:
    python src/vectorstore/embeddings.py
"""

import json
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
# tqdm is used internally by SentenceTransformer.encode(show_progress_bar=True)

# ── Config ────────────────────────────────────────────────────────────────────

CHUNK_DIR       = Path("data/processed/chunks")
EMBEDDINGS_DIR  = Path("data/embeddings")
META_FILE       = Path("data/processed/metadata.json")

# The embedding model to use
# all-MiniLM-L6-v2  → faster, smaller, good quality  (384 dimensions)
# all-mpnet-base-v2 → slower, larger, better quality (768 dimensions)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BATCH_SIZE = 64  # Number of chunks to embed at once — tune down if RAM issues

# ── Logging ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_dirs():
    """Create output directories if they don't exist."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Embeddings output directory ready: {EMBEDDINGS_DIR}")


def load_metadata() -> dict:
    """Load paper metadata from JSON file."""
    if not META_FILE.exists():
        logger.error(f"Metadata file not found: {META_FILE}")
        return {}
    with open(META_FILE, "r") as f:
        return json.load(f)


def load_chunk_registry() -> dict:
    """Load existing chunk registry if it exists."""
    registry_path = EMBEDDINGS_DIR / "chunk_registry.json"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            return json.load(f)
    return {}


def save_chunk_registry(registry: dict):
    """Save chunk registry to disk."""
    registry_path = EMBEDDINGS_DIR / "chunk_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Chunk registry saved: {len(registry)} entries")


# ── Load All Chunks ───────────────────────────────────────────────────────────

def load_all_chunks(metadata: dict, existing_registry: dict) -> list[dict]:
    """
    Load all chunk JSON files from disk.
    Skips papers whose chunks are already in the registry.

    Args:
        metadata: Paper metadata dict
        existing_registry: Already-embedded chunk registry

    Returns:
        List of chunk dicts to embed
    """
    all_chunks = []
    skipped_papers = 0

    for arxiv_id, paper in metadata.items():

        # Skip if this paper's chunks are already embedded
        if f"{arxiv_id}_chunk_0" in existing_registry:
            skipped_papers += 1
            continue

        chunks_path = paper.get("chunks_path")
        if not chunks_path:
            logger.warning(f"[SKIP] No chunks_path for {arxiv_id} — run chunker.py first")
            continue

        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            logger.warning(f"[MISS] Chunk file not found: {chunks_path}")
            continue

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            # Attach paper-level metadata to each chunk for later retrieval
            for chunk in chunks:
                chunk["title"]     = paper.get("title", "")
                chunk["authors"]   = paper.get("authors", [])
                chunk["published"] = paper.get("published", "")
                chunk["url"]       = paper.get("url", "")
                chunk["abstract"]  = paper.get("abstract", "")

            all_chunks.extend(chunks)

        except Exception as e:
            logger.error(f"Failed to load chunks for {arxiv_id}: {e}")
            continue

    logger.info(f"Loaded {len(all_chunks)} chunks from {len(metadata) - skipped_papers} papers")
    logger.info(f"Skipped {skipped_papers} already-embedded papers")
    return all_chunks


# ── Embedding Generation ──────────────────────────────────────────────────────

def load_model(model_name: str) -> SentenceTransformer:
    """
    Load the sentence-transformer embedding model.
    Downloads automatically on first run, cached locally after.

    Args:
        model_name: HuggingFace model name

    Returns:
        Loaded SentenceTransformer model
    """
    logger.info(f"Loading embedding model: {model_name}")
    logger.info("(This may take a minute on first run — model will be cached after)")
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embeddings(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int
) -> np.ndarray:
    """
    Generate embeddings for all chunks in batches.

    Args:
        chunks: List of chunk dicts (must have 'text' key)
        model: Loaded SentenceTransformer model
        batch_size: Number of chunks per batch

    Returns:
        numpy array of shape (num_chunks, embedding_dim)
    """
    texts = [chunk["text"] for chunk in chunks]
    total = len(texts)

    logger.info(f"Generating embeddings for {total} chunks in batches of {batch_size}...")

    # encode() handles batching internally but tqdm gives us a progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize — better for cosine similarity
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# ── Save Embeddings ───────────────────────────────────────────────────────────

def save_embeddings(embeddings: np.ndarray, chunks: list[dict], existing_registry: dict):
    """
    Save embeddings as a numpy .npy file and update the chunk registry.

    The registry maps chunk_id -> full chunk metadata (text, paper info, etc.)
    This lets us look up any chunk by ID during retrieval.

    Args:
        embeddings: numpy array of embeddings
        chunks: List of chunk dicts
        existing_registry: Already-existing registry to append to
    """
    # Load existing embeddings if they exist and stack with new ones
    existing_embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    existing_ids_path = EMBEDDINGS_DIR / "chunk_ids.json"

    if existing_embeddings_path.exists() and existing_ids_path.exists():
        logger.info("Found existing embeddings — appending new ones...")
        existing_embs = np.load(str(existing_embeddings_path))
        with open(existing_ids_path, "r") as f:
            existing_ids = json.load(f)

        # Stack old and new embeddings
        combined_embeddings = np.vstack([existing_embs, embeddings])
        combined_ids = existing_ids + [c["chunk_id"] for c in chunks]
    else:
        combined_embeddings = embeddings
        combined_ids = [c["chunk_id"] for c in chunks]

    # Save combined embeddings array
    np.save(str(existing_embeddings_path), combined_embeddings)
    logger.info(f"Saved embeddings: {combined_embeddings.shape} → {existing_embeddings_path}")

    # Save ordered list of chunk IDs (index i → chunk_id)
    with open(existing_ids_path, "w") as f:
        json.dump(combined_ids, f, indent=2)
    logger.info(f"Saved chunk IDs: {len(combined_ids)} entries")

    # Update registry with new chunks
    for chunk in chunks:
        existing_registry[chunk["chunk_id"]] = {
            "chunk_id":    chunk["chunk_id"],
            "arxiv_id":    chunk["arxiv_id"],
            "chunk_index": chunk["chunk_index"],
            "text":        chunk["text"],
            "word_count":  chunk["word_count"],
            "title":       chunk["title"],
            "authors":     chunk["authors"],
            "published":   chunk["published"],
            "url":         chunk["url"],
            "abstract":    chunk["abstract"],
        }

    save_chunk_registry(existing_registry)
    return combined_embeddings, combined_ids


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(embeddings: np.ndarray, chunks: list[dict]):
    """Print a summary of what was generated."""
    if not chunks:
        logger.warning("No chunks to compute stats for.")
        return
    word_counts = [c["word_count"] for c in chunks]
    logger.info(f"\n{'='*60}")
    logger.info(f"Embedding stats:")
    logger.info(f"  Total chunks embedded : {len(chunks)}")
    logger.info(f"  Embedding dimensions  : {embeddings.shape[1]}")
    logger.info(f"  Avg words per chunk   : {sum(word_counts) // len(word_counts)}")
    logger.info(f"  Min words per chunk   : {min(word_counts)}")
    logger.info(f"  Max words per chunk   : {max(word_counts)}")
    logger.info(f"  Embeddings file size  : {embeddings.nbytes / 1024 / 1024:.1f} MB")
    logger.info(f"{'='*60}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    logger.info("Starting embeddings generation...")
    setup_dirs()

    metadata = load_metadata()
    if not metadata:
        return

    existing_registry = load_chunk_registry()

    # Load all unembedded chunks
    chunks = load_all_chunks(metadata, existing_registry)
    if not chunks:
        logger.info("No new chunks to embed. All papers already processed.")
        return

    # Load model
    model = load_model(EMBEDDING_MODEL)

    # Generate embeddings
    embeddings = generate_embeddings(chunks, model, BATCH_SIZE)

    # Save to disk
    save_embeddings(embeddings, chunks, existing_registry)

    # Print summary
    print_stats(embeddings, chunks)

    logger.info("Done.")


if __name__ == "__main__":
    main()
