"""
faiss_store.py
---------------
Builds a FAISS index from precomputed embeddings,
enables fast similarity search over 500 paper chunks,
and supports metadata filtering by year, topic, and author.

Usage:
    # Build the index
    python src/vectorstore/faiss_store.py

    # Or import and use in other modules
    from src.vectorstore.faiss_store import FAISSStore
    store = FAISSStore()
    results = store.search("what is corrective RAG?", top_k=5)
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR  = Path("data/embeddings")
INDEX_PATH      = EMBEDDINGS_DIR / "faiss.index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384   # Must match the model used in embeddings.py
TOP_K_DEFAULT   = 5     # Default number of results to return

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/embeddings/faiss.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── FAISSStore Class ──────────────────────────────────────────────────────────

class FAISSStore:
    """
    Wraps a FAISS index with chunk metadata for semantic search.

    Attributes:
        index: The FAISS index object
        chunk_ids: Ordered list mapping FAISS row index → chunk_id string
        registry: Dict mapping chunk_id → full chunk metadata + text
        model: SentenceTransformer model for query encoding
    """

    def __init__(self, load_existing: bool = True):
        """
        Initialise the store. Loads existing index if available.

        Args:
            load_existing: If True, load index from disk on startup
        """
        self.index      = None
        self.chunk_ids  = []
        self.registry   = {}
        self.model      = None

        if load_existing and INDEX_PATH.exists():
            self.load()
        else:
            logger.info("No existing index found. Call build() to create one.")


    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self):
        """
        Build a FAISS index from embeddings saved by embeddings.py.
        Loads embeddings.npy, chunk_ids.json, and chunk_registry.json.
        Saves the built index to data/embeddings/faiss.index.
        """
        logger.info("Building FAISS index...")

        # Load embeddings array
        embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
        if not embeddings_path.exists():
            logger.error(f"Embeddings not found: {embeddings_path}")
            logger.error("Run embeddings.py first.")
            return

        embeddings = np.load(str(embeddings_path)).astype("float32")
        logger.info(f"Loaded embeddings: {embeddings.shape}")

        # Load chunk IDs (ordered list matching embedding rows)
        ids_path = EMBEDDINGS_DIR / "chunk_ids.json"
        with open(ids_path, "r") as f:
            self.chunk_ids = json.load(f)

        # Load chunk registry (chunk_id → metadata + text)
        registry_path = EMBEDDINGS_DIR / "chunk_registry.json"
        with open(registry_path, "r") as f:
            self.registry = json.load(f)

        logger.info(f"Loaded {len(self.chunk_ids)} chunk IDs")
        logger.info(f"Loaded {len(self.registry)} registry entries")

        # Sanity check — embeddings rows must match chunk IDs count
        assert len(embeddings) == len(self.chunk_ids), (
            f"Mismatch: {len(embeddings)} embeddings vs {len(self.chunk_ids)} chunk IDs"
        )

        # Build FAISS index
        # IndexFlatIP = Flat (brute-force) index using Inner Product (dot product)
        # Since embeddings are L2-normalised, dot product == cosine similarity
        # For 500 papers this is fast enough — no need for approximate search
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)

        logger.info(f"FAISS index built. Total vectors: {self.index.ntotal}")

        # Save index to disk
        self.save()

        # Load embedding model for query encoding
        self._load_model()


    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, str(INDEX_PATH))
        logger.info(f"FAISS index saved: {INDEX_PATH}")

        # Also save chunk_ids alongside the index
        ids_backup = EMBEDDINGS_DIR / "faiss_chunk_ids.json"
        with open(ids_backup, "w") as f:
            json.dump(self.chunk_ids, f)
        logger.info(f"Chunk IDs saved: {ids_backup}")


    def load(self):
        """Load FAISS index and supporting data from disk."""
        logger.info(f"Loading FAISS index from {INDEX_PATH}...")

        # Load index
        self.index = faiss.read_index(str(INDEX_PATH))
        logger.info(f"FAISS index loaded. Vectors: {self.index.ntotal}")

        # Load chunk IDs
        ids_path = EMBEDDINGS_DIR / "faiss_chunk_ids.json"
        if not ids_path.exists():
            ids_path = EMBEDDINGS_DIR / "chunk_ids.json"
        with open(ids_path, "r") as f:
            self.chunk_ids = json.load(f)

        # Load registry
        registry_path = EMBEDDINGS_DIR / "chunk_registry.json"
        with open(registry_path, "r") as f:
            self.registry = json.load(f)

        logger.info(f"Loaded {len(self.chunk_ids)} chunk IDs")
        logger.info(f"Loaded {len(self.registry)} registry entries")

        # Load model for query encoding
        self._load_model()


    def _load_model(self):
        """Load sentence transformer model for encoding queries."""
        if self.model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Model loaded.")


    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        year_filter: str | None = None,
        author_filter: str | None = None,
    ) -> list[dict]:
        """
        Search the FAISS index for chunks most similar to the query.

        Args:
            query: Natural language question or search string
            top_k: Number of results to return
            year_filter: Only return chunks from papers published in this year
                         e.g. "2024"
            author_filter: Only return chunks from papers by this author
                           e.g. "Hinton" (case-insensitive substring match)

        Returns:
            List of result dicts, each containing:
                - chunk_id, arxiv_id, text, score
                - title, authors, published, url
        """
        if self.index is None:
            logger.error("Index not loaded. Call build() or load() first.")
            return []

        # Step 1: Encode the query into an embedding vector
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Step 2: Search FAISS — fetch more results than needed
        # so we have room to filter by metadata
        fetch_k = top_k * 10 if (year_filter or author_filter) else top_k
        fetch_k = min(fetch_k, self.index.ntotal)  # Can't fetch more than we have

        scores, indices = self.index.search(query_embedding, fetch_k)

        # scores shape: (1, fetch_k) — one query, fetch_k results
        # indices shape: (1, fetch_k) — row indices into embeddings array
        scores  = scores[0].tolist()
        indices = indices[0].tolist()

        # Step 3: Build result list with full metadata
        results = []
        for score, idx in zip(scores, indices):

            # FAISS returns -1 for empty slots — skip them
            if idx == -1:
                continue

            chunk_id = self.chunk_ids[idx]
            chunk    = self.registry.get(chunk_id)

            if chunk is None:
                continue

            # Apply year filter if specified
            if year_filter:
                published = chunk.get("published", "")
                if not published.startswith(year_filter):
                    continue

            # Apply author filter if specified (case-insensitive)
            if author_filter:
                authors = chunk.get("authors", [])
                author_match = any(
                    author_filter.lower() in author.lower()
                    for author in authors
                )
                if not author_match:
                    continue

            results.append({
                "chunk_id":  chunk_id,
                "arxiv_id":  chunk["arxiv_id"],
                "text":      chunk["text"],
                "score":     round(score, 4),
                "title":     chunk.get("title", ""),
                "authors":   chunk.get("authors", []),
                "published": chunk.get("published", ""),
                "url":       chunk.get("url", ""),
                "abstract":  chunk.get("abstract", ""),
            })

            # Stop once we have enough filtered results
            if len(results) >= top_k:
                break

        logger.info(
            f"Query: '{query[:60]}' - {len(results)} results "
            f"(filters: year={year_filter}, author={author_filter})"
        )

        return results


    # ── Utility ───────────────────────────────────────────────────────────────

    def stats(self):
        """Print index statistics."""
        if self.index is None:
            logger.info("No index loaded.")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"FAISS Index Stats:")
        logger.info(f"  Total vectors   : {self.index.ntotal}")
        logger.info(f"  Dimensions      : {EMBEDDING_DIM}")
        logger.info(f"  Index type      : {type(self.index).__name__}")
        logger.info(f"  Registry size   : {len(self.registry)} chunks")
        logger.info(f"  Unique papers   : {len(set(c['arxiv_id'] for c in self.registry.values()))}")
        logger.info(f"{'='*60}")


    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        """
        Fetch a specific chunk by its ID.

        Args:
            chunk_id: e.g. "2401.15884_chunk_3"

        Returns:
            Chunk dict or None if not found
        """
        return self.registry.get(chunk_id)


    def get_paper_chunks(self, arxiv_id: str) -> list[dict]:
        """
        Fetch all chunks belonging to a specific paper.

        Args:
            arxiv_id: e.g. "2401.15884"

        Returns:
            List of chunk dicts sorted by chunk_index
        """
        chunks = [
            chunk for chunk in self.registry.values()
            if chunk["arxiv_id"] == arxiv_id
        ]
        return sorted(chunks, key=lambda c: c["chunk_index"])


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    """Build the FAISS index and run a test search."""
    logger.info("Starting FAISS store build...")

    store = FAISSStore(load_existing=False)
    store.build()
    store.stats()

    # Test search
    logger.info("\nRunning test search...")
    test_query = "How does corrective RAG improve retrieval quality?"
    results = store.search(test_query, top_k=3)

    logger.info(f"\nTop {len(results)} results for: '{test_query}'")
    for i, r in enumerate(results):
        logger.info(f"\n--- Result {i+1} ---")
        logger.info(f"  Score   : {r['score']}")
        logger.info(f"  Paper   : {r['title'][:70]}")
        logger.info(f"  Authors : {', '.join(r['authors'][:2])}")
        logger.info(f"  Year    : {r['published'][:4]}")
        logger.info(f"  Text    : {r['text'][:150]}...")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
