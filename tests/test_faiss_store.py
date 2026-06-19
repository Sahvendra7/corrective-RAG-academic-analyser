import json
import logging
from pathlib import Path
import numpy as np
import faiss
import pytest

# Import your production-grade store class
from src.vectorstore.faiss_store import FAISSStore

# ── Test Configuration ────────────────────────────────────────────────────────
TEST_EMBEDDINGS_DIR = Path("data/embeddings")
TEST_INDEX_PATH     = TEST_EMBEDDINGS_DIR / "faiss_test.index"
TEST_CHUNKS_FILE    = Path("data/processed/chunks/1003.3081_chunks.json")
EMBEDDING_DIM       = 384

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class FAISSStoreTestHelper(FAISSStore):
    """
    Subclass of production FAISSStore that overrides data paths 
    to test index generation on a single paper in isolation.
    """
    def build_test_index(self, test_arxiv_id: str):
        logger.info(f"--- Building Isolated FAISS Index for Paper: {test_arxiv_id} ---")
        
        # 1. Load your production artifacts
        embeddings_path = TEST_EMBEDDINGS_DIR / "embeddings.npy"
        ids_path = TEST_EMBEDDINGS_DIR / "chunk_ids.json"
        registry_path = TEST_EMBEDDINGS_DIR / "chunk_registry.json"
        
        if not embeddings_path.exists() or not ids_path.exists() or not registry_path.exists():
            logger.error("Global embedding artifacts missing. Run embeddings.py first.")
            return False

        # Load global arrays
        all_embeddings = np.load(str(embeddings_path)).astype("float32")
        with open(ids_path, "r") as f:
            all_chunk_ids = json.load(f)
        with open(registry_path, "r") as f:
            all_registry = json.load(f)

        # 2. Extract ONLY rows corresponding to our target test paper
        test_indices = [i for i, cid in enumerate(all_chunk_ids) if cid.startswith(test_arxiv_id)]
        
        if not test_indices:
            logger.error(f"No embeddings found matching paper ID: {test_arxiv_id} in global matrix.")
            return False
            
        logger.info(f"Extracting {len(test_indices)} chunk rows from the global matrix...")
        
        # Slice matrix rows and filter metadata structures
        self.chunk_ids = [all_chunk_ids[idx] for idx in test_indices]
        test_embeddings = all_embeddings[test_indices]
        self.registry = {cid: all_registry[cid] for cid in self.chunk_ids if cid in all_registry}

        # 3. Compile FAISS Structure (Using production Flat Inner Product logic)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(test_embeddings)
        
        logger.info(f"Isolated FAISS Index initialized with total rows: {self.index.ntotal}")
        
        # 4. Save test index binary to disk
        faiss.write_index(self.index, str(TEST_INDEX_PATH))
        logger.info(f"Test index safely serialized to: {TEST_INDEX_PATH}")
        
        # Initialize internal embedding engine for query handling
        self._load_model()
        return True

def run_isolated_store_test():
    test_arxiv_id = "1003.3081"
    
    # Initialize store without loading global production file
    store = FAISSStoreTestHelper(load_existing=False)
    
    # Build isolated context
    success = store.build_test_index(test_arxiv_id)
    if not success:
        return

    # Verify basic internal stats
    print("\n" + "="*60)
    print(" Running In-Memory Matrix & Integrity Verifications")
    print("="*60)
    logger.info(f"FAISS Indexed Vectors : {store.index.ntotal} (Expected: 26)")
    logger.info(f"Chunk ID Mapping Size : {len(store.chunk_ids)} matches.")
    logger.info(f"Text Registry Records : {len(store.registry)} paragraphs cached.")
    
    assert store.index.ntotal == len(store.chunk_ids) == len(store.registry), "Data synchronization mismatch!"
    logger.info("[PASS] Vector rows are perfectly synchronized with metadata structures.")

    # 4. Fire an isolated semantic retrieval query
    print("\n" + "-"*40)
    print(" Executing Live Semantic Search Test")
    print("-"*40)
    test_query = "What is limited sustained activity (LSA) in cortical column networks?"
    top_k = 2
    
    results = store.search(test_query, top_k=top_k)
    
    print(f"\nRetrieved top {len(results)} chunks for query: '{test_query}'")
    for i, r in enumerate(results):
        print(f"\n[Rank {i+1}] Similarity Score: {r['score']:.4f}")
        print(f"Chunk Location  : {r['chunk_id']}")
        print(f"Paper Title     : {r['title']}")
        print(f"Text Payload    : {r['text'][:180]}...")
    print("="*60 + "\n")

    # Clean up test binary file from disk
    if TEST_INDEX_PATH.exists():
        TEST_INDEX_PATH.unlink()

def test_isolated_store():
    """Pytest entrypoint to execute the isolated FAISS store tests."""
    run_isolated_store_test()

if __name__ == "__main__":
    run_isolated_store_test()