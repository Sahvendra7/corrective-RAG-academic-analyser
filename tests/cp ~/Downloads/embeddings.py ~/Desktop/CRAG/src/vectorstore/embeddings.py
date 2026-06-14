import json
import logging
from pathlib import Path
import numpy as np

# Note: Make sure your virtual environment has sentence-transformers installed:
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
TEST_CHUNKS_FILE = Path("data/processed/chunks/1003.3081_chunks.json")
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE       = 64

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_single_paper_embeddings():
    logger.info(f"--- Starting Embedding Test for File: {TEST_CHUNKS_FILE.name} ---")
    
    # 1. Validation checks
    if not TEST_CHUNKS_FILE.exists():
        logger.error(f"Test target chunk file not found: {TEST_CHUNKS_FILE}")
        logger.error("Please run your main chunker.py script first to generate paper chunks.")
        return

    # 2. Read the paper's chunks
    with open(TEST_CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    if not chunks:
        logger.error("The selected chunk file contains an empty list.")
        return
        
    logger.info(f"Successfully loaded {len(chunks)} text chunks from disk.")

    # 3. Spin up the Transformer Model
    logger.info(f"Initializing Transformer Model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model successfully initialized. Target Dimensionality: {embedding_dim}")

    # 4. Extract text payloads
    texts = [chunk["text"] for chunk in chunks]

    # 5. Execute production embedding generation pipeline with internally handled batching
    logger.info(f"Running vector extraction over batches (Batch Size: {BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Crucial L2 normalization check
    )

    # 6. Verify mathematical assertions and vector constraints
    print("\n" + "="*60)
    print(" Verifying Matrix Shapes & Geometric Vector Lengths")
    print("="*60)
    
    # Matrix Shape Check
    logger.info(f"Resulting Embeddings Matrix Shape: {embeddings.shape}")
    assert embeddings.shape == (len(chunks), embedding_dim), "Matrix dimensions do not match chunk counts!"
    logger.info("[PASS] Matrix rows safely correspond to individual chunk counts.")
    
    # Mathematical L2 Normalization Verification (Vector Magnitude = 1.0)
    # The dot product of a perfectly L2-normalized vector with itself must equal 1.0
    sample_vector = embeddings[0]
    vector_magnitude = np.linalg.norm(sample_vector)
    logger.info(f"Calculated Magnitude of Chunk 0 Vector: {vector_magnitude:.5f}")
    
    # Check close approximation within floating point errors
    if np.isclose(vector_magnitude, 1.0, atol=1e-5):
        logger.info("[PASS] L2 Normalization verified. Vectors are safely constrained to unit-sphere length.")
    else:
        logger.warning("[FAIL] Vectors are unnormalized. Dot product optimization will fail during indexing.")

    # 7. Print structural snippet for verification
    print("\n" + "-"*40)
    print("Sample Output Preview (First Chunk Payload)")
    print("-"*40)
    print(f"Chunk ID       : {chunks[0]['chunk_id']}")
    print(f"Word Count     : {chunks[0].get('word_count')} words")
    print(f"Vector Snippet : {sample_vector[:5].tolist()}... [Truncated]")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_single_paper_embeddings()