"""
test_pipeline.py
-----------------
Tests the complete CRAG Phase 1 + Phase 2 pipeline
on a single paper end-to-end.

Run this before processing all 500 papers to verify
every step works correctly on your machine.

Tests (in order):
    Step 1: Download 1 paper from arXiv
    Step 2: Parse the PDF to text
    Step 3: Chunk the text
    Step 4: Generate embeddings
    Step 5: Build FAISS index
    Step 6: Run a test query through the full CRAG pipeline (Gemini)

Usage:
    cd ~/Desktop/CRAG
    conda activate crag-env
    python tests/test_pipeline.py
"""

import json
import logging
import sys
import time
from pathlib import Path
import src.config as config

# ── Setup paths ───────────────────────────────────────────────────────────────

# Allow imports from project root
PROJECT_ROOT = config.PROJECT_ROOT

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Test Config ───────────────────────────────────────────────────────────────

# The CRAG paper itself — perfect for testing a RAG system about RAG
TEST_ARXIV_ID = "2401.15884"
TEST_QUERY    = "How does Corrective RAG handle irrelevant retrieved documents?"

# Directories
DATA_DIR       = config.DATA_DIR
RAW_DIR        = config.RAW_DIR
PROCESSED_DIR  = config.PROCESSED_DIR
TEXT_DIR       = config.TEXT_DIR
CHUNK_DIR      = config.CHUNK_DIR
EMBEDDINGS_DIR = config.EMBEDDINGS_DIR
META_FILE      = PROCESSED_DIR / "metadata.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(label: str, value, success: bool = True):
    icon = "✅" if success else "❌"
    print(f"  {icon}  {label}: {value}")


def print_separator():
    print(f"  {'─'*56}")


import pytest
from src.config import setup_dirs

@pytest.fixture(autouse=True)
def init_test_dirs():
    """Create all required directories before each test."""
    setup_dirs()

# ── Step 1: Locate or Download Single Paper ───────────────────────────────────

def test_step1_download():
    print_header("STEP 1: Locate or Download Single Paper")

    import arxiv
    import requests

    try:
        client = arxiv.Client(
            page_size=1,
            delay_seconds=3,
            num_retries=3,
        )

        search = arxiv.Search(id_list=[TEST_ARXIV_ID])
        results = list(client.results(search))

        if not results:
            print_result("Paper found", "NOT FOUND", success=False)
            pytest.fail("Paper not found on arXiv")

        result = results[0]
        print_result("Paper found", result.title[:60])
        print_result("Authors", ", ".join(a.name for a in result.authors[:2]))
        print_result("Published", result.published.strftime("%Y-%m-%d"))

        # --- THE FIX: Look for existing file first ---
        existing_pdfs = list(RAW_DIR.glob(f"*{TEST_ARXIV_ID}*.pdf"))
        
        if existing_pdfs:
            pdf_path = existing_pdfs[0]
            print_result("PDF", f"Found existing PDF — skipping download: {pdf_path.name}")
        else:
            pdf_filename = f"{TEST_ARXIV_ID}_test_paper.pdf"
            pdf_path     = RAW_DIR / pdf_filename
            print(f"\n  Downloading PDF directly...")
            
            # Fallback: Direct download using requests to avoid arxiv version bugs
            res = requests.get(result.pdf_url)
            with open(pdf_path, 'wb') as f:
                f.write(res.content)
            print_result("PDF downloaded", str(pdf_path))

        # Verify file exists and has content
        if not pdf_path.exists():
            raise RuntimeError("PDF file not found after download")
        if pdf_path.stat().st_size <= 10_000:
            raise RuntimeError("PDF file seems too small")
        print_result("PDF size", f"{pdf_path.stat().st_size / 1024:.1f} KB")

        # Save minimal metadata
        metadata = {
            TEST_ARXIV_ID: {
                "arxiv_id" : TEST_ARXIV_ID,
                "title"    : result.title,
                "authors"  : [a.name for a in result.authors],
                "abstract" : result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "categories": result.categories,
                "pdf_path" : str(pdf_path),
                "query"    : "test",
                "url"      : result.entry_id,
            }
        }
        with open(META_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        print_result("Metadata saved", str(META_FILE))

        print(f"\n  ✅ Step 1 PASSED")

    except Exception as e:
        print_result("Error", str(e), success=False)
        print(f"\n  ❌ Step 1 FAILED")
        raise

# ── Step 2: Parse PDF ─────────────────────────────────────────────────────────

def test_step2_parse():
    print_header("STEP 2: Parse PDF to Text")

    try:
        import fitz  # PyMuPDF

        # Load metadata
        with open(META_FILE, "r") as f:
            metadata = json.load(f)

        paper    = metadata[TEST_ARXIV_ID]
        pdf_path = Path(paper["pdf_path"])

        assert pdf_path.exists(), f"PDF not found: {pdf_path}"
        print_result("PDF found", pdf_path.name)

        # Extract text
        doc        = fitz.open(str(pdf_path))
        pages_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pages_text.append(page.get_text())

        doc.close()
        raw_text = "\n".join(pages_text)
        print_result("Pages extracted", len(pages_text))
        print_result("Raw text length", f"{len(raw_text):,} chars")

        # Basic cleaning
        import re
        text = raw_text
        text = text.replace("\x0c", " ")
        text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f]", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        assert len(text) > 500, "Extracted text too short — likely scanned PDF"
        print_result("Cleaned text length", f"{len(text):,} chars")
        print_result("Word count", f"{len(text.split()):,} words")

        # Save text file
        txt_path = TEXT_DIR / f"{TEST_ARXIV_ID}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print_result("Text file saved", str(txt_path))

        # Update metadata
        metadata[TEST_ARXIV_ID]["txt_path"]   = str(txt_path)
        metadata[TEST_ARXIV_ID]["char_count"] = len(text)
        metadata[TEST_ARXIV_ID]["word_count"] = len(text.split())
        with open(META_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        # Print a sample of extracted text
        print_separator()
        print(f"  Sample text (first 300 chars):")
        print(f"  {text[:300].replace(chr(10), ' ')}")

        print(f"\n  ✅ Step 2 PASSED")

    except Exception as e:
        logger.exception("Step 2 error")
        print_result("Error", str(e), success=False)
        print(f"\n  ❌ Step 2 FAILED")
        raise


# ── Step 3: Chunk Text ────────────────────────────────────────────────────────

def test_step3_chunk():
    print_header("STEP 3: Chunk Text")

    try:
        import re

        CHUNK_SIZE    = config.CHUNK_SIZE * 2
        CHUNK_OVERLAP = config.CHUNK_OVERLAP
        MIN_CHUNK     = config.MIN_CHUNK_SIZE
        # NOTE: These values differ from production chunker.py (which uses 256/64/50).
        # This is intentional — the test uses a larger chunk size for a single paper
        # to reduce the number of chunks and speed up the test.

        # Load text
        txt_path = TEXT_DIR / f"{TEST_ARXIV_ID}.txt"
        assert txt_path.exists(), f"Text file not found: {txt_path}"

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        print_result("Text loaded", f"{len(text):,} chars")

        # Split into sentences
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        print_result("Sentences found", len(sentences))

        # Chunk
        chunks         = []
        chunk_index    = 0
        curr_sentences = []
        curr_words     = 0

        for sentence in sentences:
            sent_words = len(sentence.split())
            curr_sentences.append(sentence)
            curr_words += sent_words

            if curr_words >= CHUNK_SIZE:
                chunk_text = " ".join(curr_sentences)
                if len(chunk_text.split()) >= MIN_CHUNK:
                    chunks.append({
                        "chunk_id"   : f"{TEST_ARXIV_ID}_chunk_{chunk_index}",
                        "arxiv_id"   : TEST_ARXIV_ID,
                        "chunk_index": chunk_index,
                        "text"       : chunk_text,
                        "word_count" : len(chunk_text.split()),
                        "char_count" : len(chunk_text),
                    })
                    chunk_index += 1

                # Build overlap
                overlap_sents = []
                overlap_words = 0
                for s in reversed(curr_sentences):
                    sw = len(s.split())
                    if overlap_words + sw <= CHUNK_OVERLAP:
                        overlap_sents.insert(0, s)
                        overlap_words += sw
                    else:
                        break

                curr_sentences = overlap_sents
                curr_words     = overlap_words

        # Last chunk
        if curr_sentences:
            chunk_text = " ".join(curr_sentences)
            if len(chunk_text.split()) >= MIN_CHUNK:
                chunks.append({
                    "chunk_id"   : f"{TEST_ARXIV_ID}_chunk_{chunk_index}",
                    "arxiv_id"   : TEST_ARXIV_ID,
                    "chunk_index": chunk_index,
                    "text"       : chunk_text,
                    "word_count" : len(chunk_text.split()),
                    "char_count" : len(chunk_text),
                })

        assert len(chunks) > 0, "No chunks produced"
        print_result("Chunks produced", len(chunks))

        word_counts = [c["word_count"] for c in chunks]
        print_result("Avg words/chunk", f"{sum(word_counts)//len(word_counts)}")
        print_result("Min words/chunk", min(word_counts))
        print_result("Max words/chunk", max(word_counts))

        # Save chunks
        chunk_path = CHUNK_DIR / f"{TEST_ARXIV_ID}_chunks.json"
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        print_result("Chunks saved", str(chunk_path))

        # Update metadata
        with open(META_FILE, "r") as f:
            metadata = json.load(f)
        metadata[TEST_ARXIV_ID]["chunks_path"] = str(chunk_path)
        metadata[TEST_ARXIV_ID]["num_chunks"]  = len(chunks)
        with open(META_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        # Print sample chunk
        print_separator()
        print(f"  Sample chunk 0 (first 200 chars):")
        print(f"  {chunks[0]['text'][:200].replace(chr(10), ' ')}")

        print(f"\n  ✅ Step 3 PASSED")

    except Exception as e:
        logger.exception("Step 3 error")
        print_result("Error", str(e), success=False)
        print(f"\n  ❌ Step 3 FAILED")
        raise


# ── Step 4: Generate Embeddings ───────────────────────────────────────────────

def test_step4_embeddings():
    print_header("STEP 4: Generate Embeddings")

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        EMBEDDING_MODEL = "all-MiniLM-L6-v2"

        # Load chunks
        chunk_path = CHUNK_DIR / f"{TEST_ARXIV_ID}_chunks.json"
        assert chunk_path.exists(), f"Chunk file not found: {chunk_path}"

        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        print_result("Chunks loaded", len(chunks))

        # Load model
        print(f"\n  Loading model: {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        dim   = model.get_sentence_embedding_dimension()
        print_result("Model loaded", EMBEDDING_MODEL)
        print_result("Embedding dim", dim)

        # Generate embeddings
        texts = [c["text"] for c in chunks]
        print(f"\n  Generating embeddings for {len(texts)} chunks...")

        start      = time.time()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = time.time() - start

        print_result("Embeddings shape", embeddings.shape)
        print_result("Time taken", f"{elapsed:.1f}s")
        print_result("Speed", f"{len(texts)/elapsed:.1f} chunks/sec")

        # Verify shape
        assert embeddings.shape[0] == len(chunks), "Embedding count mismatch"
        assert embeddings.shape[1] == dim, "Embedding dimension mismatch"

        # Save embeddings
        emb_path = EMBEDDINGS_DIR / "embeddings.npy"
        ids_path = EMBEDDINGS_DIR / "chunk_ids.json"
        np.save(str(emb_path), embeddings)
        chunk_ids = [c["chunk_id"] for c in chunks]
        with open(ids_path, "w") as f:
            json.dump(chunk_ids, f)

        # Save registry
        with open(META_FILE, "r") as f:
            metadata = json.load(f)
        paper = metadata[TEST_ARXIV_ID]

        registry = {}
        for chunk in chunks:
            registry[chunk["chunk_id"]] = {
                **chunk,
                "title"    : paper.get("title", ""),
                "authors"  : paper.get("authors", []),
                "published": paper.get("published", ""),
                "url"      : paper.get("url", ""),
                "abstract" : paper.get("abstract", ""),
            }

        registry_path = EMBEDDINGS_DIR / "chunk_registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        print_result("Embeddings saved", str(emb_path))
        print_result("Chunk IDs saved", str(ids_path))
        print_result("Registry saved", str(registry_path))
        print_result("File size", f"{embeddings.nbytes / 1024:.1f} KB")

        print(f"\n  ✅ Step 4 PASSED")

    except Exception as e:
        logger.exception("Step 4 error")
        print_result("Error", str(e), success=False)
        print(f"\n  ❌ Step 4 FAILED")
        raise


# ── Step 5: Build FAISS Index ─────────────────────────────────────────────────

def test_step5_faiss():
    print_header("STEP 5: Build FAISS Index and Test Search")

    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        EMBEDDING_DIM   = 384

        # Load embeddings
        emb_path = EMBEDDINGS_DIR / "embeddings.npy"
        ids_path = EMBEDDINGS_DIR / "chunk_ids.json"
        reg_path = EMBEDDINGS_DIR / "chunk_registry.json"

        assert emb_path.exists(), "Embeddings not found — run Step 4 first"

        embeddings = np.load(str(emb_path)).astype("float32")
        with open(ids_path, "r") as f:
            chunk_ids = json.load(f)
        with open(reg_path, "r") as f:
            registry = json.load(f)

        print_result("Embeddings loaded", embeddings.shape)
        print_result("Chunk IDs loaded", len(chunk_ids))
        print_result("Registry loaded", len(registry))

        # Build FAISS index
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)
        print_result("FAISS index built", f"{index.ntotal} vectors")

        # Save index
        index_path = EMBEDDINGS_DIR / "faiss.index"
        faiss.write_index(index, str(index_path))

        # Save chunk_ids for FAISS
        faiss_ids_path = EMBEDDINGS_DIR / "faiss_chunk_ids.json"
        with open(faiss_ids_path, "w") as f:
            json.dump(chunk_ids, f)

        print_result("Index saved", str(index_path))

        # Test search
        print(f"\n  Testing search...")
        print(f"  Query: '{TEST_QUERY[:60]}'")

        model           = SentenceTransformer(EMBEDDING_MODEL)
        query_embedding = model.encode(
            [TEST_QUERY],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = index.search(query_embedding, 3)
        scores  = scores[0].tolist()
        indices = indices[0].tolist()

        print_separator()
        print(f"  Top 3 search results:")
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            chunk_id = chunk_ids[idx]
            chunk    = registry[chunk_id]
            print(f"\n  [{rank}] Score: {score:.4f}")
            print(f"       Chunk: {chunk_id}")
            print(f"       Text : {chunk['text'][:150].replace(chr(10), ' ')}...")

        assert scores[0] > 0.3, "Top result score too low — embeddings may be wrong"
        print_result("\n  Top score", f"{scores[0]:.4f}", success=scores[0] > 0.3)

        print(f"\n  ✅ Step 5 PASSED")

    except Exception as e:
        logger.exception("Step 5 error")
        print_result("Error", str(e), success=False)
        print(f"\n  ❌ Step 5 FAILED")
        raise


# ── Step 6: Run Full CRAG Pipeline ────────────────────────────────────────────

def test_step6_pipeline():
    print_header("STEP 6: Full CRAG Pipeline (requires Google/Tavily API keys)")

    import os
    import src.config as config

    # 1. Check for your exact variable name from the .env file
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not gemini_api_key or gemini_api_key == "your_key_here":
        print(f"  ⚠️  GEMINI_API_KEY not set in .env file or is empty.")
        print(f"  Skipping Step 6 — add your Gemini API key to .env to test the full pipeline")
        return
        
    # 2. CRITICAL FIX: LangChain's Google GenAI package explicitly looks for "GOOGLE_API_KEY".
    # We take your GEMINI_API_KEY and map it so the LLM nodes can see it.
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
    if not tavily_api_key:
        print(f"  ⚠️  TAVILY_API_KEY not set. Web search fallback might fail, but proceeding anyway...")
        os.environ["TAVILY_API_KEY"] = ""

    try:
        from src.pipeline.graph import run_query

        print(f"  Query: '{TEST_QUERY}'")
        print(f"\n  Running pipeline...\n")

        start  = time.time()
        result = run_query(TEST_QUERY, verbose=False)
        elapsed = time.time() - start

        print_separator()
        print_result("Grade", result.get("grade", "N/A").upper())
        print_result("Source", result.get("source", "N/A"))
        print_result("Docs retrieved", len(result.get("documents", [])))
        print_result("Retry count", result.get("retry_count", 0))
        print_result("Web search used", result.get("web_search_used", False))
        print_result("Hallucination", result.get("hallucination", False))
        print_result("Time taken", f"{elapsed:.1f}s")

        generation = result.get("generation", "")
        assert generation, "No answer generated"
        assert len(generation) > 50, "Answer too short"

        print_separator()
        print(f"\n  GENERATED ANSWER (first 500 chars):")
        print(f"  {generation[:500].replace(chr(10), ' ')}")

        print(f"\n  ✅ Step 6 PASSED")

    except Exception as e:
        logger.exception("Step 6 error")
        print_result("Error", str(e), success=False)
        print(f"\n  ❌ Step 6 FAILED")
        raise


# ── Pytest Runner ─────────────────────────────────────────────────────────────

# To run this script:
#   pytest tests/test_pipeline.py -v -s