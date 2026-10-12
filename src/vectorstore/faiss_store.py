"""
FAISS index over corpus chunks, with a manifest that keeps vectors, chunk
metadata and the embedding model consistent.

On-disk layout (``Settings.index_dir``, default ``data/index``)::

    manifest.json   schema version, embedder name + dim, counts, chunking params
    faiss.index     IndexFlatIP over L2-normalised vectors (inner product = cosine)
    chunks.jsonl    one chunk per line; line i ↔ FAISS row i

The directory is written to a temporary sibling and swapped in with a rename,
so a crash mid-build never leaves a half-written index. Loading validates row
counts, vector dimension and the embedder name; any mismatch raises
:class:`IndexConsistencyError` instead of silently returning wrong chunks
(the v0.1 store loaded ``faiss.index``, ``chunk_ids.json`` and
``chunk_registry.json`` independently and appended new papers by guessing
from chunk-id prefixes, so a re-chunked paper produced a stale, misaligned
index).

Scores are cosine similarities in [-1, 1]. They rank passages for one query;
they are **not** probabilities of relevance and are not comparable across
embedding models.

Usage::

    python -m src.vectorstore.faiss_store --build              # PDFs (or .txt fallback) → index
    python -m src.vectorstore.faiss_store --build --text-only  # legacy .txt only
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import src.config as config
from src.ingestion.chunker import embedding_text, read_jsonl, write_jsonl
from src.ingestion.pdf_parser import PARSER_VERSION
from src.vectorstore.embeddings import Embedder, get_embedder

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
TOP_K_DEFAULT = config.TOP_K


class IndexConsistencyError(RuntimeError):
    """The on-disk index is incomplete, corrupt or built with another embedder."""


def _sha256_lines(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk["chunk_id"].encode())
        digest.update(chunk.get("content_hash", "").encode())
    return digest.hexdigest()


# ── Build ─────────────────────────────────────────────────────────────────────


def build_index(
    chunks: list[dict[str, Any]],
    out_dir: Path,
    embedder: Embedder | None = None,
    settings: config.Settings | None = None,
) -> dict[str, Any]:
    """Embed ``chunks`` and atomically (re)write the index directory. Returns the manifest."""
    settings = settings or config.get_settings()
    embedder = embedder or get_embedder(settings.embedding_model)
    if not chunks:
        raise ValueError("refusing to build an empty index")
    ids = [c["chunk_id"] for c in chunks]
    duplicates = [cid for cid, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate chunk ids: {duplicates[:5]}")

    started = time.perf_counter()
    texts = [embedding_text(c, settings.contextual_headers) for c in chunks]
    vectors = embedder.encode(texts, batch_size=settings.embedding_batch_size)
    if vectors.shape != (len(chunks), embedder.dim):
        raise IndexConsistencyError(f"embedder returned {vectors.shape}, expected ({len(chunks)}, {embedder.dim})")
    index = faiss.IndexFlatIP(embedder.dim)
    index.add(vectors)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": embedder.name,
        "dim": embedder.dim,
        "metric": "inner_product_on_l2_normalised_vectors",
        "num_vectors": int(index.ntotal),
        "num_papers": len({c["arxiv_id"] for c in chunks}),
        "chunks_sha256": _sha256_lines(chunks),
        "contextual_headers": settings.contextual_headers,
        "parser_version": PARSER_VERSION,
        "chunking": {
            "max_words": settings.chunk_max_words,
            "overlap_words": settings.chunk_overlap_words,
            "min_words": settings.chunk_min_words,
            "index_references": settings.index_references,
        },
        "source_formats": dict(Counter(c.get("source_format", "unknown") for c in chunks)),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embed_seconds": round(time.perf_counter() - started, 2),
    }

    out_dir = Path(out_dir)
    tmp_dir = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    try:
        faiss.write_index(index, str(tmp_dir / "faiss.index"))
        write_jsonl(chunks, tmp_dir / "chunks.jsonl")
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        old_dir = out_dir.with_name(f".{out_dir.name}.old-{os.getpid()}")
        if out_dir.exists():
            out_dir.rename(old_dir)
        tmp_dir.rename(out_dir)
        shutil.rmtree(old_dir, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    logger.info("Index built: %d vectors, %d papers → %s", manifest["num_vectors"], manifest["num_papers"], out_dir)
    return manifest


# ── Store ─────────────────────────────────────────────────────────────────────


class FAISSStore:
    """Read-only, validated view of an index directory.

    ``search`` returns plain dicts with the chunk text, paper metadata, page
    range, section and cosine ``score``.
    """

    def __init__(
        self,
        load_existing: bool = True,
        index_dir: Path | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        settings = config.get_settings()
        self.index_dir = Path(index_dir or settings.index_dir)
        self._embedder = embedder
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] = {}
        self._rows_by_paper: dict[str, list[int]] = {}
        self._row_by_id: dict[str, int] = {}
        self._lock = threading.Lock()
        if load_existing:
            if (self.index_dir / "manifest.json").exists():
                self.load()
            else:
                logger.warning("No index at %s — run `python -m src.vectorstore.faiss_store --build`", self.index_dir)

    # Compatibility with the v0.1 attribute names.
    @property
    def registry(self) -> dict[str, dict[str, Any]]:
        return {c["chunk_id"]: c for c in self.chunks}

    @property
    def chunk_ids(self) -> list[str]:
        return [c["chunk_id"] for c in self.chunks]

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.manifest.get("embedding_model"))
        return self._embedder

    def load(self) -> None:
        manifest_path = self.index_dir / "manifest.json"
        index_path = self.index_dir / "faiss.index"
        chunks_path = self.index_dir / "chunks.jsonl"
        for path in (manifest_path, index_path, chunks_path):
            if not path.exists():
                raise IndexConsistencyError(f"index file missing: {path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(index_path))
            chunks = read_jsonl(chunks_path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise IndexConsistencyError(f"cannot read index in {self.index_dir}: {exc}") from exc

        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise IndexConsistencyError(f"index schema {manifest.get('schema_version')} != {SCHEMA_VERSION}; rebuild it")
        if not (index.ntotal == len(chunks) == manifest.get("num_vectors")):
            raise IndexConsistencyError(
                f"row mismatch: faiss={index.ntotal} chunks={len(chunks)} manifest={manifest.get('num_vectors')}"
            )
        if index.d != manifest.get("dim"):
            raise IndexConsistencyError(f"dimension mismatch: faiss={index.d} manifest={manifest.get('dim')}")
        if _sha256_lines(chunks) != manifest.get("chunks_sha256"):
            raise IndexConsistencyError("chunks.jsonl does not match the manifest checksum")
        if self._embedder is not None and (
            self._embedder.name != manifest.get("embedding_model") or self._embedder.dim != index.d
        ):
            raise IndexConsistencyError(
                f"index built with {manifest.get('embedding_model')} ({index.d}d) "
                f"but query embedder is {self._embedder.name} ({self._embedder.dim}d)"
            )

        configured = config.get_settings().embedding_model
        if self._embedder is None and configured != manifest.get("embedding_model"):
            logger.warning(
                "Settings.embedding_model=%s but the index was built with %s; queries use the index's model",
                configured,
                manifest.get("embedding_model"),
            )
        self.manifest, self.index, self.chunks = manifest, index, chunks
        self._rows_by_paper = {}
        for row, chunk in enumerate(chunks):
            self._rows_by_paper.setdefault(chunk["arxiv_id"], []).append(row)
        self._row_by_id = {c["chunk_id"]: row for row, c in enumerate(chunks)}
        logger.info("Loaded index: %d vectors, %d papers (%s)", index.ntotal, len(self._rows_by_paper), manifest["embedding_model"])

    # ── Search ────────────────────────────────────────────────────────────────

    def _allowed_rows(
        self, paper_ids: list[str] | None, year_filter: str | None, author_filter: str | None
    ) -> list[int] | None:
        if not (paper_ids or year_filter or author_filter):
            return None
        rows = (
            [r for pid in paper_ids for r in self._rows_by_paper.get(pid, [])]
            if paper_ids
            else range(len(self.chunks))
        )
        out = []
        for row in rows:
            chunk = self.chunks[row]
            if year_filter and not str(chunk.get("published", "")).startswith(year_filter):
                continue
            if author_filter and not any(author_filter.lower() in a.lower() for a in chunk.get("authors") or []):
                continue
            out.append(row)
        return out

    def search(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        year_filter: str | None = None,
        author_filter: str | None = None,
        paper_ids: list[str] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``top_k`` distinct chunks ranked by cosine similarity.

        ``paper_ids``/``year_filter``/``author_filter`` restrict the search space
        *before* ranking (exact, not post-filtered), and ``exclude_ids`` skips
        chunks the caller already holds.
        """
        if not self.is_ready or not query.strip() or top_k <= 0:
            return []
        allowed = self._allowed_rows(paper_ids, year_filter, author_filter)
        if allowed is not None and not allowed:
            return []
        exclude_ids = exclude_ids or set()
        with self._lock:
            vector = self.embedder.encode([query])
        fetch = min(top_k + len(exclude_ids) + 10, len(allowed) if allowed is not None else self.index.ntotal)
        params = None
        if allowed is not None:
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.asarray(allowed, dtype="int64")))
        scores, rows = self.index.search(vector, fetch, params=params)

        results: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        for score, row in zip(scores[0].tolist(), rows[0].tolist()):
            if row < 0:
                continue
            chunk = self.chunks[row]
            key = chunk.get("content_hash") or chunk["text"]
            if chunk["chunk_id"] in exclude_ids or key in seen_hashes:
                continue
            seen_hashes.add(key)
            results.append({**chunk, "score": round(float(score), 4)})
            if len(results) >= top_k:
                break
        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    def papers(self) -> list[dict[str, Any]]:
        """One entry per indexed paper, sorted by title."""
        out = []
        for pid, rows in self._rows_by_paper.items():
            first = self.chunks[rows[0]]
            out.append(
                {
                    "arxiv_id": pid,
                    "title": first.get("title", ""),
                    "published": first.get("published", ""),
                    "url": first.get("url", ""),
                    "num_chunks": len(rows),
                }
            )
        return sorted(out, key=lambda p: p["title"].lower())

    def stats(self) -> dict[str, Any]:
        if not self.is_ready:
            return {"ready": False}
        return {
            "ready": True,
            "vectors": int(self.index.ntotal),
            "papers": len(self._rows_by_paper),
            "embedding_model": self.manifest.get("embedding_model"),
            "built_at": self.manifest.get("built_at"),
        }

    def get_chunk_by_id(self, chunk_id: str) -> dict[str, Any] | None:
        row = self._row_by_id.get(chunk_id)
        return self.chunks[row] if row is not None else None

    def get_paper_chunks(self, arxiv_id: str) -> list[dict[str, Any]]:
        return [self.chunks[r] for r in self._rows_by_paper.get(arxiv_id, [])]


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    from src.ingestion.chunker import chunk_corpus
    from src.utils.metadata_utils import load_metadata

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build or inspect the FAISS index")
    ap.add_argument("--build", action="store_true", help="chunk the corpus and rebuild the index")
    ap.add_argument("--text-only", action="store_true", help="ignore PDFs; use legacy .txt files")
    ap.add_argument("--out", type=Path, default=None, help="index directory (default: Settings.index_dir)")
    args = ap.parse_args()

    settings = config.get_settings()
    out = args.out or settings.index_dir
    if args.build:
        config.setup_dirs()
        chunks, stats = chunk_corpus(load_metadata(settings.metadata_file), settings, prefer_pdf=not args.text_only)
        logger.info("Chunking: %s", stats)
        manifest = build_index(chunks, out, settings=settings)
        print(json.dumps(manifest, indent=2))
    else:
        print(json.dumps(FAISSStore(index_dir=out).stats(), indent=2))


if __name__ == "__main__":
    main()
