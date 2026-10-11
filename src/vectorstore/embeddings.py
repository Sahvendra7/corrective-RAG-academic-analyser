"""
Embedding models behind one small interface.

Every embedder returns L2-normalised ``float32`` vectors, so inner product in
FAISS equals cosine similarity. The index manifest records the embedder name
and dimension; :class:`~src.vectorstore.faiss_store.FAISSStore` refuses to
query an index with a different embedder (mixed vector spaces return
plausible-looking but meaningless neighbours).

``HashingEmbedder`` ("hashing") is a deterministic bag-of-words embedder with
no download and no model weights. It exists so tests and CI can exercise the
full ingestion → index → retrieval path offline; it is not meant for real use.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

HASHING_MODEL = "hashing"


class Embedder(Protocol):
    name: str

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of L2-normalised vectors."""
        ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class HashingEmbedder:
    """Deterministic signed feature hashing of lower-cased word unigrams and bigrams."""

    def __init__(self, dim: int = 512) -> None:
        self.name = HASHING_MODEL
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype="float32")
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for feature in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dim
            vec[bucket] += 1.0 if digest[4] & 1 else -1.0
        return vec

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")
        return _normalise(np.stack([self._vector(t) for t in texts]))


class SentenceTransformerEmbedder:
    """Wrapper around a sentence-transformers model (loaded lazily, once)."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self.name)
                    self._model = SentenceTransformer(self.name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    @property
    def max_seq_length(self) -> int:
        return int(self._load().max_seq_length)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return _normalise(vectors)


@lru_cache(maxsize=4)
def get_embedder(model_name: str | None = None) -> Embedder:
    """Process-wide embedder for ``model_name`` (defaults to ``Settings.embedding_model``)."""
    if model_name is None:
        from src.config import get_settings

        model_name = get_settings().embedding_model
    if model_name == HASHING_MODEL:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name)
