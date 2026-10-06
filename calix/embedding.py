"""Deterministic 768-d text embedding.

The paper specifies a deterministic 768-dimensional text embedding for node
features, memory items and candidate-policy encoding psi(.). To keep the
reference implementation fully reproducible without downloading external
embedding models, we use a seeded hashing-trick embedder: each whitespace /
punctuation token is mapped (via SHA-256) to ``hash_bins`` random projection
directions, summed and L2-normalised. The mapping is platform-independent and
identical across runs, which is all the CALIX pipeline requires.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class DeterministicEmbedder:
    """Seeded hashing embedder producing L2-normalised vectors."""

    def __init__(self, dim: int = 768, hash_bins: int = 4096, seed: int = 0):
        self.dim = dim
        self.hash_bins = hash_bins
        self.seed = seed
        # One fixed random projection matrix shared by all tokens.  Generated
        # from ``seed`` so it is identical on every machine and every run.
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((hash_bins, dim)).astype(np.float64)
        self._proj /= np.linalg.norm(self._proj, axis=1, keepdims=True)

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{token}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.hash_bins

    def embed(self, text: str) -> np.ndarray:
        tokens = _TOKEN_RE.findall(text.lower())
        vec = np.zeros(self.dim, dtype=np.float64)
        if not tokens:
            return vec
        counts: dict[int, float] = {}
        for tok in tokens:
            b = self._bucket(tok)
            counts[b] = counts.get(b, 0.0) + 1.0
        for b, c in counts.items():
            vec += c * self._proj[b]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
