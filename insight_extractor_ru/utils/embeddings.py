"""A lightweight, dependency-free stand-in for a real embedding model.

design.md Decision 9 calls for "in-process embedding cosine similarity, no
vector DB yet" for tag normalization, without committing to a specific
embedding source. This hashed character-n-gram embedding needs no model
download, no network call, and no heavy ML dependency — good enough to
catch lexical near-duplicates ("дорого" / "слишком дорого") via cosine
similarity, though it will not catch paraphrases with no shared substrings.
Kept behind the same swappable interface as everything else in
``aggregation.tag_normalization``, so a real embedding model or API can
replace it later without changing callers.
"""

from __future__ import annotations

import hashlib

import numpy as np

DEFAULT_DIM = 256
DEFAULT_NGRAM = 3


def char_ngram_embedding(
    text: str, *, dim: int = DEFAULT_DIM, n: int = DEFAULT_NGRAM
) -> np.ndarray:
    """Bag-of-character-n-grams, hashed into a fixed-size vector."""
    normalized = text.strip().lower()
    vector = np.zeros(dim, dtype=np.float64)
    if not normalized:
        return vector

    ngrams = (
        [normalized]
        if len(normalized) < n
        else [normalized[i : i + n] for i in range(len(normalized) - n + 1)]
    )
    for ngram in ngrams:
        index = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16) % dim
        vector[index] += 1.0
    return vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
