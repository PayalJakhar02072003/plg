"""L2-normalized cosine similarity and aggregation helpers."""

from __future__ import annotations

import numpy as np


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of a (n×d) and b (m×d). Returns n×m."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ah = l2_normalize_rows(a)
    bh = l2_normalize_rows(b)
    return (ah @ bh.T).astype(np.float32)


def score_max(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.max(matrix))


def score_mean_pooled_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between mean vectors of a and b."""
    if a.size == 0 or b.size == 0:
        return 0.0
    va = np.mean(a, axis=0, keepdims=True)
    vb = np.mean(b, axis=0, keepdims=True)
    m = cosine_similarity_matrix(va, vb)
    return float(m[0, 0])


def score_mean_top_k(matrix: np.ndarray, k: int = 3) -> float:
    """Mean of the top-k entries in the pairwise matrix (flattened)."""
    if matrix.size == 0:
        return 0.0
    flat = matrix.reshape(-1)
    k = max(1, min(k, flat.size))
    idx = np.argpartition(-flat, k - 1)[:k]
    return float(np.mean(flat[idx]))


def top_aligned_pairs(
    matrix: np.ndarray,
    chunks_a: list[str],
    chunks_b: list[str],
    top_k: int = 5,
) -> list[tuple[float, str, str]]:
    """Return (score, sentence_a, sentence_b) for highest chunk pairs."""
    if matrix.size == 0:
        return []
    flat = matrix.reshape(-1)
    n, m = matrix.shape
    k = min(top_k, flat.size)
    if k <= 0:
        return []
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    out: list[tuple[float, str, str]] = []
    for linear in idx:
        i, j = divmod(int(linear), m)
        score = float(matrix[i, j])
        out.append((score, chunks_a[i], chunks_b[j]))
    return out
