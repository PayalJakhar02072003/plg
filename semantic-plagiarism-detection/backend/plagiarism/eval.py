"""Lightweight baselines for evaluation chapters (Jaccard, TF-IDF cosine)."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


def _tokens(text: str) -> set[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return {t for t in text.split() if t}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a, b = _tokens(text_a), _tokens(text_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def tfidf_cosine_whole_documents(text_a: str, text_b: str) -> float:
    """Cosine similarity of TF-IDF vectors for two full strings (bag-of-words style)."""
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b")
    m = vec.fit_transform([text_a, text_b])
    sim = sk_cosine(m[0:1], m[1:2])[0, 0]
    return float(np.clip(sim, 0.0, 1.0))
