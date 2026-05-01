"""Defaults for models, aggregation, and demo verdict bands."""

from enum import Enum

# Sentence-Transformers model id (384-d MiniLM; widely used baseline)
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Demo verdict bands on primary cosine score — calibrate on your labeled data for real use
VERDICT_HIGH_MIN = 0.75
VERDICT_MODERATE_MIN = 0.45


class AggregationMode(str, Enum):
    """How to turn chunk–chunk similarities into one document-level score."""

    MAX = "max"  # best localized match (default in plan)
    MEAN = "mean"  # mean-pooled embeddings, then cosine
    MEAN_TOP3 = "mean_top3"  # mean of top 3 pairwise chunk scores (excluding diagonal if same doc)


def verdict_label(score: float) -> str:
    if score >= VERDICT_HIGH_MIN:
        return "high_similarity"
    if score >= VERDICT_MODERATE_MIN:
        return "moderate_similarity"
    return "low_similarity"


def verdict_display(score: float) -> str:
    """Human-readable band for UI."""
    if score >= VERDICT_HIGH_MIN:
        return "High similarity"
    if score >= VERDICT_MODERATE_MIN:
        return "Moderate similarity"
    return "Low similarity"
