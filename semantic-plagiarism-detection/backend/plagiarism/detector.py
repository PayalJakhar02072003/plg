"""Orchestrates preprocessing, embeddings, similarity, and verdict."""

from __future__ import annotations

from dataclasses import dataclass

from plagiarism.config import AggregationMode, verdict_display, verdict_label
from plagiarism.embedder import SentenceEmbedder
from plagiarism import preprocess
from plagiarism import similarity


@dataclass
class PlagiarismResult:
    """Outcome of comparing two texts."""

    score: float
    similarity_percent: float
    verdict_key: str
    verdict_text: str
    chunks_a: list[str]
    chunks_b: list[str]
    aggregation: str
    model_name: str
    top_pairs: list[tuple[float, str, str]]
    matrix_shape: tuple[int, int]

    @property
    def disclaimer(self) -> str:
        return (
            "This score measures semantic similarity only. It is not proof of plagiarism "
            "and can be high for same-topic text, templates, or common phrases."
        )


class SemanticPlagiarismDetector:
    def __init__(
        self,
        embedder: SentenceEmbedder,
        aggregation: AggregationMode | str = AggregationMode.MAX,
    ) -> None:
        self.embedder = embedder
        if isinstance(aggregation, str):
            self.aggregation = AggregationMode(aggregation)
        else:
            self.aggregation = aggregation

    def compare(
        self,
        text_a: str,
        text_b: str,
        top_k_pairs: int = 5,
    ) -> PlagiarismResult:
        chunks_a = preprocess.chunk_document(text_a)
        chunks_b = preprocess.chunk_document(text_b)

        emb_a = self.embedder.encode(chunks_a)
        emb_b = self.embedder.encode(chunks_b)
        mat = similarity.cosine_similarity_matrix(emb_a, emb_b)

        if self.aggregation == AggregationMode.MAX:
            primary = similarity.score_max(mat)
        elif self.aggregation == AggregationMode.MEAN:
            primary = similarity.score_mean_pooled_cosine(emb_a, emb_b)
        elif self.aggregation == AggregationMode.MEAN_TOP3:
            primary = similarity.score_mean_top_k(mat, k=3)
        else:
            primary = similarity.score_max(mat)

        primary = max(0.0, min(1.0, float(primary)))
        pairs = similarity.top_aligned_pairs(mat, chunks_a, chunks_b, top_k=top_k_pairs)

        return PlagiarismResult(
            score=primary,
            similarity_percent=round(100.0 * primary, 2),
            verdict_key=verdict_label(primary),
            verdict_text=verdict_display(primary),
            chunks_a=chunks_a,
            chunks_b=chunks_b,
            aggregation=self.aggregation.value,
            model_name=self.embedder.model_name,
            top_pairs=pairs,
            matrix_shape=(int(mat.shape[0]), int(mat.shape[1])),
        )
