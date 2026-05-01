"""Sentence-Transformer loading and batch encoding."""

from __future__ import annotations

import numpy as np

from plagiarism.config import DEFAULT_MODEL_NAME


class SentenceEmbedder:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        name = model_name or DEFAULT_MODEL_NAME
        self.model_name = name
        kwargs = {}
        if device:
            kwargs["device"] = device
        self._model = SentenceTransformer(name, **kwargs)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())
