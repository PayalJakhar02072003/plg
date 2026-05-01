"""Text normalization and sentence chunking."""

from __future__ import annotations

import re
from functools import lru_cache


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@lru_cache(maxsize=1)
def _ensure_nltk_punkt() -> bool:
    try:
        import nltk

        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        return True
    except Exception:
        return False


def split_sentences(text: str) -> list[str]:
    """Split into sentences; prefers NLTK punkt, falls back to simple regex."""
    text = normalize_whitespace(text)
    if not text:
        return []

    if _ensure_nltk_punkt():
        try:
            from nltk.tokenize import sent_tokenize

            parts = sent_tokenize(text)
            out = [p.strip() for p in parts if p.strip()]
            if out:
                return out
        except Exception:
            pass

    # Fallback: split on . ! ? followed by space or end
    rough = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in rough if s.strip()]


def chunk_document(text: str, min_sentence_len: int = 3) -> list[str]:
    """Return non-empty sentence chunks suitable for embedding."""
    chunks = split_sentences(text)
    return [c for c in chunks if len(c) >= min_sentence_len]
