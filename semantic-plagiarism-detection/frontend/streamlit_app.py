"""Frontend: Streamlit UI. Backend (embeddings, scoring) lives under ../backend/plagiarism."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Project root = parent of frontend/
_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _ROOT / "backend"
_FRONTEND = _ROOT / "frontend"
for p in (_BACKEND, _FRONTEND):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from components.controls import render_sidebar
from components.result_view import render_result


@st.cache_resource
def load_embedder(model_name: str):
    """Load model only when needed (heavy deps: torch, transformers, sentence-transformers)."""
    from plagiarism.embedder import SentenceEmbedder

    return SentenceEmbedder(model_name=model_name)


def main() -> None:
    st.set_page_config(page_title="Semantic similarity", layout="wide")
    st.title("Semantic document similarity")
    st.caption("Frontend (Streamlit) · Backend (Python) — decision support only.")

    settings = render_sidebar()

    col1, col2 = st.columns(2)
    with col1:
        text_a = st.text_area("Document A", height=220, placeholder="Paste first text…")
    with col2:
        text_b = st.text_area("Document B", height=220, placeholder="Paste second text…")

    run = st.button("Compare", type="primary")
    if not run:
        return
    if not text_a.strip() or not text_b.strip():
        st.warning("Enter non-empty text in both boxes.")
        return

    from plagiarism.detector import SemanticPlagiarismDetector

    with st.spinner("Loading model (first run may download weights)…"):
        embedder = load_embedder(settings["model"])
    detector = SemanticPlagiarismDetector(embedder, aggregation=settings["aggregation"])

    with st.spinner("Encoding and scoring…"):
        result = detector.compare(text_a, text_b, top_k_pairs=settings["top_k_pairs"])

    render_result(result)


# Streamlit executes this file on each run; do not gate on __name__ == "__main__"
# (Cloud and `streamlit run` both expect the UI to start from top-level execution.)
main()
