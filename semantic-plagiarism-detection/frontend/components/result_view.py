"""Display PlagiarismResult in Streamlit."""

from __future__ import annotations

import streamlit as st

from plagiarism.detector import PlagiarismResult


def render_result(result: PlagiarismResult) -> None:
    st.subheader("Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Similarity index", f"{result.similarity_percent}%")
    c2.metric("Raw score", f"{result.score:.4f}")
    c3.metric("Verdict", result.verdict_text)

    st.info(result.disclaimer)

    with st.expander("Technical details"):
        st.write(
            {
                "model": result.model_name,
                "aggregation": result.aggregation,
                "chunks_a": len(result.chunks_a),
                "chunks_b": len(result.chunks_b),
                "matrix_shape": result.matrix_shape,
            }
        )

    if result.top_pairs:
        st.subheader("Strongest aligned sentences")
        for i, (score, sa, sb) in enumerate(result.top_pairs, start=1):
            st.markdown(f"**{i}.** score `{score:.4f}`")
            st.caption("Document A")
            st.write(sa)
            st.caption("Document B")
            st.write(sb)
            st.divider()
