"""Sidebar controls for model and aggregation."""

from __future__ import annotations

import streamlit as st

from plagiarism.config import DEFAULT_MODEL_NAME, AggregationMode


def render_sidebar() -> dict:
    st.sidebar.header("Settings")
    model = st.sidebar.text_input("Model id", value=DEFAULT_MODEL_NAME, help="Sentence-Transformers model name")
    aggregation = st.sidebar.selectbox(
        "Aggregation",
        options=[m.value for m in AggregationMode],
        format_func=lambda x: {
            AggregationMode.MAX.value: "Max chunk pair (localized)",
            AggregationMode.MEAN.value: "Mean-pooled document vectors",
            AggregationMode.MEAN_TOP3.value: "Mean of top 3 chunk pairs",
        }.get(x, x),
        index=0,
    )
    top_k = st.sidebar.slider("Top aligned pairs to show", 1, 15, 5)
    return {"model": model.strip(), "aggregation": aggregation, "top_k_pairs": int(top_k)}
