"""Streamlit UI helper functions."""

from __future__ import annotations

from pathlib import Path
import streamlit as st


def apply_dark_theme():
    """Inject the custom dark theme CSS used by the dashboard."""
    css = ""
    if st.runtime.exists():
        try:
            css = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
        except FileNotFoundError:
            css = ""
    st.markdown("<style>\n" + css + "\n</style>", unsafe_allow_html=True)
