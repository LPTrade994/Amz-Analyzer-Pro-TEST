"""Utility functions for loading and parsing data files."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import duckdb
import streamlit as st


def load_data(uploaded_file: Any) -> Optional[pd.DataFrame]:
    """Load a CSV or XLSX file into a pandas DataFrame.

    This function imports pandas lazily to avoid requiring it at import time.
    It returns ``None`` if ``uploaded_file`` is falsy.
    """
    if not uploaded_file:
        return None
    import pandas as pd  # imported lazily

    fname = uploaded_file.name.lower()
    if fname.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, dtype=str)
    else:
        try:
            df = pd.read_csv(uploaded_file, sep=";", dtype=str)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",", dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


@st.cache_data(show_spinner=False)
def load_keepa(path: str | Path) -> pd.DataFrame:
    """Load a Keepa export file from ``path``."""
    df = pd.read_excel(path, dtype=str)
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


@st.cache_data(show_spinner=False)
def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a price CSV/Excel file from ``path``."""
    if str(path).lower().endswith(".xlsx"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, sep=";", dtype=str)
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def merge_data(df_keepa: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """Merge data using DuckDB for large datasets."""
    if len(df_keepa) > 100_000:
        return duckdb.query_df(
            {"k": df_keepa, "p": df_prices}, "k", "SELECT * FROM k JOIN p USING(ASIN)"
        ).to_df()
    return pd.merge(df_keepa, df_prices, on="ASIN", how="inner")


def parse_float(value: Any) -> float:
    """Convert a string to ``float`` handling common number formats."""
    if not isinstance(value, str):
        return math.nan
    cleaned = value.replace("€", "").replace(",", ".").strip()
    try:
        return float(cleaned)
    except Exception:
        return math.nan


def parse_int(value: Any) -> Optional[int]:
    """Convert a string to ``int``; returns ``math.nan`` on failure."""
    if not isinstance(value, str):
        return math.nan
    try:
        return int(value.strip())
    except Exception:
        return math.nan


def parse_weight(text: Any) -> float:
    """Extract weight in kilograms from various textual formats."""
    if not isinstance(text, str):
        return math.nan

    kg_match = re.search(r"(\d+\.?\d*)\s*kg", text.lower())
    g_match = re.search(r"(\d+\.?\d*)\s*g", text.lower())

    if kg_match:
        return float(kg_match.group(1))
    if g_match:
        return float(g_match.group(1)) / 1000

    numeric_match = re.fullmatch(r"\s*(\d+\.?\d*)\s*", text)
    if numeric_match:
        try:
            return float(numeric_match.group(1))
        except ValueError:
            return math.nan
    return math.nan
