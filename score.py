"""Core scoring utilities used by the dashboard."""

from __future__ import annotations

import math
import re
from typing import Any, Dict

import pandas as pd
import streamlit as st

from settings import SHIPPING_TABLE as SHIPPING_COSTS, VAT_RATES


def normalize_locale(locale_str: Any) -> str:
    """Return a two letter country code from a locale string."""
    if not isinstance(locale_str, str):
        return ""
    s = locale_str.strip().upper()
    matches = re.findall(r"[A-Z]{2}", s)
    code = matches[-1] if matches else ""
    if code == "GB":
        code = "UK"
    return code


def calculate_shipping_cost(weight_kg: Any) -> float:
    """Compute shipping cost from the ``SHIPPING_COSTS`` table."""
    if (
        weight_kg is None
        or (isinstance(weight_kg, float) and math.isnan(weight_kg))
        or weight_kg <= 0
    ):
        return 0.0
    for limit, cost in sorted(SHIPPING_COSTS.items()):
        if weight_kg <= limit:
            return cost
    return SHIPPING_COSTS[100]


def calc_final_purchase_price(row: Dict[str, Any], discount: float) -> float:
    """Return the net purchase price for a row of data."""
    gross = row.get("Price_Base")
    if gross is None or (isinstance(gross, float) and math.isnan(gross)):
        return math.nan
    locale = normalize_locale(row.get("Locale (base)", ""))
    vat_rate = VAT_RATES.get(locale, 0) / 100.0
    net_price = gross / (1 + vat_rate)
    if locale == "IT":
        discount_amount = gross * discount
        final_price = net_price - discount_amount
    else:
        final_price = net_price * (1 - discount)
    return max(final_price, 0)


def format_trend(trend: Any) -> str:
    """Return a textual representation for a trend value."""
    if trend is None or (isinstance(trend, float) and math.isnan(trend)):
        return "N/D"
    if trend > 0.1:
        return "🔼 Crescente"
    if trend < -0.1:
        return "🔽 Decrescente"
    return "➖ Stabile"


def classify_opportunity(score: float):
    """Return a textual class and tag for a given opportunity score."""
    if score > 100:
        return "Eccellente", "success-tag"
    if score > 50:
        return "Buona", "success-tag"
    if score > 20:
        return "Discreta", "warning-tag"
    return "Bassa", "danger-tag"


def _minmax(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def margin_score(df: pd.DataFrame) -> pd.Series:
    return _minmax(df["Margine_Netto_%"].fillna(0))


def demand_score(df: pd.DataFrame) -> pd.Series:
    return 1 - _minmax(df["SalesRank_Comp"].fillna(df["SalesRank_Comp"].max()))


def competition_score(df: pd.DataFrame) -> pd.Series:
    return 1 - _minmax(df["NewOffer_Comp"].fillna(df["NewOffer_Comp"].max()))


def volatility_score(df: pd.DataFrame) -> pd.Series:
    return _minmax(df["Trend_Bonus"].fillna(0))


def risk_score(df: pd.DataFrame) -> pd.Series:
    return _minmax(df["ROI_Factor"].fillna(0))


@st.cache_data(show_spinner=False)
def compute_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Return ``df`` with a normalized opportunity score."""
    df = df.copy()
    subs = {
        "margin": margin_score(df),
        "demand": demand_score(df),
        "competition": competition_score(df),
        "volatility": volatility_score(df),
        "risk": risk_score(df),
    }
    score = sum(weights.get(k, 1.0) * subs[k] for k in subs)
    df["final_score"] = _minmax(score) * 100
    return df


def aggregate_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per ASIN with the best market and score."""
    if df is None or df.empty or "ASIN" not in df.columns:
        return pd.DataFrame(columns=["ASIN", "Best_Market", "Opportunity_Score"])

    if "Opportunity_Score" not in df.columns:
        return pd.DataFrame(columns=["ASIN", "Best_Market", "Opportunity_Score"])

    idx = df.groupby("ASIN") ["Opportunity_Score"].idxmax()
    best = df.loc[idx].copy()
    best = best.rename(columns={"Locale (comp)": "Best_Market"})

    cols = ["ASIN"]
    if "Title (base)" in best.columns:
        cols.append("Title (base)")
    cols += ["Best_Market", "Opportunity_Score"]

    return best[cols].sort_values("Opportunity_Score", ascending=False).reset_index(drop=True)
