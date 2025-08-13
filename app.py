# app.py
import streamlit as st

# QUESTO DEVE ESSERE IL PRIMO COMANDO STREAMLIT
# Prima di qualsiasi altra importazione che potrebbe contenere comandi Streamlit
st.set_page_config(
    page_title="Amazon Market Analyzer - Arbitraggio Multi-Mercato",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ora possiamo importare altri moduli
import pandas as pd
import numpy as np
import re
import math
import statistics
import warnings
from typing import Optional, Dict, Any
from io import StringIO

from loaders import (
    load_file,
    load_many,
    parse_float,
    parse_percent,
    parse_int,
    parse_weight,
)
from score import (
    SHIPPING_COSTS,
    VAT_RATES,
    normalize_locale,
    calculate_shipping_cost,
    calc_final_purchase_price,
    format_trend,
    classify_opportunity,
    compute_scores,
    aggregate_opportunities,
)
from utils import load_preset, save_preset
from ui import apply_dark_theme

apply_dark_theme()

# Suppress noisy openpyxl warnings about missing default styles
warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style",
    category=UserWarning,
    module="openpyxl",
)

# Result grid column order
# Only the following columns are displayed in this exact sequence.
DISPLAY_COLS_ORDER = [
    "Locale (base)",
    "Locale (comp)",
    "Title (base)",
    "ASIN",
    "Margine_Stimato",
    "Margine_Netto_%",
    "Margine_Netto",
    "Price_Base",
    "Acquisto_Netto",
    "Shipping_Cost",
    "Price_Comp",
    "Vendita_Netto",
    "Bought_Comp",
    "SalesRank_Comp",
    "Trend",
    "NewOffer_Comp",
    "Opportunity_Score",
    "Opportunity_Class",
    "Volume_Score",
    "Weight_kg",
    "Package: Dimension (cm³) (base)",
    "IVA_Origine",
    "IVA_Confronto",
]


# Helper functions
def float_or_nan(x) -> float:
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (int, float)):
            return float(x)
        s = (
            str(x)
            .strip()
            .replace("%", "")
            .replace("\u202f", "")
            .replace(" ", "")
        )
        s = s.replace(".", "").replace(",", ".") if s.count(",") and s.count(".") <= 1 else s
        return float(s)
    except Exception:
        return float("nan")


def euro_to_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    s = str(x).replace("€", "").strip()
    return float_or_nan(s)


def apply_discounts(price_gross: float, coupon_abs, coupon_pct, business_pct) -> float:
    p = float(price_gross) if math.isfinite(price_gross) else float("nan")
    if not math.isfinite(p):
        return p
    ca = euro_to_float(coupon_abs)
    cp = float_or_nan(coupon_pct)
    bp = float_or_nan(business_pct)
    if math.isfinite(ca) and ca > 0:
        p = max(0.0, p - ca)
    if math.isfinite(cp) and cp > 0:
        p *= (1.0 - cp / 100.0)
    if math.isfinite(bp) and bp > 0:
        p *= (1.0 - bp / 100.0)
    return p


def pick_current_price(row: pd.Series) -> float:
    # Priorità: Buy Box 🚚 > Amazon > New > FBM 🚚 (se presenti)
    candidates = [
        "Buy Box 🚚: Current",
        "Amazon: Current",
        "New: Current",
        "New, 3rd Party FBM 🚚: Current",
    ]
    for c in candidates:
        if c in row:
            v = euro_to_float(row.get(c))
            if math.isfinite(v) and v > 0:
                return v
    return float("nan")


def fair_price_row(row: pd.Series) -> Optional[float]:
    # Mediana robusta delle medie storiche (BB/Amazon/New) 90/180/365
    cols = [
        "Buy Box 🚚: 90 days avg.",
        "Buy Box 🚚: 180 days avg.",
        "Buy Box 🚚: 365 days avg.",
        "Amazon: 90 days avg.",
        "Amazon: 180 days avg.",
        "Amazon: 365 days avg.",
        "New: 90 days avg.",
        "New: 180 days avg.",
        "New: 365 days avg.",
    ]
    vals = [euro_to_float(row.get(c)) for c in cols if c in row]
    vals = [v for v in vals if math.isfinite(v) and v > 0]
    if not vals:
        return float("nan")
    fair = statistics.median(sorted(vals))
    # clamp entro min/max storico BB se disponibili
    low = euro_to_float(row.get("Buy Box 🚚: Lowest"))
    high = euro_to_float(row.get("Buy Box 🚚: Highest"))
    if math.isfinite(low) and fair < low:
        fair = low
    if math.isfinite(high) and fair > high:
        fair = high
    return fair


def get_vat_for_locale(locale_raw: str) -> float:
    # RIUSA la tua mappa IVA se esiste (VAT_RATES + normalize_locale).
    try:
        loc = normalize_locale(locale_raw)
        vat = VAT_RATES.get(loc, VAT_RATES.get("IT", 22.0))
        return float(vat) / 100.0  # ritorno in frazione (0.22)
    except Exception:
        return 0.22


def kg_from_row(row: pd.Series) -> float:
    # Prendi il primo peso disponibile tra più varianti Keepa
    candidates = ["Weight", "Item Weight", "Package: Weight (kg)", "Package: Weight (g)"]
    for c in candidates:
        if c in row and pd.notna(row[c]):
            v = float_or_nan(row[c])
            if " (g)" in c:
                v = v / 1000.0
            if math.isfinite(v) and v > 0:
                return v
    return float("nan")


def estimate_fulfillment_fee(row: pd.Series) -> float:
    # Stima basilare: se è presente una colonna con FBA Pick/Pack usala, altrimenti euristica
    for c in ["FBA Pick & Pack Fee", "FBA Fee", "Fulfillment Fee"]:
        if c in row:
            v = euro_to_float(row[c])
            if math.isfinite(v) and v >= 0:
                return float(v)
    kg = kg_from_row(row)
    if not math.isfinite(kg):
        # fallback generico
        return 3.5
    # euristica semplice per FBM/FBA di massima
    if kg <= 0.25:
        return 3.0
    elif kg <= 0.5:
        return 4.0
    elif kg <= 1:
        return 5.0
    elif kg <= 2:
        return 7.0
    elif kg <= 5:
        return 10.0
    else:
        return 15.0


def demand_score(row: pd.Series) -> float:
    rank_c = float_or_nan(row.get("Sales Rank: Current"))
    rank_90 = float_or_nan(row.get("Sales Rank: 90 days avg."))
    bought = float_or_nan(row.get("Bought in past month"))
    rev_now = float_or_nan(row.get("Reviews: Rating Count"))
    rev_90 = float_or_nan(row.get("Reviews: Rating Count - 90 days avg."))

    def vol(r):
        if not math.isfinite(r) or r <= 0:
            return 0.0
        return max(0.0, min(100.0, 1000.0 / math.log(r + 10.0)))

    base = vol(rank_c)
    if math.isfinite(rank_c) and math.isfinite(rank_90) and rank_c < rank_90:
        base *= 1.10
    if math.isfinite(bought) and bought > 0:
        base += min(30.0, 10.0 * math.log(1.0 + bought))
    if math.isfinite(rev_now) and math.isfinite(rev_90) and rev_now > rev_90:
        base += min(10.0, (rev_now - rev_90) * 0.02)
    return float(max(0.0, min(100.0, base)))


def competition_score(row: pd.Series) -> float:
    offers = float_or_nan(row.get("New Offer Count: Current"))
    amz90 = float_or_nan(row.get("Buy Box: % Amazon 90 days"))
    amz180 = float_or_nan(row.get("Buy Box: % Amazon 180 days"))
    amz = max(
        amz90 if math.isfinite(amz90) else 0.0,
        amz180 if math.isfinite(amz180) else 0.0,
    )
    unq = 100.0 if str(row.get("Buy Box: Unqualified")).strip().lower() == "yes" else 0.0
    off_pen = min(100.0, (offers / 50.0) * 50.0) if math.isfinite(offers) else 0.0
    amz_pen = min(100.0, amz) if math.isfinite(amz) else 0.0
    return float(max(0.0, min(100.0, 0.6 * off_pen + 0.4 * amz_pen + 0.5 * unq)))


def scale_0_100(series: pd.Series) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if not math.isfinite(mn) or not math.isfinite(mx) or mx == mn:
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - mn) * 100.0 / (mx - mn)


@st.cache_data(show_spinner=False)
def compute_historic_deals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rileva "affari storici" su un singolo marketplace:
    - usa il prezzo attuale (Buy Box / Amazon / New: Current) applicando coupon/BD
    - stima un "FairPrice" dalla mediana delle medie storiche (BB/Amazon/New 90/180/365) clampata tra min/max BB
    - calcola il margine potenziale ipotizzando: *compro ora al prezzo scontato* e *rivendo al FairPrice*
      (ricavo netto = FairPrice al netto IVA - referral fee - fulfillment stimato)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    # Prezzo lordo attuale migliore disponibile
    work["PriceNowGross"] = work.apply(pick_current_price, axis=1)

    # Applica coupon/BD al lordo
    work["PriceNowGrossAfterDisc"] = work.apply(
        lambda r: apply_discounts(
            work.at[r.name, "PriceNowGross"],
            r.get("One Time Coupon: Absolute"),
            r.get("One Time Coupon: Percentage"),
            r.get("Business Discount: Percentage"),
        ),
        axis=1,
    )

    # Locale -> IVA
    if "Locale" not in work.columns:
        work["Locale"] = (
            work.get("Locale (comp)", pd.Series("", index=work.index)).fillna("").astype(str)
            + work.get("Locale (base)", pd.Series("", index=work.index)).fillna("").astype(str)
        )
    work["VAT"] = work["Locale"].apply(get_vat_for_locale)

    # Prezzo NETTO di acquisto (compriamo ora al prezzo scontato)
    work["NetPurchase"] = work["PriceNowGrossAfterDisc"] / (1.0 + work["VAT"])

    # Fair price (lordo) + corrispondente netto di vendita
    work["FairPrice"] = work.apply(fair_price_row, axis=1)
    work["NetSaleFair"] = work["FairPrice"] / (1.0 + work["VAT"])

    # Referral fee %: usa la colonna se esiste, altrimenti 0 (fallback semplice)
    work["ReferralFeePct"] = work.get("Referral Fee %", pd.Series(np.nan, index=work.index)).apply(float_or_nan).fillna(0.0)

    # Fulfillment stimato (dipende da peso/ingombro, non dal prezzo)
    work["Fulfillment€"] = work.apply(estimate_fulfillment_fee, axis=1)

    # Proventi netti stimati rivendendo al FairPrice
    work["ReferralFeeFair€"] = work["NetSaleFair"] * (work["ReferralFeePct"] / 100.0)
    work["NetProceedFair€"] = work["NetSaleFair"] - work["ReferralFeeFair€"] - work["Fulfillment€"]
    # Alias per compatibilità UI
    work["NetSale"] = work["NetSaleFair"]
    work["ReferralFee€"] = work["ReferralFeeFair€"]
    work["NetProceed€"] = work["NetProceedFair€"]

    # Sottoprezzo vs fair price (solo per ranking/filtro)
    work["UnderPct"] = (work["FairPrice"] - work["PriceNowGrossAfterDisc"]) / work["FairPrice"]
    work.loc[~np.isfinite(work["UnderPct"]), "UnderPct"] = np.nan

    # Se l'app aveva già Acquisto_Netto (da pipeline arbitraggio), lascialo; in caso contrario usa NetPurchase
    if "Acquisto_Netto" not in work.columns or work["Acquisto_Netto"].isna().all():
        work["Acquisto_Netto"] = work["NetPurchase"]

    # Margine potenziale: (rivendo al fair) - (compro ora)
    work["Marg€"] = work["NetProceedFair€"] - work["Acquisto_Netto"]
    work["Marg%"] = np.where(
        work["Acquisto_Netto"] > 0,
        work["Marg€"] / work["Acquisto_Netto"],
        np.nan,
    )

    # Scoring di supporto
    work["Demand"] = work.apply(demand_score, axis=1)
    vol_candidates = []
    for c in [
        "Buy Box: Standard Deviation 90 days",
        "Buy Box: Standard Deviation 30 days",
        "Buy Box: Standard Deviation 365 days",
    ]:
        if c in work.columns:
            vol_candidates.append(work[c].apply(euro_to_float))
    work["Volatility"] = (
        pd.concat(vol_candidates, axis=1).bfill(axis=1).iloc[:, 0]
        if vol_candidates
        else np.nan
    )
    work["Competition"] = work.apply(competition_score, axis=1)

    # Badge utili
    work["Badge_AMZ_OOS"] = (work.get("Amazon: 90 days OOS", pd.Series(0, index=work.index)).fillna(0) > 0)
    amzbb90 = work.get("Buy Box: % Amazon 90 days", pd.Series(0, index=work.index)).apply(float_or_nan)
    work["Badge_BB_Amazon"] = amzbb90.fillna(0) > 50
    work["Badge_Coupon"] = (
        work.get("One Time Coupon: Absolute", pd.Series(0, index=work.index)).fillna(0).apply(euro_to_float) > 0
    ) | (
        work.get("One Time Coupon: Percentage", pd.Series(0, index=work.index)).fillna(0).apply(float_or_nan) > 0
    ) | (
        work.get("Business Discount: Percentage", pd.Series(0, index=work.index)).fillna(0).apply(float_or_nan) > 0
    )

    # DealScore (0..100) -> pesi fix di default, si può portare in UI se serve
    work["NormUnder"] = scale_0_100(work["UnderPct"])
    work["NormMarg€"] = scale_0_100(work["Marg€"])
    work["NormDemand"] = scale_0_100(work["Demand"])
    work["NormCompInv"] = 100.0 - scale_0_100(work["Competition"].fillna(0))

    work["DealScore"] = (
        0.35 * work["NormUnder"]
        + 0.35 * work["NormMarg€"]
        + 0.20 * work["NormDemand"]
        + 0.10 * work["NormCompInv"]
    )

    # Colonne finali principali
    out_cols = [
        "ASIN", "Title", "Locale",
        "PriceNowGross", "PriceNowGrossAfterDisc",
        "FairPrice", "UnderPct",
        "NetPurchase", "NetSaleFair", "ReferralFeePct", "Fulfillment€",
        "NetProceedFair€", "Acquisto_Netto", "Marg€", "Marg%",
        "Demand", "Volatility", "Competition",
        "Badge_AMZ_OOS", "Badge_BB_Amazon", "Badge_Coupon",
        "DealScore"
    ]
    # Aggiungi colonne presenti
    final_cols = [c for c in out_cols if c in work.columns] + [c for c in ["Brand", "Product Group"] if c in work.columns]
    return work[final_cols].copy()


# ==========================
# UI
# ==========================

st.title("Amazon Market Analyzer")
st.caption("Trova opportunità di arbitraggio tra marketplace e **affari storici** nel singolo mercato.")

# Sidebar - Caricamento file
st.sidebar.header("Carica i dati")
base_file = st.sidebar.file_uploader("Lista di Origine (Keepa CSV/XLSX)", type=["csv", "xlsx"])
comp_files = st.sidebar.file_uploader(
    "Liste di Confronto (Keepa CSV/XLSX, puoi selezionarne più di una)",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

# Sidebar - Parametri
st.sidebar.header("Parametri")
discount_pct = st.sidebar.slider("Sconto (%) su mercato di **origine**", min_value=0.0, max_value=60.0, value=10.0, step=0.5)
include_shipping = st.sidebar.checkbox("Includi Spedizione nel Margine", value=True)
min_margin_pct = st.sidebar.number_input("Margine Netto minimo (%)", value=5.0, step=0.5)
min_margin_eur = st.sidebar.number_input("Margine Netto minimo (€)", value=1.0, step=0.5)

# Preset (ricette)
st.sidebar.header("Preset")
preset_name = st.sidebar.text_input("Nome preset")
cc1, cc2 = st.sidebar.columns(2)
with cc1:
    if st.button("💾 Salva preset") and preset_name.strip():
        save_preset({"discount_pct": discount_pct, "include_shipping": include_shipping, "min_margin_pct": min_margin_pct, "min_margin_eur": min_margin_eur}, preset_name.strip())
with cc2:
    if st.button("📂 Carica preset"):
        data = load_preset()
        if data:
            discount_pct = float(data.get("discount_pct", discount_pct))
            include_shipping = bool(data.get("include_shipping", include_shipping))
            min_margin_pct = float(data.get("min_margin_pct", min_margin_pct))
            min_margin_eur = float(data.get("min_margin_eur", min_margin_eur))
            st.success("Preset caricato.")

# Tabs
tab_main, tab_main2, tab_rank, tab_deals, tab_help = st.tabs([
    "Analisi Opportunità",
    "Dashboard",
    "Classifica Prodotti",
    "Affari Storici",
    "Guida"
])

# Caricamento e merge
df_base = load_file(base_file, label="base") if base_file else None
df_comp = load_many(comp_files) if comp_files else None

if df_base is not None and df_comp is not None and not df_base.empty and not df_comp.empty:
    # Normalizzazioni minime
    df_base["ASIN"] = df_base["ASIN"].astype(str).str.upper().str.strip()
    df_comp["ASIN"] = df_comp["ASIN"].astype(str).str.upper().str.strip()

    # Calcolo colonne prezzo da usare
    price_opts = ["Buy Box 🚚: Current", "Amazon: Current", "New: Current"]
    price_base_col = price_opts[0] if price_opts[0] in df_base.columns else price_opts[1] if price_opts[1] in df_base.columns else price_opts[2]
    price_comp_col = price_opts[0] if price_opts[0] in df_comp.columns else price_opts[1] if price_opts[1] in df_comp.columns else price_opts[2]

    # Prepara dataframe base
    df_base["Locale"] = df_base["Locale"].astype(str)
    df_base["Locale (base)"] = df_base["Locale"]
    df_base["Price_Base"] = df_base[price_base_col].apply(euro_to_float)
    df_base["Weight_kg"] = df_base.apply(kg_from_row, axis=1)

    # Calcolo Acquisto_Netto (logica IVA/sconto su base)
    df_base["IVA_Origine"] = df_base["Locale"].apply(lambda x: VAT_RATES.get(normalize_locale(x), VAT_RATES.get("IT", 22.0)))
    df_base["Acquisto_Netto"] = df_base.apply(lambda r: calc_final_purchase_price(
        r["Price_Base"], normalize_locale(r["Locale (base)"]), discount_pct
    ), axis=1)

    # Confronto
    df_comp["Locale"] = df_comp["Locale"].astype(str)
    df_comp["Locale (comp)"] = df_comp["Locale"]
    df_comp["Price_Comp"] = df_comp[price_comp_col].apply(euro_to_float)

    # Merge su ASIN
    df_merged = pd.merge(
        df_base,
        df_comp,
        on="ASIN",
        how="inner",
        suffixes=(" (base)", " (comp)"),
    )

    # Vendita_Netto (comp)
    df_merged["IVA_Confronto"] = df_merged["Locale (comp)"].apply(lambda x: VAT_RATES.get(normalize_locale(x), VAT_RATES.get("IT", 22.0)))
    df_merged["Vendita_Netto"] = df_merged.apply(
        lambda row: row["Price_Comp"] / (1.0 + df_merged.at[row.name, "IVA_Confronto"] / 100.0),
        axis=1,
    )

    # Shipping (euristica semplice basata su peso, Italia per default)
    if include_shipping:
        df_merged["Shipping_Cost"] = df_merged.apply(
            lambda r: calculate_shipping_cost(r.get("Weight_kg", np.nan), origin=normalize_locale(r.get("Locale (base)", "")), dest=normalize_locale(r.get("Locale (comp)", ""))),
            axis=1,
        )
    else:
        df_merged["Shipping_Cost"] = 0.0

    # Margini e score
    df_merged = compute_scores(
        df_merged,
        include_shipping=include_shipping,
        min_margin_pct=min_margin_pct,
        min_margin_eur=min_margin_eur,
    )

    # Classifica cross-market
    agg = aggregate_opportunities(df_merged)

    # Conserva per altre tab
    st.session_state["full_data"] = df_merged.copy()
    st.session_state["agg"] = agg.copy()

else:
    st.session_state["full_data"] = None
    st.session_state["agg"] = None

# ==========================
# Tab: Analisi Opportunità
# ==========================
with tab_main:
    st.subheader("Analisi Opportunità tra mercati")
    df_finale = st.session_state.get("full_data")
    if df_finale is None or df_finale.empty:
        st.info("Carica una lista di origine e una o più liste di confronto per iniziare.")
    else:
        # Filtri rapidi
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _min_rank = st.number_input("Max Sales Rank (comp)", min_value=0, value=200000)
        with c2:
            _min_bought = st.number_input("Min. Bought in past month", min_value=0, value=5)
        with c3:
            _min_margin = st.number_input("Min. Margine Netto (€)", value=1.0)
        with c4:
            _min_margin_pct = st.number_input("Min. Margine Netto (%)", value=5.0)

        mask = pd.Series(True, index=df_finale.index)
        if "SalesRank_Comp" in df_finale.columns:
            mask &= df_finale["SalesRank_Comp"].fillna(9e12) <= _min_rank
        if "Bought_Comp" in df_finale.columns:
            mask &= df_finale["Bought_Comp"].fillna(0) >= _min_bought
        if "Margine_Netto" in df_finale.columns:
            mask &= df_finale["Margine_Netto"].fillna(-1e9) >= _min_margin
        if "Margine_Netto_%" in df_finale.columns:
            mask &= df_finale["Margine_Netto_%"].fillna(-1e9) >= _min_margin_pct

        view = df_finale[mask].copy()

        # Ordine colonne
        cols_disp = [c for c in DISPLAY_COLS_ORDER if c in view.columns]
        extra = [c for c in view.columns if c not in cols_disp]
        view = view[cols_disp + extra]

        st.dataframe(view, use_container_width=True, height=500)

# ==========================
# Tab: Dashboard
# ==========================
def render_results(df_finale: pd.DataFrame, df_ranked: pd.DataFrame, include_shipping: bool) -> None:
    """Render the dashboard and detailed results grids."""
    #################################
    # Dashboard Interattiva
    #################################
    with tab_main2:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("📊 Dashboard delle Opportunità")

        # Metriche principali
        if not df_finale.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prodotti Trovati", len(df_finale))
            with col2:
                st.metric(
                    "Margine Netto Medio (%)",
                    f"{df_finale['Margine_Netto_%'].mean():.2f}%",
                )
            with col3:
                st.metric(
                    "Margine Netto Medio (€)",
                    f"{df_finale['Margine_Netto'].mean():.2f}€",
                )
            with col4:
                st.metric(
                    "Media Spedizione (€)" + (" (incl.)" if include_shipping else " (escl.)"),
                    f"{df_finale['Shipping_Cost'].mean():.2f}€" if "Shipping_Cost" in df_finale else "—",
                )

        st.markdown("</div>", unsafe_allow_html=True)

        #################################
        # Risultati Dettagliati
        #################################
        st.subheader("📋 Risultati Dettagliati")
        st.dataframe(df_finale, use_container_width=True, height=540)

    #################################
    # Classifica prodotti (cross-market)
    #################################
    with tab_rank:
        st.subheader("Classifica Prodotti (miglior marketplace per ASIN)")
        if df_ranked is None or df_ranked.empty:
            st.info("Carica i file per vedere la classifica.")
        else:
            st.dataframe(df_ranked, use_container_width=True, height=540)

# Render nelle tab
df_finale = st.session_state.get("full_data")
df_ranked = st.session_state.get("agg")
if df_finale is not None and not df_finale.empty:
    render_results(df_finale.copy(), df_ranked.copy() if df_ranked is not None else None, include_shipping)

# ==========================
# Tab: Guida
# ==========================
with tab_help:
    st.subheader("Guida rapida")
    st.markdown(
        """
**Come funziona l'analisi tra mercati**

1. Carica una lista di **origine** e una o più liste di **confronto** (Keepa Product Finder).
2. L'app normalizza `ASIN` e `Locale` e abbina i prodotti.
3. Scegli quali colonne di prezzo usare (di default `Buy Box 🚚: Current`).
4. Imposta **sconto**, **spedizione**, **filtri** minimi di margine.
5. Guarda la tab *Analisi Opportunità* e la *Dashboard* per i dettagli, e la *Classifica Prodotti* per sapere dove conviene vendere.

**IVA & Sconto (riassunto)**
- Normalizzazione `Locale` (es. Amazon.de → DE, GB→UK).
- Per mercati esteri: `Net = Price / (1 + VAT) * (1 - Discount)`.
- Per acquisti in Italia: `Net = Price / (1 + VAT) - Price * Discount`.

**Affari Storici**
- Individua prezzi **sotto valore** rispetto al fair price storico (mediana delle medie 90/180/365, clamp tra min/max BB).
- Applica **coupon** e **Business Discount** prima di togliere l’IVA.
- Stima margine potenziale: **compro ora** (netto) → **rivendo al fair price** (netto – fee referral – fulfillment).
"""
    )

# Tab Affari Storici
df_final = st.session_state.get("full_data")

with tab_deals:
    st.subheader("Affari Storici")

    colw1, colw2, colw3, colw4, colw5 = st.columns(5)
    with colw1:
        min_marg_eur = st.number_input("Min. Margine (€)", value=3.0, step=0.5)
    with colw2:
        min_marg_pct = st.number_input("Min. Margine (%)", value=8.0, step=0.5)
    with colw3:
        min_under = st.number_input("Sottoprezzo vs Fair (min %)", value=5.0, step=0.5)
    with colw4:
        req_coupon = st.checkbox("Solo con Coupon/BD", value=False)
    with colw5:
        sort_key = st.selectbox("Ordina per", ["DealScore", "UnderPct", "Marg€", "Marg%", "Demand"])

    src = df_final if df_final is not None and not df_final.empty else base_file
    if isinstance(src, pd.DataFrame):
        base_df = src.copy()
    elif base_file is not None:
        base_df = load_file(base_file, label="base")
    else:
        base_df = pd.DataFrame()

    deals_df = compute_historic_deals(base_df)

    if deals_df.empty:
        st.info("Nessun dato disponibile per Affari Storici.")
    else:
        vol_series = deals_df["Volatility"].replace([np.inf, -np.inf], np.nan)
        vol_thr = (
            np.nanpercentile(vol_series.dropna(), 75)
            if vol_series.notna().any()
            else np.nan
        )
        if math.isfinite(vol_thr):
            deals_df["Badge_VolHigh"] = deals_df["Volatility"] > vol_thr

        def pct_amz_bb(row):
            s = [
                float_or_nan(row.get("Buy Box: % Amazon 90 days")),
                float_or_nan(row.get("Buy Box: % Amazon 180 days")),
            ]
            return max([x for x in s if math.isfinite(x)] + [0.0])

        mask = pd.Series(True, index=deals_df.index)
        if math.isfinite(min_marg_eur):
            mask &= deals_df["Marg€"].fillna(-1e9) >= min_marg_eur
        if math.isfinite(min_marg_pct):
            mask &= deals_df["Marg%"].fillna(-1e9) >= min_marg_pct
        if math.isfinite(min_under):
            mask &= deals_df["UnderPct"].fillna(-1e9) >= min_under
        if req_coupon:
            mask &= deals_df["Badge_Coupon"] == True

        deals_f = deals_df[mask].copy()
        deals_f = deals_f.sort_values(sort_key, ascending=False, na_position="last")

        st.write(f"**{len(deals_f)}** affari trovati")
        show_cols = [c for c in [
            "Locale",
            "ASIN",
            "Title",
            "PriceNowGrossAfterDisc",
            "FairPrice",
            "UnderPct",
            "NetSale",
            "ReferralFee€",
            "Fulfillment€",
            "NetProceed€",
            "Acquisto_Netto",
            "Marg€",
            "Marg%",
            "Demand",
            "Competition",
            "Volatility",
            "DealScore",
            "Badge_Coupon",
            "Badge_AMZ_OOS",
            "Badge_BB_Amazon",
        ] if c in deals_f.columns]

        # Formattazioni
        disp = deals_f.copy()
        for c in [
            "PriceNowGrossAfterDisc",
            "FairPrice",
            "NetSale",
            "ReferralFee€",
            "Fulfillment€",
            "NetProceed€",
            "Acquisto_Netto",
            "Marg€",
        ]:
            if c in disp.columns:
                disp[c] = disp[c].map(
                    lambda v: f"€ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                )
        if "Marg%" in disp.columns:
            disp["Marg%"] = disp["Marg%"].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")
        if "UnderPct" in disp.columns:
            disp["UnderPct"] = disp["UnderPct"].map(lambda v: f"{v*100:.1f}%" if pd.notna(v) else "")

        st.dataframe(disp[show_cols], use_container_width=True, height=560)

# Footer
st.markdown(
    """
<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #333; color: #aaa;">
    Amazon Market Analyzer - Arbitraggio Multi-Mercato © 2025<br>
    Versione 2.0
</div>
""",
    unsafe_allow_html=True,
)
