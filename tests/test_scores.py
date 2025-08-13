import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from loaders import load_keepa
import pandas as pd
from score import (
    margin_score,
    demand_score,
    competition_score,
    volatility_score,
    risk_score,
    aggregate_opportunities,
)


def test_subscores_range():
    df = load_keepa("sample_data/keepa_sample.xlsx")
    # minimal columns needed for subscore functions
    df = df.rename(columns={
        "Sales Rank: Current": "SalesRank_Comp",
        "New Offer Count: Current": "NewOffer_Comp",
    })
    df["SalesRank_Comp"] = pd.to_numeric(df["SalesRank_Comp"], errors="coerce")
    df["NewOffer_Comp"] = pd.to_numeric(df["NewOffer_Comp"], errors="coerce")
    df["Margine_Netto_%"] = 0.0
    df["Trend_Bonus"] = 0.0
    df["ROI_Factor"] = 0.0

    for func in [margin_score, demand_score, competition_score, volatility_score, risk_score]:
        scores = func(df)
        assert ((0.0 <= scores) & (scores <= 1.0)).all()


def test_aggregate_opportunities():
    df = pd.DataFrame(
        {
            "ASIN": ["A1", "A1", "A2"],
            "Opportunity_Score": [10, 20, 15],
            "Locale (comp)": ["DE", "FR", "IT"],
        }
    )
    agg = aggregate_opportunities(df)
    assert len(agg) == 2
    a1 = agg[agg["ASIN"] == "A1"].iloc[0]
    assert a1["Opportunity_Score"] == 20
    assert a1["Best_Market"] == "FR"
