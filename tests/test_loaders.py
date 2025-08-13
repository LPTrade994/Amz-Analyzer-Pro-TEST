import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from loaders import load_keepa


def test_keepa_columns():
    df = load_keepa("sample_data/keepa_sample.xlsx")
    assert len(df.columns) >= 40
    assert not any(col.startswith("Unnamed") for col in df.columns)
