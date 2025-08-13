import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from score import VAT_RATES, normalize_locale, calc_final_purchase_price


def test_purchase_price_de():
    row = {"Price_Base": 100, "Locale (base)": "DE"}
    expected = 100 / (1 + 0.19) * (1 - 0.21)
    assert math.isclose(calc_final_purchase_price(row, 0.21), expected, rel_tol=1e-6)


def test_purchase_price_it():
    row = {"Price_Base": 100, "Locale (base)": "IT"}
    expected = 100 / (1 + 0.22) - 100 * 0.21
    assert math.isclose(calc_final_purchase_price(row, 0.21), expected, rel_tol=1e-6)


def test_locale_variants():
    for variant in ["Amazon.de", "de-DE"]:
        row = {"Price_Base": 100, "Locale (base)": variant}
        expected = 100 / (1 + 0.19) * (1 - 0.21)
        assert math.isclose(
            calc_final_purchase_price(row, 0.21), expected, rel_tol=1e-6
        )
    for variant in ["Amazon.it", "it_IT"]:
        row = {"Price_Base": 100, "Locale (base)": variant}
        expected = 100 / (1 + 0.22) - 100 * 0.21
        assert math.isclose(
            calc_final_purchase_price(row, 0.21), expected, rel_tol=1e-6
        )
