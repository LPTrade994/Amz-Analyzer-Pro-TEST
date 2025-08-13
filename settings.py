"""Application constants and defaults."""

from __future__ import annotations

VAT_RATES = {
    "IT": 22,
    "DE": 19,
    "FR": 20,
    "ES": 21,
    "UK": 20,
}

SHIPPING_TABLE = {
    3: 5.14,
    4: 6.41,
    5: 6.95,
    10: 8.54,
    25: 12.51,
    50: 21.66,
    100: 34.16,
}

SLIDER_DEFAULTS = {
    "alpha": 1.0,
    "beta": 1.0,
    "delta": 1.0,
    "epsilon": 3.0,
    "zeta": 1.0,
    "gamma": 2.0,
    "theta": 1.5,
    "min_margin_multiplier": 1.2,
}
