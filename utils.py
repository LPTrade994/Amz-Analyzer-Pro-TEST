"""Utility helpers used across the application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

PRESET_DIR = Path(".streamlit/score_presets")
PRESET_DIR.mkdir(parents=True, exist_ok=True)


def save_preset(name: str, weights: Dict[str, float]) -> None:
    """Save ``weights`` dictionary to ``name.json`` inside ``PRESET_DIR``."""
    path = PRESET_DIR / f"{name}.json"
    path.write_text(json.dumps(weights), encoding="utf-8")


def load_preset(name: str) -> Dict[str, float]:
    """Load a preset by ``name``."""
    path = PRESET_DIR / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}
