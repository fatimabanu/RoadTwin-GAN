"""Configuration constants and path settings for RoadTwin-GAN backend."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SAMPLES_DIR = ARTIFACTS_DIR / "samples"
DB_PATH = DATA_DIR / "civic_reports.db"

CLASS_NAMES: tuple[str, ...] = ("normal", "pothole", "crack")
CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 64
PREVIEW_SIZE = 192
LATENT_DIM = 64
DEFAULT_SEED = 42

RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "normal": {
        "None": "Road surface appears intact. Continue routine scheduled monitoring.",
    },
    "pothole": {
        "Low": "Minor surface depression detected. Schedule preventive cold-patch inspection.",
        "Moderate": "Clear pothole detected. Dispatch municipal patching crew within 7 days.",
        "High": "Severe pothole hazard detected. Prioritize immediate barricade and hot-mix repair.",
    },
    "crack": {
        "Low": "Hairline surface cracking detected. Monitor during next seasonal inspection.",
        "Moderate": "Moderate pavement cracking detected. Schedule crack sealing to prevent water ingress.",
        "High": "Extensive pavement cracking detected. Schedule resurfacing and structural evaluation.",
    },
}


def ensure_directories() -> None:
    """Ensure runtime data and artifact directories exist."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SYNTHETIC_DATA_DIR,
        AUGMENTED_DATA_DIR,
        ARTIFACTS_DIR,
        PLOTS_DIR,
        SAMPLES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
