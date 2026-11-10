"""Prepare curated real train/test splits for RoadTwin-GAN."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ARTIFACTS_DIR, DEFAULT_SEED, ensure_directories
from ml.dataset_builder import prepare_curated_dataset, save_dataset_summary


def main() -> None:
    ensure_directories()
    summary = prepare_curated_dataset(seed=DEFAULT_SEED)
    save_dataset_summary(summary, ARTIFACTS_DIR)
    print("Prepared curated dataset:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
