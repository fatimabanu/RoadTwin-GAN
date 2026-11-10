"""Run full RoadTwin-GAN pipeline: prepare data, train cDCGAN, and evaluate detectors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import prepare_data, train_detector, train_gan


def main() -> None:
    prepare_data.main()
    train_gan.main()
    train_detector.main()


if __name__ == "__main__":
    main()
