"""Train the Conditional DCGAN and synthesize augmented road damage images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ARTIFACTS_DIR, CLASS_NAMES, DEFAULT_SEED, IMAGE_SIZE, PROCESSED_DATA_DIR, ensure_directories
from ml.dataset_builder import build_augmented_dataset, save_dataset_summary
from ml.evaluation import plot_gan_loss_curve
from ml.gan import save_gan_artifacts, train_conditional_gan
from ml.preprocessing import load_labeled_images


def _balance_for_gan(X: np.ndarray, y: np.ndarray, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes via random flips so the conditional GAN trains evenly across all damage types."""
    rng = np.random.default_rng(seed)
    counts = [int(np.sum(y == idx)) for idx in range(len(CLASS_NAMES))]
    max_count = max(counts) if counts else 0

    balanced_x: list[np.ndarray] = []
    balanced_y: list[np.ndarray] = []
    for class_idx in range(len(CLASS_NAMES)):
        mask = y == class_idx
        cls_x = X[mask]
        if len(cls_x) == 0:
            continue
        balanced_x.append(cls_x)
        balanced_y.append(np.full(len(cls_x), class_idx, dtype=np.int64))

        needed = max_count - len(cls_x)
        if needed > 0:
            pick_indices = rng.integers(0, len(cls_x), size=needed)
            extra = cls_x[pick_indices].copy()
            for i in range(needed):
                if rng.random() < 0.5:
                    extra[i] = extra[i, :, :, ::-1]
                if rng.random() < 0.5:
                    extra[i] = extra[i, :, ::-1, :]
            balanced_x.append(extra)
            balanced_y.append(np.full(needed, class_idx, dtype=np.int64))

    return np.concatenate(balanced_x, axis=0), np.concatenate(balanced_y, axis=0)


def main() -> None:
    ensure_directories()
    X_train, y_train = load_labeled_images(PROCESSED_DATA_DIR / "train", image_size=IMAGE_SIZE)
    if len(X_train) == 0:
        raise RuntimeError("No training images found. Run scripts/prepare_data.py first.")

    X_gan, y_gan = _balance_for_gan(X_train, y_train, seed=DEFAULT_SEED)
    generator, history = train_conditional_gan(
        X_gan,
        y_gan,
        epochs=55,
        batch_size=24,
        lr=0.00025,
        seed=DEFAULT_SEED,
        device="cpu",
    )
    weights_path = save_gan_artifacts(generator, history, ARTIFACTS_DIR)
    plot_gan_loss_curve(history, ARTIFACTS_DIR / "plots" / "gan_loss_curve.png")

    aug_summary = build_augmented_dataset(
        generator=generator,
        target_per_class=110,
        seed=DEFAULT_SEED,
        device="cpu",
    )
    save_dataset_summary(aug_summary, ARTIFACTS_DIR)
    print(f"Saved GAN weights to {weights_path}")
    print("Dataset augmentation summary:")
    print(json.dumps(aug_summary, indent=2))


if __name__ == "__main__":
    main()
