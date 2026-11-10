"""Train baseline vs GAN-augmented detectors and run comparative evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ARTIFACTS_DIR, DEFAULT_SEED, ensure_directories
from ml.dataset_builder import load_train_and_test_arrays
from ml.detector import predict_dataset, save_detector, train_detector
from ml.evaluation import generate_comparative_evaluation


def main() -> None:
    ensure_directories()
    arrays = load_train_and_test_arrays()
    X_real_train, y_real_train = arrays["real_train"]
    X_aug_train, y_aug_train = arrays["augmented_train"]
    X_test, y_test = arrays["real_test"]

    if len(X_real_train) == 0 or len(X_aug_train) == 0 or len(X_test) == 0:
        raise RuntimeError("Missing dataset splits. Run prepare_data.py and train_gan.py first.")

    print("Training baseline detector (before GAN augmentation)...")
    baseline_model = train_detector(X_real_train, y_real_train, epochs=26, seed=DEFAULT_SEED)
    save_detector(baseline_model, ARTIFACTS_DIR / "detector_baseline.pt")

    print("Training augmented detector (after GAN augmentation)...")
    augmented_model = train_detector(X_aug_train, y_aug_train, epochs=26, seed=DEFAULT_SEED)
    save_detector(augmented_model, ARTIFACTS_DIR / "detector_augmented.pt")

    baseline_preds = predict_dataset(baseline_model, X_test)
    augmented_preds = predict_dataset(augmented_model, X_test)

    dataset_summary_path = ARTIFACTS_DIR / "dataset_summary.json"
    dataset_summary = (
        json.loads(dataset_summary_path.read_text(encoding="utf-8"))
        if dataset_summary_path.exists()
        else {}
    )
    gan_history_path = ARTIFACTS_DIR / "gan_history.json"
    gan_history = (
        json.loads(gan_history_path.read_text(encoding="utf-8"))
        if gan_history_path.exists()
        else []
    )

    report = generate_comparative_evaluation(
        y_test=y_test,
        baseline_preds=baseline_preds,
        augmented_preds=augmented_preds,
        dataset_summary=dataset_summary,
        gan_history=gan_history,
        artifacts_dir=ARTIFACTS_DIR,
    )
    print("Comparative Evaluation Summary:")
    print(
        json.dumps(
            {
                "before_augmentation": {
                    k: report["before_augmentation"][k]
                    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
                },
                "after_augmentation": {
                    k: report["after_augmentation"][k]
                    for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
                },
                "improvement_delta": report["improvement_delta"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
