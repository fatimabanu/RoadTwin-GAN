"""Evaluation metrics (precision, recall, F1) and Matplotlib comparison plots."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import CLASS_NAMES


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Compute overall and per-class precision, recall, F1-score, accuracy, and confusion matrix."""
    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        name: {
            "precision": round(float(report[name]["precision"]), 4),
            "recall": round(float(report[name]["recall"]), 4),
            "f1_score": round(float(report[name]["f1-score"]), 4),
            "support": int(report[name]["support"]),
        }
        for name in CLASS_NAMES
    }

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _plot_metrics_comparison(
    baseline_metrics: dict[str, object],
    augmented_metrics: dict[str, object],
    out_path: Path,
) -> None:
    """Save grouped bar chart comparing detection metrics before vs after GAN augmentation."""
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
    before_vals = [float(baseline_metrics[k]) * 100 for k in metric_keys]
    after_vals = [float(augmented_metrics[k]) * 100 for k in metric_keys]

    df = pd.DataFrame({"Before GAN": before_vals, "After GAN": after_vals}, index=labels)
    ax = df.plot(kind="bar", figsize=(8.2, 5.0), color=["#94a3b8", "#2563eb"], width=0.62)
    ax.set_title("Road Damage Detection: Before vs After GAN Augmentation", fontsize=12, fontweight="bold", pad=22)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 116)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False)
    plt.xticks(rotation=15, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_confusion_matrices(
    baseline_cm: list[list[int]],
    augmented_cm: list[list[int]],
    out_path: Path,
) -> None:
    """Save side-by-side confusion matrices before and after GAN augmentation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    configs = [
        ("Before GAN (Original Scarce Data)", np.array(baseline_cm), axes[0]),
        ("After GAN (Augmented Data)", np.array(augmented_cm), axes[1]),
    ]

    for title, cm, ax in configs:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=20)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                color = "white" if cm[row, col] > threshold else "black"
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", color=color, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_gan_loss_curve(history: list[dict[str, float]], out_path: Path) -> None:
    """Save Generator vs Discriminator loss curve from GAN training history."""
    if not history:
        return
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["epoch"], df["g_loss"], label="Generator Loss", color="#2563eb", linewidth=2)
    ax.plot(df["epoch"], df["d_loss"], label="Discriminator Loss", color="#dc2626", linewidth=2)
    ax.set_title("RoadTwin-GAN Training Loss Progression", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_comparative_evaluation(
    y_test: np.ndarray,
    baseline_preds: np.ndarray,
    augmented_preds: np.ndarray,
    dataset_summary: dict[str, object],
    gan_history: list[dict[str, float]],
    artifacts_dir: Path,
) -> dict[str, object]:
    """Evaluate baseline vs GAN-augmented detectors, render plots, and persist JSON report."""
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = compute_classification_metrics(y_test, baseline_preds)
    augmented_metrics = compute_classification_metrics(y_test, augmented_preds)

    deltas = {
        key: round(float(augmented_metrics[key]) - float(baseline_metrics[key]), 4)
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    }

    _plot_metrics_comparison(baseline_metrics, augmented_metrics, plots_dir / "metrics_comparison.png")
    _plot_confusion_matrices(
        baseline_metrics["confusion_matrix"],
        augmented_metrics["confusion_matrix"],
        plots_dir / "confusion_matrices.png",
    )
    plot_gan_loss_curve(gan_history, plots_dir / "gan_loss_curve.png")

    report: dict[str, object] = {
        "classes": list(CLASS_NAMES),
        "dataset_summary": dataset_summary,
        "before_augmentation": baseline_metrics,
        "after_augmentation": augmented_metrics,
        "improvement_delta": deltas,
        "gan_training_history": gan_history,
        "plots": {
            "metrics_comparison": "/api/evaluation/plots/metrics_comparison.png",
            "confusion_matrices": "/api/evaluation/plots/confusion_matrices.png",
            "gan_loss_curve": "/api/evaluation/plots/gan_loss_curve.png",
        },
    }

    report_path = artifacts_dir / "evaluation_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
