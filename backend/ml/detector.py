"""Road damage CNN detector, training helper, and inference analyzer."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CLASS_NAMES, DEFAULT_SEED, IDX_TO_CLASS, IMAGE_SIZE, RECOMMENDATIONS
from ml.gan import set_global_seed
from ml.preprocessing import (
    encode_image_base64,
    extract_road_features,
    locate_damage_regions,
    preprocess_for_model,
)


class RoadDamageCNN(nn.Module):
    """Lightweight CNN classifier for road damage categories (normal, pothole, crack)."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_detector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 24,
    batch_size: int = 24,
    lr: float = 0.0015,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
) -> RoadDamageCNN:
    """Train a RoadDamageCNN classifier on the provided (X_train, y_train) split."""
    set_global_seed(seed)
    torch_device = torch.device(device)

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    model = RoadDamageCNN(num_classes=len(CLASS_NAMES)).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(torch_device)
            batch_y = batch_y.to(torch_device)

            # Random horizontal flip during training improves generalization across road angles.
            if torch.rand(1).item() < 0.5:
                batch_x = torch.flip(batch_x, dims=[3])

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def save_detector(model: RoadDamageCNN, output_path: Path) -> Path:
    """Save RoadDamageCNN weights to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": len(CLASS_NAMES)}, output_path)
    return output_path


def load_detector(weights_path: Path, device: str = "cpu") -> RoadDamageCNN:
    """Load RoadDamageCNN weights from disk."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = RoadDamageCNN(num_classes=int(checkpoint.get("num_classes", len(CLASS_NAMES))))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_dataset(model: RoadDamageCNN, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return integer predictions for a numpy batch of preprocessed images."""
    model.eval()
    tensor = torch.from_numpy(X).to(device)
    logits = model(tensor)
    return torch.argmax(logits, dim=1).cpu().numpy()


def _compute_severity(damage_type: str, confidence: float, affected_area_pct: float) -> str:
    """Map damage category, confidence, and affected surface area into civic severity tier."""
    if damage_type == "normal":
        return "None"
    if affected_area_pct >= 14.0 or (damage_type == "pothole" and confidence >= 0.85 and affected_area_pct >= 7.0):
        return "High"
    if affected_area_pct >= 4.5 or confidence >= 0.7:
        return "Moderate"
    return "Low"


@torch.no_grad()
def analyze_road_image(
    bgr_img: np.ndarray,
    model: RoadDamageCNN,
    model_type: str = "augmented",
    device: str = "cpu",
) -> dict[str, object]:
    """Run road damage classification, OpenCV region localization, and civic recommendation."""
    started = time.perf_counter()
    tensor = preprocess_for_model(bgr_img, image_size=IMAGE_SIZE).to(device)

    model.eval()
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])

    boxes, affected_pct, annotated_bgr = locate_damage_regions(bgr_img, pred_class, confidence)
    severity = _compute_severity(pred_class, confidence, affected_pct)
    recommendation = RECOMMENDATIONS[pred_class][severity]
    features = extract_road_features(bgr_img)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

    return {
        "model_type": model_type,
        "damage_type": pred_class,
        "confidence": round(confidence, 4),
        "class_probabilities": {
            name: round(float(probs[idx]), 4) for idx, name in enumerate(CLASS_NAMES)
        },
        "severity": severity,
        "affected_area_pct": affected_pct,
        "bounding_boxes": boxes,
        "surface_features": features,
        "recommendation": recommendation,
        "annotated_image_base64": encode_image_base64(annotated_bgr),
        "inference_ms": elapsed_ms,
    }
