"""Image preprocessing, feature extraction, and OpenCV damage localization."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CLASS_NAMES, CLASS_TO_IDX, IMAGE_SIZE, PREVIEW_SIZE

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR uint8 OpenCV image."""
    if not raw_bytes:
        raise ValueError("Empty image payload received.")
    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode image. Supported formats: PNG, JPG, BMP, WEBP.")
    return bgr


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode a base64 string (with or without data URL prefix) into a BGR image."""
    if not b64_string or not isinstance(b64_string, str):
        raise ValueError("Invalid base64 image string.")
    payload = b64_string.split(",", 1)[1] if "," in b64_string else b64_string
    try:
        raw_bytes = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("Malformed base64 image payload.") from exc
    return decode_image_bytes(raw_bytes)


def encode_image_base64(bgr_img: np.ndarray, ext: str = ".png") -> str:
    """Encode a BGR uint8 image into a base64 data URI."""
    ok, encoded = cv2.imencode(ext, bgr_img)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def bgr_to_normalized_chw(bgr_img: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize BGR image, convert to RGB, and normalize pixel values to [-1, 1] CHW float32."""
    resized = cv2.resize(bgr_img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb / 127.5) - 1.0
    return np.transpose(normalized, (2, 0, 1))


def tensor_to_bgr(tensor: torch.Tensor, target_size: int = PREVIEW_SIZE) -> np.ndarray:
    """Convert a normalized [-1, 1] CHW RGB tensor into a BGR uint8 image."""
    arr = tensor.detach().cpu().clamp(-1.0, 1.0).numpy()
    rgb = ((np.transpose(arr, (1, 2, 0)) + 1.0) * 127.5).round().astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if target_size and (bgr.shape[0] != target_size or bgr.shape[1] != target_size):
        bgr = cv2.resize(bgr, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return bgr


def preprocess_for_model(bgr_img: np.ndarray, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Convert a BGR image into a batched (1, 3, H, W) tensor ready for PyTorch inference."""
    chw = bgr_to_normalized_chw(bgr_img, image_size=image_size)
    return torch.from_numpy(chw).unsqueeze(0)


def extract_road_features(bgr_img: np.ndarray) -> dict[str, float]:
    """Compute OpenCV surface metrics (edge density, texture roughness, dark depression ratio)."""
    resized = cv2.resize(bgr_img, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 45, 130)
    edge_density = float(np.mean(edges > 0))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Dark regions relative to median asphalt intensity indicate potholes or deep fissures.
    median_val = float(np.median(gray))
    dark_mask = gray < max(20.0, median_val * 0.62)
    dark_ratio = float(np.mean(dark_mask))

    return {
        "mean_intensity": round(float(np.mean(gray)), 2),
        "edge_density": round(edge_density, 4),
        "texture_variance": round(laplacian_var, 2),
        "dark_depression_ratio": round(dark_ratio, 4),
    }


def _find_damage_contours(gray: np.ndarray, damage_type: str) -> list[np.ndarray]:
    """Extract candidate contours tailored to potholes vs pavement cracks."""
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if damage_type == "pothole":
        median_val = float(np.median(blurred))
        thresh_val = int(max(25, min(140, median_val * 0.68)))
        _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        min_area = max(36.0, 0.004 * h * w)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 18, 255, cv2.THRESH_BINARY)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        min_area = max(24.0, 0.002 * h * w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]


def locate_damage_regions(
    bgr_img: np.ndarray,
    predicted_class: str,
    confidence: float,
) -> tuple[list[dict[str, float | int | str]], float, np.ndarray]:
    """Identify bounding boxes for detected damage and render an annotated preview image."""
    preview = cv2.resize(bgr_img, (256, 256), interpolation=cv2.INTER_AREA)
    annotated = preview.copy()

    if predicted_class == "normal":
        cv2.rectangle(annotated, (4, 4), (251, 251), (46, 204, 113), 2)
        cv2.putText(
            annotated,
            f"NORMAL ({confidence * 100:.1f}%)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (46, 204, 113),
            2,
            cv2.LINE_AA,
        )
        return [], 0.0, annotated

    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    contours = _find_damage_contours(gray, predicted_class)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    color = (0, 69, 255) if predicted_class == "pothole" else (0, 191, 255)
    boxes: list[dict[str, float | int | str]] = []
    total_area = 0.0
    img_area = float(preview.shape[0] * preview.shape[1])

    for idx, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        area = float(w * h)
        total_area += area
        boxes.append(
            {
                "id": idx,
                "label": predicted_class,
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area_pct": round((area / img_area) * 100.0, 2),
            }
        )
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated,
            f"{predicted_class.upper()} #{idx}",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if not boxes:
        # Fallback highlight around center region if classifier predicted damage without crisp edges.
        cv2.rectangle(annotated, (36, 36), (220, 220), color, 2)
        boxes.append(
            {
                "id": 1,
                "label": predicted_class,
                "x": 36,
                "y": 36,
                "width": 184,
                "height": 184,
                "area_pct": 12.0,
            }
        )
        total_area = 0.12 * img_area

    affected_pct = min(100.0, round((total_area / img_area) * 100.0, 2))
    cv2.putText(
        annotated,
        f"{predicted_class.upper()} ({confidence * 100:.1f}%)",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return boxes, affected_pct, annotated


def load_labeled_images(
    root_dir: Path,
    image_size: int = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed (N, 3, H, W) images and integer labels from class subdirectories."""
    images: list[np.ndarray] = []
    labels: list[int] = []

    for class_name in CLASS_NAMES:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue
        label_idx = CLASS_TO_IDX[class_name]
        for file_path in sorted(class_dir.iterdir()):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            bgr = cv2.imread(str(file_path))
            if bgr is None:
                continue
            images.append(bgr_to_normalized_chw(bgr, image_size=image_size))
            labels.append(label_idx)

    if not images:
        return np.empty((0, 3, image_size, image_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(images, axis=0).astype(np.float32), np.array(labels, dtype=np.int64)
