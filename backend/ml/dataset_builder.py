"""Dataset curation, procedural starter dataset creation, and GAN augmentation."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
    AUGMENTED_DATA_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    DEFAULT_SEED,
    IMAGE_SIZE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLES_DIR,
    SYNTHETIC_DATA_DIR,
)
from ml.gan import RoadConditionalGenerator, generate_synthetic_tensors
from ml.preprocessing import SUPPORTED_EXTENSIONS, load_labeled_images, tensor_to_bgr


def _create_asphalt_base(rng: np.random.Generator, size: int = 96) -> np.ndarray:
    """Render a realistic asphalt background with aggregate grain and lighting variation."""
    base_tone = rng.integers(68, 128)
    bg = np.full((size, size, 3), base_tone, dtype=np.float32)

    # Subtle color shift for warm/cool pavement lighting
    bg[:, :, 0] += rng.uniform(-6, 6)
    bg[:, :, 1] += rng.uniform(-5, 5)
    bg[:, :, 2] += rng.uniform(-6, 6)

    # Aggregate gravel speckle
    grain = rng.normal(0.0, rng.uniform(7.0, 14.0), size=(size, size, 1)).astype(np.float32)
    fine = rng.normal(0.0, 4.0, size=(size, size, 3)).astype(np.float32)
    img = bg + grain + fine

    # Directional illumination gradient (sun/shadow across road)
    grad_x = np.linspace(rng.uniform(-14, 14), rng.uniform(-14, 14), size, dtype=np.float32)
    grad_y = np.linspace(rng.uniform(-12, 12), rng.uniform(-12, 12), size, dtype=np.float32)
    img += grad_x[None, :, None] + grad_y[:, None, None]

    # Occasional road lane marking on edge
    if rng.random() < 0.25:
        x_line = int(rng.integers(6, 18)) if rng.random() < 0.5 else int(rng.integers(size - 18, size - 6))
        line_color = rng.uniform(175, 225)
        cv2.line(img, (x_line, 0), (x_line + int(rng.integers(-4, 5)), size), (line_color, line_color, line_color), 3)

    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_pothole(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overlay an irregular pothole crater with recessed shadow and spalled rim."""
    h, w, _ = img.shape
    out = img.astype(np.float32)

    cx = int(rng.integers(w // 4, (3 * w) // 4))
    cy = int(rng.integers(h // 4, (3 * h) // 4))
    rx = int(rng.integers(12, 24))
    ry = int(rng.integers(10, 20))

    angles = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    radii_noise = rng.uniform(0.72, 1.25, size=len(angles))
    pts = np.stack(
        [
            cx + rx * radii_noise * np.cos(angles),
            cy + ry * radii_noise * np.sin(angles),
        ],
        axis=1,
    ).astype(np.int32)

    rim_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(rim_mask, [pts], 255)
    rim_dilated = cv2.dilate(rim_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    rim_ring = (rim_dilated > 0) & (rim_mask == 0)

    # Spalled lighter edge around crater
    out[rim_ring] = np.clip(out[rim_ring] + rng.uniform(16, 34), 0, 255)

    # Dark recessed interior with rough gravel depth
    interior = rim_mask > 0
    depth_shade = rng.uniform(0.28, 0.45)
    interior_noise = rng.normal(0.0, 9.0, size=(h, w, 1)).astype(np.float32)
    out = np.where(interior[:, :, None], out * depth_shade + interior_noise, out)

    # Blur boundary slightly for natural asphalt transition
    blurred = cv2.GaussianBlur(out, (3, 3), 0)
    edge_band = cv2.dilate(rim_mask, np.ones((3, 3), np.uint8)) ^ cv2.erode(rim_mask, np.ones((3, 3), np.uint8))
    out = np.where((edge_band > 0)[:, :, None], blurred, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_crack(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overlay realistic jagged longitudinal, transverse, or branching pavement cracks."""
    h, w, _ = img.shape
    out = img.copy()

    num_branches = int(rng.integers(1, 4))
    for _ in range(num_branches):
        horizontal = rng.random() < 0.5
        if horizontal:
            x, y = int(rng.integers(2, 16)), int(rng.integers(16, h - 16))
            dx_step, dy_range = 4, (-4, 5)
            steps = (w - x) // dx_step
        else:
            x, y = int(rng.integers(16, w - 16)), int(rng.integers(2, 16))
            dx_step, dy_range = 0, (3, 6)
            steps = (h - y) // 4

        pts = [(x, y)]
        for _ in range(max(4, steps)):
            if horizontal:
                x = min(w - 2, x + int(rng.integers(3, 6)))
                y = int(np.clip(y + int(rng.integers(*dy_range)), 4, h - 4))
            else:
                x = int(np.clip(x + int(rng.integers(-4, 5)), 4, w - 4))
                y = min(h - 2, y + int(rng.integers(*dy_range)))
            pts.append((x, y))

        thickness = int(rng.integers(1, 3))
        dark_val = int(rng.integers(14, 34))
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(out, p1, p2, (dark_val, dark_val, dark_val), thickness, cv2.LINE_AA)

        # Occasional secondary hairline fork
        if len(pts) > 5 and rng.random() < 0.6:
            fork_start = pts[len(pts) // 2]
            fx, fy = fork_start
            fork_pts = [(fx, fy)]
            for _ in range(5):
                fx = int(np.clip(fx + int(rng.integers(-5, 6)), 2, w - 2))
                fy = int(np.clip(fy + int(rng.integers(-5, 6)), 2, h - 2))
                fork_pts.append((fx, fy))
            for p1, p2 in zip(fork_pts[:-1], fork_pts[1:]):
                cv2.line(out, p1, p2, (dark_val + 6, dark_val + 6, dark_val + 6), 1, cv2.LINE_AA)

    return out


def _render_road_sample(damage_class: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a single 96x96 BGR road sample for the given class."""
    img = _create_asphalt_base(rng, size=96)
    if damage_class == "pothole":
        return _draw_pothole(img, rng)
    if damage_class == "crack":
        return _draw_crack(img, rng)
    return img


def _has_raw_rdd_images() -> bool:
    """Check whether user-supplied Road Damage Dataset images exist in data/raw/."""
    for class_name in CLASS_NAMES:
        class_dir = RAW_DATA_DIR / class_name
        if class_dir.exists() and any(
            p.suffix.lower() in SUPPORTED_EXTENSIONS for p in class_dir.iterdir()
        ):
            return True
    return False


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def count_split_images(split_dir: Path) -> dict[str, int]:
    """Count image files per class inside a split directory."""
    counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        cdir = split_dir / class_name
        if not cdir.exists():
            counts[class_name] = 0
        else:
            counts[class_name] = sum(
                1 for p in cdir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
    return counts


def prepare_curated_dataset(
    seed: int = DEFAULT_SEED,
    train_counts: dict[str, int] | None = None,
    test_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    """Prepare processed train/test splits from raw RDD images or curated starter dataset."""
    train_counts = train_counts or {"normal": 110, "pothole": 24, "crack": 24}
    test_counts = test_counts or {"normal": 45, "pothole": 45, "crack": 45}

    train_dir = PROCESSED_DATA_DIR / "train"
    test_dir = PROCESSED_DATA_DIR / "test"
    _reset_dir(train_dir)
    _reset_dir(test_dir)
    _reset_dir(SAMPLES_DIR / "real")

    rng = np.random.default_rng(seed)
    source_mode = "curated_starter_dataset"

    if _has_raw_rdd_images():
        source_mode = "road_damage_dataset_raw"
        py_rng = random.Random(seed)
        for class_name in CLASS_NAMES:
            (train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (test_dir / class_name).mkdir(parents=True, exist_ok=True)
            (SAMPLES_DIR / "real" / class_name).mkdir(parents=True, exist_ok=True)

            files = [
                p
                for p in (RAW_DATA_DIR / class_name).iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            py_rng.shuffle(files)
            split_idx = max(1, int(len(files) * 0.75)) if len(files) > 1 else len(files)
            for idx, src in enumerate(files):
                bgr = cv2.imread(str(src))
                if bgr is None:
                    continue
                resized = cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_AREA)
                dest_root = train_dir if idx < split_idx else test_dir
                cv2.imwrite(str(dest_root / class_name / f"{class_name}_{idx:04d}.png"), resized)
                if idx < 4:
                    cv2.imwrite(str(SAMPLES_DIR / "real" / class_name / f"sample_{idx + 1}.png"), resized)
    else:
        for split_name, counts, split_root in (
            ("train", train_counts, train_dir),
            ("test", test_counts, test_dir),
        ):
            for class_name in CLASS_NAMES:
                cdir = split_root / class_name
                cdir.mkdir(parents=True, exist_ok=True)
                if split_name == "test":
                    (SAMPLES_DIR / "real" / class_name).mkdir(parents=True, exist_ok=True)

                for idx in range(counts[class_name]):
                    img = _render_road_sample(class_name, rng)
                    cv2.imwrite(str(cdir / f"{class_name}_{idx:04d}.png"), img)
                    if split_name == "test" and idx < 4:
                        cv2.imwrite(
                            str(SAMPLES_DIR / "real" / class_name / f"sample_{idx + 1}.png"),
                            cv2.resize(img, (192, 192), interpolation=cv2.INTER_CUBIC),
                        )

    summary = {
        "source": source_mode,
        "image_size": IMAGE_SIZE,
        "real_train_counts": count_split_images(train_dir),
        "real_test_counts": count_split_images(test_dir),
    }
    return summary


def build_augmented_dataset(
    generator: RoadConditionalGenerator,
    target_per_class: int = 110,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
) -> dict[str, object]:
    """Combine real training images with GAN-generated synthetic images into augmented split."""
    real_train_dir = PROCESSED_DATA_DIR / "train"
    aug_train_dir = AUGMENTED_DATA_DIR / "train"
    _reset_dir(SYNTHETIC_DATA_DIR)
    _reset_dir(aug_train_dir)
    _reset_dir(SAMPLES_DIR / "synthetic")

    synthetic_counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        real_cdir = real_train_dir / class_name
        aug_cdir = aug_train_dir / class_name
        syn_cdir = SYNTHETIC_DATA_DIR / class_name
        sample_cdir = SAMPLES_DIR / "synthetic" / class_name
        aug_cdir.mkdir(parents=True, exist_ok=True)
        syn_cdir.mkdir(parents=True, exist_ok=True)
        sample_cdir.mkdir(parents=True, exist_ok=True)

        existing = [
            p for p in real_cdir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ] if real_cdir.exists() else []

        for src in existing:
            shutil.copy2(src, aug_cdir / src.name)

        needed = max(12 if class_name != "normal" else 0, target_per_class - len(existing))
        synthetic_counts[class_name] = needed

        if needed > 0:
            generated_batches: list[torch.Tensor] = []
            remaining = needed
            batch_seed = seed + CLASS_TO_IDX[class_name] * 100
            while remaining > 0:
                chunk = min(32, remaining)
                generated_batches.append(
                    generate_synthetic_tensors(
                        generator=generator,
                        damage_class=class_name,
                        count=chunk,
                        seed=batch_seed,
                        device=device,
                    )
                )
                remaining -= chunk
                batch_seed += 1

            all_syn = torch.cat(generated_batches, dim=0)
            for idx in range(all_syn.size(0)):
                bgr_64 = tensor_to_bgr(all_syn[idx], target_size=IMAGE_SIZE)
                cv2.imwrite(str(syn_cdir / f"syn_{class_name}_{idx:04d}.png"), bgr_64)
                cv2.imwrite(str(aug_cdir / f"syn_{class_name}_{idx:04d}.png"), bgr_64)
                if idx < 4:
                    bgr_preview = tensor_to_bgr(all_syn[idx], target_size=192)
                    cv2.imwrite(str(sample_cdir / f"sample_{idx + 1}.png"), bgr_preview)

    summary = {
        "real_train_counts": count_split_images(real_train_dir),
        "synthetic_added_counts": synthetic_counts,
        "augmented_train_counts": count_split_images(aug_train_dir),
        "real_test_counts": count_split_images(PROCESSED_DATA_DIR / "test"),
    }
    return summary


def load_train_and_test_arrays() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load real train, augmented train, and held-out test numpy arrays."""
    return {
        "real_train": load_labeled_images(PROCESSED_DATA_DIR / "train", image_size=IMAGE_SIZE),
        "augmented_train": load_labeled_images(AUGMENTED_DATA_DIR / "train", image_size=IMAGE_SIZE),
        "real_test": load_labeled_images(PROCESSED_DATA_DIR / "test", image_size=IMAGE_SIZE),
    }


def save_dataset_summary(summary: dict[str, object], artifacts_dir: Path) -> Path:
    """Persist dataset split summary JSON inside artifacts directory."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "dataset_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
