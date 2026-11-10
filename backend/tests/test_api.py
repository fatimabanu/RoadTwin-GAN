"""End-to-end tests for RoadTwin-GAN Flask API and ML artifacts."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import create_app
from config import ARTIFACTS_DIR


def _make_sample_png_bytes() -> bytes:
    """Create a small synthetic 96x96 asphalt road PNG with a dark pothole patch."""
    img = np.full((96, 96, 3), 105, dtype=np.uint8)
    cv2.circle(img, (48, 48), 16, (25, 25, 25), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client(tmp_path: Path):
    """Provide an isolated Flask test client with temporary SQLite DB and trained artifacts."""
    app = create_app(artifacts_dir=ARTIFACTS_DIR, db_path=tmp_path / "test_reports.db")
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_index_and_health(client) -> None:
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert root_res.get_json()["service"] == "RoadTwin-GAN Backend API"

    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    data = health_res.get_json()
    assert data["status"] == "ok"
    assert data["classes"] == ["normal", "pothole", "crack"]
    assert data["models_loaded"]["gan_generator"] is True
    assert data["models_loaded"]["detector_baseline"] is True
    assert data["models_loaded"]["detector_augmented"] is True


def test_predict_multipart_and_base64(client) -> None:
    png_bytes = _make_sample_png_bytes()

    # Multipart upload with report persistence
    res = client.post(
        "/api/predict",
        data={
            "image": (io.BytesIO(png_bytes), "road.png"),
            "model_type": "augmented",
            "save_report": "true",
            "location": "MG Road Sector 4",
        },
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    pred = body["prediction"]
    assert pred["damage_type"] in {"normal", "pothole", "crack"}
    assert 0.0 <= pred["confidence"] <= 1.0
    assert pred["annotated_image_base64"].startswith("data:image/png;base64,")
    assert body["civic_report"]["location"] == "MG Road Sector 4"

    # JSON base64 upload with baseline model
    b64_payload = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    res_b64 = client.post(
        "/api/predict",
        json={"image_base64": b64_payload, "model_type": "baseline"},
    )
    assert res_b64.status_code == 200
    assert res_b64.get_json()["prediction"]["model_type"] == "baseline"

    # Invalid input error checks
    bad_res = client.post("/api/predict", json={"image_base64": "not-valid-base64!!!"})
    assert bad_res.status_code == 400


def test_gan_generate_and_samples(client) -> None:
    gen_res = client.post("/api/gan/generate", json={"damage_type": "pothole", "count": 3, "seed": 7})
    assert gen_res.status_code == 200
    payload = gen_res.get_json()
    assert payload["count"] == 3
    assert len(payload["images"]) == 3
    assert payload["images"][0]["image_base64"].startswith("data:image/png;base64,")

    bad_gen = client.post("/api/gan/generate", json={"damage_type": "unknown", "count": 2})
    assert bad_gen.status_code == 400

    samples_res = client.get("/api/gan/samples?damage_type=crack")
    assert samples_res.status_code == 200
    samples_data = samples_res.get_json()["samples"]
    assert "crack" in samples_data["real"]
    assert "crack" in samples_data["synthetic"]


def test_dataset_summary_and_evaluation_plots(client) -> None:
    ds_res = client.get("/api/dataset/summary")
    assert ds_res.status_code == 200
    assert "augmented_train_counts" in ds_res.get_json()["dataset_summary"]

    eval_res = client.get("/api/evaluation")
    assert eval_res.status_code == 200
    eval_data = eval_res.get_json()["evaluation"]
    assert "before_augmentation" in eval_data
    assert "after_augmentation" in eval_data
    assert "f1_macro" in eval_data["after_augmentation"]

    for plot_name in ("metrics_comparison.png", "confusion_matrices.png", "gan_loss_curve.png"):
        plot_res = client.get(f"/api/evaluation/plots/{plot_name}")
        assert plot_res.status_code == 200
        assert plot_res.mimetype == "image/png"

    missing_plot = client.get("/api/evaluation/plots/secret.png")
    assert missing_plot.status_code == 404


def test_civic_reports_crud(client) -> None:
    create_res = client.post(
        "/api/civic-reports",
        json={
            "location": "Outer Ring Road Junction",
            "damage_type": "crack",
            "severity": "Moderate",
            "confidence": 0.91,
            "affected_area_pct": 6.4,
            "notes": "Longitudinal crack near lane edge",
        },
    )
    assert create_res.status_code == 201
    report_id = create_res.get_json()["report"]["id"]

    list_res = client.get("/api/civic-reports")
    assert list_res.status_code == 200
    assert list_res.get_json()["count"] == 1

    patch_res = client.patch(f"/api/civic-reports/{report_id}", json={"status": "Scheduled"})
    assert patch_res.status_code == 200
    assert patch_res.get_json()["report"]["status"] == "Scheduled"
