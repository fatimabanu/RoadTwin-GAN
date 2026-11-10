"""Flask REST API for RoadTwin-GAN road damage augmentation and detection."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config import ARTIFACTS_DIR, CLASS_NAMES, DB_PATH, ensure_directories
from ml.detector import RoadDamageCNN, analyze_road_image, load_detector
from ml.gan import RoadConditionalGenerator, generate_synthetic_images, load_gan_generator
from ml.preprocessing import SUPPORTED_EXTENSIONS, decode_base64_image, decode_image_bytes, encode_image_base64
from storage import CivicReportStore

load_dotenv()

ALLOWED_PLOTS = {
    "metrics_comparison.png",
    "confusion_matrices.png",
    "gan_loss_curve.png",
}
ALLOWED_REPORT_STATUSES = {"Open", "Scheduled", "Resolved"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB


def _load_json_artifact(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_samples(samples_dir: Path, damage_filter: str | None = None) -> dict[str, object]:
    """Read sample real and GAN-synthetic thumbnails into base64 payloads."""
    result: dict[str, dict[str, list[dict[str, str]]]] = {"real": {}, "synthetic": {}}
    target_classes = (damage_filter,) if damage_filter in CLASS_NAMES else CLASS_NAMES

    for source_kind in ("real", "synthetic"):
        for class_name in target_classes:
            cdir = samples_dir / source_kind / class_name
            items: list[dict[str, str]] = []
            if cdir.exists():
                for file_path in sorted(cdir.iterdir()):
                    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    bgr = cv2.imread(str(file_path))
                    if bgr is None:
                        continue
                    items.append(
                        {
                            "filename": file_path.name,
                            "damage_type": class_name,
                            "image_base64": encode_image_base64(bgr),
                        }
                    )
            result[source_kind][class_name] = items
    return result


def create_app(
    artifacts_dir: Path | None = None,
    db_path: Path | None = None,
) -> Flask:
    """Create and configure the RoadTwin-GAN Flask application."""
    ensure_directories()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    cors_raw = os.getenv("CORS_ORIGINS", "*").strip()
    origins: str | list[str] = "*" if cors_raw == "*" else [o.strip() for o in cors_raw.split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": origins}})

    resolved_artifacts = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    resolved_db = Path(db_path) if db_path else DB_PATH
    store = CivicReportStore(resolved_db)

    gan_path = resolved_artifacts / "gan_generator.pt"
    baseline_path = resolved_artifacts / "detector_baseline.pt"
    augmented_path = resolved_artifacts / "detector_augmented.pt"

    gan_model: RoadConditionalGenerator | None = load_gan_generator(gan_path) if gan_path.exists() else None
    detector_models: dict[str, RoadDamageCNN] = {}
    if baseline_path.exists():
        detector_models["baseline"] = load_detector(baseline_path)
    if augmented_path.exists():
        detector_models["augmented"] = load_detector(augmented_path)

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "RoadTwin-GAN Backend API",
                "description": "GAN-Based Synthetic Road Damage Augmentation & Civic Monitoring API",
                "classes": list(CLASS_NAMES),
                "endpoints": {
                    "health": "GET /api/health",
                    "predict": "POST /api/predict",
                    "gan_generate": "POST /api/gan/generate",
                    "gan_samples": "GET /api/gan/samples",
                    "dataset_summary": "GET /api/dataset/summary",
                    "evaluation": "GET /api/evaluation",
                    "evaluation_plot": "GET /api/evaluation/plots/<filename>",
                    "civic_reports": "GET, POST /api/civic-reports",
                    "update_civic_report": "PATCH /api/civic-reports/<id>",
                },
            }
        )

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "RoadTwin-GAN Backend",
                "device": "cpu",
                "classes": list(CLASS_NAMES),
                "models_loaded": {
                    "gan_generator": gan_model is not None,
                    "detector_baseline": "baseline" in detector_models,
                    "detector_augmented": "augmented" in detector_models,
                },
            }
        )

    @app.post("/api/predict")
    def predict():
        if not detector_models:
            return jsonify({"error": "Detector models are not loaded. Run training pipeline first."}), 503

        if request.content_type and "multipart/form-data" in request.content_type:
            file_storage = request.files.get("image") or request.files.get("file")
            if file_storage is None:
                return jsonify({"error": "Missing multipart file field 'image'."}), 400
            model_type = str(request.form.get("model_type", "augmented")).strip().lower()
            save_report = str(request.form.get("save_report", "false")).strip().lower() == "true"
            location = str(request.form.get("location", "Unspecified Road Segment")).strip()
            notes = str(request.form.get("notes", "")).strip()
            try:
                bgr_img = decode_image_bytes(file_storage.read())
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        else:
            payload = request.get_json(silent=True) or {}
            b64_img = payload.get("image_base64")
            if not b64_img:
                return jsonify({"error": "Provide multipart 'image' file or JSON 'image_base64'."}), 400
            model_type = str(payload.get("model_type", "augmented")).strip().lower()
            save_report = bool(payload.get("save_report", False))
            location = str(payload.get("location", "Unspecified Road Segment")).strip()
            notes = str(payload.get("notes", "")).strip()
            try:
                bgr_img = decode_base64_image(str(b64_img))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

        if model_type not in detector_models:
            if model_type in {"baseline", "augmented"}:
                return jsonify({"error": f"Requested model '{model_type}' is not available."}), 503
            return jsonify({"error": "model_type must be 'augmented' or 'baseline'."}), 400

        analysis = analyze_road_image(bgr_img, detector_models[model_type], model_type=model_type)

        created_report = None
        if save_report:
            created_report = store.create_report(
                location=location or "Unspecified Road Segment",
                damage_type=str(analysis["damage_type"]),
                severity=str(analysis["severity"]),
                confidence=float(analysis["confidence"]),
                affected_area_pct=float(analysis["affected_area_pct"]),
                notes=notes,
            )

        return jsonify({"status": "ok", "prediction": analysis, "civic_report": created_report})

    @app.post("/api/gan/generate")
    def gan_generate():
        if gan_model is None:
            return jsonify({"error": "GAN generator is not loaded. Run training pipeline first."}), 503

        payload = request.get_json(silent=True) or {}
        damage_type = str(payload.get("damage_type", "pothole")).strip().lower()
        seed_raw = payload.get("seed")
        try:
            count = int(payload.get("count", 4))
            seed = int(seed_raw) if seed_raw is not None else None
            images = generate_synthetic_images(
                generator=gan_model,
                damage_class=damage_type,
                count=count,
                seed=seed,
                device="cpu",
            )
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(
            {
                "status": "ok",
                "damage_type": damage_type,
                "count": len(images),
                "images": images,
            }
        )

    @app.get("/api/gan/samples")
    def gan_samples():
        damage_filter = request.args.get("damage_type")
        if damage_filter is not None and damage_filter not in CLASS_NAMES:
            return jsonify({"error": f"damage_type must be one of {list(CLASS_NAMES)}."}), 400
        samples = _collect_samples(resolved_artifacts / "samples", damage_filter=damage_filter)
        return jsonify({"status": "ok", "classes": list(CLASS_NAMES), "samples": samples})

    @app.get("/api/dataset/summary")
    def dataset_summary():
        summary = _load_json_artifact(resolved_artifacts / "dataset_summary.json")
        if summary is None:
            return jsonify({"error": "Dataset summary not found. Run preparation pipeline first."}), 404
        return jsonify({"status": "ok", "dataset_summary": summary})

    @app.get("/api/evaluation")
    def evaluation():
        report = _load_json_artifact(resolved_artifacts / "evaluation_comparison.json")
        if report is None:
            return jsonify({"error": "Evaluation report not found. Run evaluation pipeline first."}), 404
        return jsonify({"status": "ok", "evaluation": report})

    @app.get("/api/evaluation/plots/<path:filename>")
    def evaluation_plot(filename: str):
        if filename not in ALLOWED_PLOTS:
            return jsonify({"error": f"Unknown plot '{filename}'. Allowed: {sorted(ALLOWED_PLOTS)}"}), 404
        plots_dir = resolved_artifacts / "plots"
        if not (plots_dir / filename).exists():
            return jsonify({"error": f"Plot '{filename}' has not been generated yet."}), 404
        return send_from_directory(plots_dir, filename, mimetype="image/png")

    @app.get("/api/civic-reports")
    def list_civic_reports():
        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            return jsonify({"error": "limit must be an integer."}), 400
        reports = store.list_reports(limit=limit)
        return jsonify({"status": "ok", "count": len(reports), "reports": reports})

    @app.post("/api/civic-reports")
    def create_civic_report():
        payload = request.get_json(silent=True) or {}
        location = str(payload.get("location", "")).strip()
        damage_type = str(payload.get("damage_type", "")).strip().lower()
        severity = str(payload.get("severity", "Moderate")).strip()
        notes = str(payload.get("notes", "")).strip()

        if not location:
            return jsonify({"error": "location is required."}), 400
        if damage_type not in CLASS_NAMES:
            return jsonify({"error": f"damage_type must be one of {list(CLASS_NAMES)}."}), 400

        try:
            confidence = float(payload.get("confidence", 0.0))
            affected_area_pct = float(payload.get("affected_area_pct", 0.0))
        except (TypeError, ValueError):
            return jsonify({"error": "confidence and affected_area_pct must be numeric."}), 400

        report = store.create_report(
            location=location,
            damage_type=damage_type,
            severity=severity,
            confidence=confidence,
            affected_area_pct=affected_area_pct,
            notes=notes,
        )
        return jsonify({"status": "ok", "report": report}), 201

    @app.patch("/api/civic-reports/<int:report_id>")
    def update_civic_report(report_id: int):
        payload = request.get_json(silent=True) or {}
        status = str(payload.get("status", "")).strip()
        if status not in ALLOWED_REPORT_STATUSES:
            return jsonify({"error": f"status must be one of {sorted(ALLOWED_REPORT_STATUSES)}."}), 400
        try:
            updated = store.update_status(report_id, status)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        return jsonify({"status": "ok", "report": updated})

    @app.errorhandler(404)
    def handle_404(_err):
        return jsonify({"error": "Endpoint not found."}), 404

    @app.errorhandler(405)
    def handle_405(_err):
        return jsonify({"error": "HTTP method not allowed."}), 405

    @app.errorhandler(413)
    def handle_413(_err):
        return jsonify({"error": "Uploaded image exceeds 8 MB size limit."}), 413

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_ENV") == "development")
