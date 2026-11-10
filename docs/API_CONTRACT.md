# RoadTwin-GAN Backend API Contract (For Frontend Integration)

Base URL:
- Local development: `http://127.0.0.1:5000`
- Production (Render): `https://<your-render-service>.onrender.com`

CORS is enabled on all `/api/*` endpoints.

---

## 1. Health Check
`GET /api/health`

Response `200`:
```json
{
  "status": "ok",
  "service": "RoadTwin-GAN Backend",
  "device": "cpu",
  "classes": ["normal", "pothole", "crack"],
  "models_loaded": {
    "gan_generator": true,
    "detector_baseline": true,
    "detector_augmented": true
  }
}
```

---

## 2. Predict Road Damage (Upload Image)
`POST /api/predict`

Supports **either**:
- `multipart/form-data`:
  - `image` (file, required — PNG/JPG/BMP/WEBP, max 8 MB)
  - `model_type` (`"augmented"` default or `"baseline"`)
  - `save_report` (`"true"` or `"false"`)
  - `location` (string, optional)
  - `notes` (string, optional)
- OR `application/json`:
```json
{
  "image_base64": "data:image/png;base64,...",
  "model_type": "augmented",
  "save_report": true,
  "location": "Main Street Mile 4",
  "notes": "Submitted from civic dashboard"
}
```

Response `200`:
```json
{
  "status": "ok",
  "prediction": {
    "model_type": "augmented",
    "damage_type": "pothole",
    "confidence": 0.9421,
    "class_probabilities": {
      "normal": 0.021,
      "pothole": 0.9421,
      "crack": 0.0369
    },
    "severity": "High",
    "affected_area_pct": 14.5,
    "bounding_boxes": [
      { "id": 1, "label": "pothole", "x": 68, "y": 72, "width": 108, "height": 94, "area_pct": 14.5 }
    ],
    "surface_features": {
      "mean_intensity": 88.4,
      "edge_density": 0.0821,
      "texture_variance": 291.5,
      "dark_depression_ratio": 0.165
    },
    "recommendation": "Severe pothole hazard detected. Prioritize immediate barricade and hot-mix repair.",
    "annotated_image_base64": "data:image/png;base64,...",
    "inference_ms": 14.2
  },
  "civic_report": { "id": 1, "location": "Main Street Mile 4", "status": "Open" }
}
```

---

## 3. Generate Synthetic Road Damage Images (Conditional DCGAN)
`POST /api/gan/generate`

Request JSON:
```json
{
  "damage_type": "pothole",
  "count": 4,
  "seed": 42
}
```
- `damage_type`: `"normal" | "pothole" | "crack"`
- `count`: integer `1..32`
- `seed`: optional integer

Response `200`:
```json
{
  "status": "ok",
  "damage_type": "pothole",
  "count": 4,
  "images": [
    {
      "index": 1,
      "damage_type": "pothole",
      "width": 192,
      "height": 192,
      "mean_brightness": 91.4,
      "image_base64": "data:image/png;base64,..."
    }
  ]
}
```

---

## 4. Real vs GAN-Synthetic Samples Gallery
`GET /api/gan/samples` (optional `?damage_type=pothole|crack|normal`)

Returns base64 sample thumbnails under `samples.real.<class>` and `samples.synthetic.<class>`.

---

## 5. Dataset Split Summary
`GET /api/dataset/summary`

Returns counts for `real_train_counts`, `synthetic_added_counts`, `augmented_train_counts`, and `real_test_counts`.

---

## 6. Comparative Evaluation (Before vs After GAN Augmentation)
`GET /api/evaluation`

Returns precision, recall, F1-score, accuracy, per-class metrics, confusion matrices, GAN loss history, and plot URLs (`before_augmentation`, `after_augmentation`, `improvement_delta`).

### Evaluation Plots (PNG Images)
- `GET /api/evaluation/plots/metrics_comparison.png`
- `GET /api/evaluation/plots/confusion_matrices.png`
- `GET /api/evaluation/plots/gan_loss_curve.png`

---

## 7. Civic Monitoring Reports
- `GET /api/civic-reports?limit=50` — List inspection logs
- `POST /api/civic-reports` — Log new report (`location`, `damage_type`, `severity`, `confidence`, `affected_area_pct`, `notes`)
- `PATCH /api/civic-reports/<id>` — Update `status` (`"Open" | "Scheduled" | "Resolved"`)
