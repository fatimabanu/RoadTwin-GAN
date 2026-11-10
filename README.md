# RoadTwin-GAN

**GAN-Based Synthetic Road Damage Augmentation for Improved Civic Infrastructure Monitoring**

> Course: Societal Based Mini Project (`24CSE57`) — Department of Computer Science and Engineering  
> Team: **Fatima Banu** (Backend & ML Pipeline) & **Fabiya M** (Frontend)

---

## Overview

Automated road damage detection requires large, balanced, and diverse labeled image datasets, whereas real-world pothole and crack images are often scarce. **RoadTwin-GAN** uses a **PyTorch Conditional DCGAN** to generate realistic synthetic road damage images (`pothole`, `crack`, `normal`), augments scarce real training splits, trains **RoadDamageCNN** classifiers before and after augmentation, and evaluates detection improvements using **Precision, Recall, F1-Score, and Confusion Matrices**.

A **Flask REST API** exposes detection, OpenCV damage localization, on-demand GAN image synthesis, comparative evaluation plots, and civic inspection logging for seamless frontend integration and **Render** deployment.

---

## Tech Stack

- **Language:** Python 3.10+ (configured for Python 3.12)
- **Deep Learning:** PyTorch, torchvision (Conditional DCGAN + RoadDamageCNN)
- **Computer Vision & Data:** OpenCV (`opencv-python-headless`), NumPy, pandas, Pillow
- **Evaluation & Visualization:** scikit-learn, Matplotlib
- **Backend API & Deployment:** Flask, Flask-CORS, Gunicorn, SQLite, Render Blueprint (`render.yaml`)

---

## Repository Structure

```text
RoadTwin-GAN/
├── backend/
│   ├── app.py                   # Flask REST API entry point (Gunicorn-ready)
│   ├── config.py                # Paths, labels, hyperparameters, recommendations
│   ├── storage.py               # SQLite civic damage report storage
│   ├── ml/
│   │   ├── preprocessing.py     # Resize, normalization, OpenCV features & damage localization
│   │   ├── gan.py               # PyTorch Conditional DCGAN generator & discriminator
│   │   ├── dataset_builder.py   # Curated dataset prep (supports raw RDD) & GAN augmentation
│   │   ├── detector.py          # RoadDamageCNN training & image inference pipeline
│   │   └── evaluation.py        # Precision/Recall/F1 comparison & Matplotlib plot generation
│   ├── scripts/
│   │   ├── prepare_data.py      # Build/split curated dataset
│   │   ├── train_gan.py         # Train cDCGAN & synthesize augmented training split
│   │   ├── train_detector.py    # Train baseline vs augmented CNNs & export evaluation plots
│   │   └── run_pipeline.py      # Run all three steps sequentially
│   ├── artifacts/               # Pre-trained weights, evaluation JSON, sample previews & plots
│   ├── tests/
│   │   └── test_api.py          # End-to-end API tests (pytest)
│   ├── requirements.txt
│   ├── Procfile
│   └── .env.example
├── docs/
│   └── API_CONTRACT.md          # Complete endpoint payloads & examples for frontend integration
├── render.yaml                  # Render Web Service Blueprint
└── README.md
```

---

## Running the Backend Locally

1. **Create & activate virtual environment**
   ```bash
   cd backend
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   # source .venv/bin/activate
   ```

2. **Install dependencies (CPU PyTorch + project requirements)**
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   pip install -r requirements.txt
   ```

3. **Run tests**
   ```bash
   pytest -v
   ```

4. **Start the Flask server**
   ```bash
   python app.py
   ```
   The API will be available at `http://127.0.0.1:5000` (health check: `GET /api/health`).

5. **(Optional) Retrain the GAN & Detectors from scratch**
   - To train on your own **Road Damage Dataset (RDD)** images, place labeled images into:
     - `backend/data/raw/normal/`
     - `backend/data/raw/pothole/`
     - `backend/data/raw/crack/`
   - If `backend/data/raw/` is empty, the pipeline automatically builds a curated starter road-damage dataset.
   - Run the full ML pipeline:
     ```bash
     python scripts/run_pipeline.py
     ```

---

## Team GitHub Workflow (Backend + Frontend Merge)

1. Backend development lives on the `backend` branch (`backend/`, `docs/API_CONTRACT.md`, `render.yaml`).
2. Frontend development goes in a `frontend/` folder on a separate branch (e.g., `frontend`), calling the API documented in [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).
3. Open Pull Requests on GitHub for both branches and merge them cleanly into `main`.

---

## Deploying to Render

This repository includes [`render.yaml`](render.yaml) configured for Render's Python runtime:

1. Merge `backend` (and `frontend`) into `main` on GitHub.
2. On [Render](https://render.com), click **New + -> Blueprint** (or **New Web Service**) and connect `fatimabanu/RoadTwin-GAN`.
3. Render automatically picks up `render.yaml`:
   - **Root Directory:** `backend`
   - **Build Command:** Installs CPU PyTorch + `requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   - **Health Check Path:** `/api/health`
4. Set `CORS_ORIGINS` in Render Environment Variables to your deployed frontend URL (e.g., `https://roadtwin-frontend.onrender.com`) once frontend is live.
