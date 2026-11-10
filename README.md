# RoadTwin-GAN

AI-Driven Road Surface Digital Twin Generation & Quality Inspection Framework.

---

## 📌 Project Overview

**RoadTwin-GAN** is an engineering project aimed at creating high-fidelity digital twin representations of road surfaces from inspection photographs. The platform will combine deep generative adversarial networks (GANs) and computer vision to analyze road distress (potholes, cracks, rutting) and simulate road surface aging and maintenance scenarios.

> **Status Notice:** This repository represents the initial foundation containing the user interface, Flask REST API, and OpenCV validation pipeline. **Actual PyTorch GAN and defect detection model integrations are pending** in subsequent milestones. No fake or mock predictions are generated.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3 (Modern Responsive UI), Vanilla JavaScript (Fetch API) |
| **Backend** | Python 3.9+, Flask, Flask-CORS, Werkzeug |
| **Image Processing** | OpenCV (`opencv-python`), NumPy, Pillow |
| **Future AI Pipeline** | PyTorch (`torch`, `torchvision`) for Generative Adversarial Networks |

---

## 📁 Repository Structure

```text
RoadTwin-GAN/
│
├── frontend/
│   ├── index.html          # Main web dashboard with road image upload & result slots
│   ├── css/
│   │   └── style.css       # Responsive styling, modern dark slate theme & animations
│   └── js/
│       └── script.js       # Client logic: drag-and-drop, preview, REST API communication
│
├── backend/
│   ├── app.py              # Flask application with /api/health and /api/upload endpoints
│   ├── requirements.txt    # Python dependencies (Flask, OpenCV, PyTorch, etc.)
│   └── uploads/            # Directory where validated uploaded images are stored
│       └── .gitkeep        # Keeps folder tracked in Git
│
├── .gitignore              # Ignores venv, model weights, cache, and uploaded images
└── README.md               # Project setup and execution documentation
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- **Python 3.9+** installed on your system.
- Modern web browser (Chrome, Edge, Firefox, etc.).

---

### Step 1: Set Up and Run the Flask Backend

1. Open a terminal and navigate to the project directory:
   ```bash
   cd RoadTwin-GAN
   ```

2. (Recommended) Create and activate a Python virtual environment:
   * **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install the required dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. Run the Flask development server:
   ```bash
   python backend/app.py
   ```
   The backend will start at `http://127.0.0.1:5000`.

---

### Step 2: Launch the Frontend

You can run the frontend in any of the following ways:

* **Option A (VS Code Live Server - Recommended):**
  Right-click `frontend/index.html` in VS Code and click **"Open with Live Server"**.

* **Option B (Python Local HTTP Server):**
  Open a new terminal window:
  ```bash
  cd frontend
  python -m http.server 8000
  ```
  Open `http://127.0.0.1:8000` in your web browser.

* **Option C (Direct File Open):**
  Double-click `frontend/index.html` to open it directly in your web browser (CORS is enabled on Flask to allow cross-origin requests).

---

## 📡 API Endpoints

### 1. Health Check
* **Endpoint:** `GET /api/health`
* **Description:** Verifies that the Flask service is alive and reports model integration status.
* **Sample Response:**
  ```json
  {
    "status": "healthy",
    "service": "RoadTwin-GAN Backend API",
    "version": "0.1.0",
    "ai_status": {
      "gan_model": "pending_integration",
      "detection_model": "pending_integration",
      "message": "AI inference pipeline is awaiting PyTorch model weights and architecture definition."
    }
  }
  ```

### 2. Road Image Upload & Validation
* **Endpoint:** `POST /api/upload`
* **Form Field:** `image` (Multipart Form Data; allowed types: `.png`, `.jpg`, `.jpeg`, `.webp`; max size: 16 MB)
* **Description:** Receives the road photo, decodes the image via OpenCV to ensure file integrity, extracts dimensions and channels, saves the file safely, and returns image metadata with the pending AI status.
* **Sample Response:**
  ```json
  {
    "status": "success",
    "message": "Road image successfully uploaded and validated by OpenCV.",
    "file_info": {
      "original_filename": "pavement_01.jpg",
      "saved_filename": "20261110_132000_pavement_01.jpg",
      "width_px": 1920,
      "height_px": 1080,
      "channels": 3,
      "size_kb": 342.15
    },
    "ai_integration": {
      "status": "pending",
      "message": "Actual GAN generation and defect detection models are pending integration. No synthetic predictions are returned.",
      "pending_modules": [
        "PyTorch RoadTwin Generator (GAN)",
        "Road Condition / Defect Segmentation Model",
        "Digital Twin 3D / Synthetic Surface Reconstructor"
      ]
    }
  }
  ```

---

## 🔄 How Frontend Communicates with Backend

1. **Initial Connection & Health Heartbeat**:
   * On page load, `script.js` sends a `GET` request to `http://127.0.0.1:5000/api/health`.
   * The status badge in the UI updates dynamically to **"Backend Online"** (green indicator) or **"Backend Offline"** (red indicator) if the server cannot be reached.
2. **File Selection & Client-Side Validation**:
   * The user selects or drags and drops a road image into the drop zone.
   * `script.js` validates file size and MIME type, then renders a live local preview using the HTML5 `FileReader` API.
3. **Multipart Form Upload**:
   * When the user clicks **"Upload & Inspect"**, `script.js` packages the binary image file into a `FormData` object and issues a `POST` request to `/api/upload`.
4. **Backend Processing & Verification**:
   * Flask receives the payload and passes the byte buffer to OpenCV (`cv2.imdecode`).
   * OpenCV validates that the bytes form a valid, non-corrupted raster image and extracts its resolution (width, height) and color channels.
5. **Metadata Display & Pending Notice**:
   * The backend returns a 200 OK JSON payload containing the image metadata and an explicit `status: "pending"` marker for AI models.
   * The frontend parses the JSON and populates the **Image Validation Metadata** card, while keeping the future GAN/defect output slots labeled with **"Model Status: Pending Integration"**.

---

## 🔮 Roadmap / Future Milestones

- [ ] Define PyTorch GAN generator architecture for road digital twin synthesis.
- [ ] Implement dataset loaders and preprocessing pipelines for road surface distress images.
- [ ] Integrate deep learning defect segmentation (cracks, potholes, patches).
- [ ] Add 3D depth/elevation surface visualization in the frontend.
