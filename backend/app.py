import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for frontend communication
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB maximum upload limit

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a supported image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify backend status and service availability.
    """
    return jsonify({
        "status": "healthy",
        "service": "RoadTwin-GAN Backend API",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ai_status": {
            "gan_model": "pending_integration",
            "detection_model": "pending_integration",
            "message": "AI inference pipeline is awaiting PyTorch model weights and architecture definition."
        }
    }), 200


@app.route('/api/upload', methods=['POST'])
def upload_road_image():
    """
    Image upload and validation endpoint.
    Validates file presence, extension, and decodes the image using OpenCV to verify integrity.
    Does NOT produce fake predictions; explicitly notes that AI inference is pending.
    """
    # 1. Check if file is included in request
    if 'image' not in request.files:
        return jsonify({
            "status": "error",
            "error_code": "NO_FILE_PROVIDED",
            "message": "No 'image' field was found in the multipart form data."
        }), 400

    file = request.files['image']

    # 2. Check if a file was selected
    if file.filename == '':
        return jsonify({
            "status": "error",
            "error_code": "EMPTY_FILENAME",
            "message": "No file was selected for upload."
        }), 400

    # 3. Check allowed file extension
    if not is_allowed_file(file.filename):
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({
            "status": "error",
            "error_code": "UNSUPPORTED_FILE_TYPE",
            "message": f"Unsupported file type. Allowed formats: {allowed_list}"
        }), 400

    try:
        # Read file into memory buffer
        file_bytes = file.read()
        file_size_kb = len(file_bytes) / 1024.0

        # 4. Use OpenCV to decode and validate image content integrity
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "status": "error",
                "error_code": "INVALID_IMAGE_DATA",
                "message": "The file could not be decoded as a valid image. The file may be corrupted."
            }), 400

        height, width, channels = img.shape

        # 5. Save the validated file safely to the upload directory
        original_name = secure_filename(file.filename)
        timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp_prefix}_{original_name}"
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)

        with open(saved_path, 'wb') as f:
            f.write(file_bytes)

        # 6. Return response with image metadata and explicit pending AI status
        return jsonify({
            "status": "success",
            "message": "Road image successfully uploaded and validated by OpenCV.",
            "file_info": {
                "original_filename": original_name,
                "saved_filename": saved_filename,
                "width_px": width,
                "height_px": height,
                "channels": channels,
                "size_kb": round(file_size_kb, 2)
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
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "error_code": "SERVER_PROCESSING_ERROR",
            "message": f"An unexpected error occurred while processing the image: {str(e)}"
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "status": "error",
        "error_code": "PAYLOAD_TOO_LARGE",
        "message": "File exceeds the 16 MB maximum upload limit."
    }), 413


if __name__ == '__main__':
    print("==================================================")
    print(" Starting RoadTwin-GAN Flask Backend")
    print(" Health Endpoint: http://127.0.0.1:5000/api/health")
    print(" Upload Endpoint: http://127.0.0.1:5000/api/upload")
    print("==================================================")
    app.run(host='127.0.0.1', port=5000, debug=True)
