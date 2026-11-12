/**
 * RoadTwin-GAN Frontend Client
 * Communicates with Flask Backend via REST APIs (/api/health, /api/upload)
 */

// Configuration: Default Flask backend URL
const BACKEND_BASE_URL = 'http://127.0.0.1:5000';

// DOM Element References
const serverStatusBadge = document.getElementById('serverStatusBadge');
const statusDot = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const dropZonePrompt = document.getElementById('dropZonePrompt');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const previewFilename = document.getElementById('previewFilename');
const previewFilesize = document.getElementById('previewFilesize');

const uploadBtn = document.getElementById('uploadBtn');
const resetBtn = document.getElementById('resetBtn');
const btnSpinner = document.getElementById('btnSpinner');

const feedbackAlert = document.getElementById('feedbackAlert');
const alertIcon = document.getElementById('alertIcon');
const alertMessage = document.getElementById('alertMessage');

const metadataSection = document.getElementById('metadataSection');
const metaWidth = document.getElementById('metaWidth');
const metaHeight = document.getElementById('metaHeight');
const metaChannels = document.getElementById('metaChannels');
const metaSize = document.getElementById('metaSize');

// Application State
let selectedFile = null;

// ==========================================================================
// 1. Backend Health Check
// ==========================================================================

async function checkBackendHealth() {
  try {
    statusDot.className = 'status-dot pulse';
    statusLabel.textContent = 'Connecting...';

    const response = await fetch(`${BACKEND_BASE_URL}/api/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });

    if (response.ok) {
      const data = await response.json();
      statusDot.className = 'status-dot online';
      statusLabel.textContent = `Backend Online (${data.service.split(' ')[0]})`;
      statusLabel.title = `Status: ${data.status} | AI: ${data.ai_status.gan_model}`;
    } else {
      throw new Error(`Server returned status ${response.status}`);
    }
  } catch (error) {
    statusDot.className = 'status-dot offline';
    statusLabel.textContent = 'Backend Offline';
    statusLabel.title = `Cannot reach ${BACKEND_BASE_URL}. Ensure Flask app is running.`;
    console.warn('Backend health check failed:', error.message);
  }
}

// ==========================================================================
// 2. File Selection & Drag-and-Drop Handlers
// ==========================================================================

// Trigger hidden input on browse button click or drop zone click
browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener('click', () => {
  if (!selectedFile) {
    fileInput.click();
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileSelection(e.target.files[0]);
  }
});

// Drag & drop event listeners
['dragenter', 'dragover'].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');
  });
});

dropZone.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  if (dt && dt.files && dt.files[0]) {
    handleFileSelection(dt.files[0]);
  }
});

function handleFileSelection(file) {
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  
  if (!allowedTypes.includes(file.type)) {
    showAlert('error', '❌ Please select a valid road image (JPEG, PNG, or WebP).');
    return;
  }

  // Max 16 MB validation
  if (file.size > 16 * 1024 * 1024) {
    showAlert('error', '❌ File size exceeds 16 MB limit.');
    return;
  }

  selectedFile = file;

  // Render Image Preview
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
    
    dropZonePrompt.style.display = 'none';
    previewContainer.style.display = 'block';
    uploadBtn.disabled = false;
    hideAlert();
  };
  reader.readAsDataURL(file);
}

// ==========================================================================
// 3. Image Upload & OpenCV Validation Request
// ==========================================================================

uploadBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  setLoadingState(true);
  hideAlert();

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch(`${BACKEND_BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (response.ok && data.status === 'success') {
      showAlert(
        'success',
        `✔️ Image validated by OpenCV! Note: GAN & Defect models are pending integration.`
      );

      // Display OpenCV image metadata
      metaWidth.textContent = `${data.file_info.width_px} px`;
      metaHeight.textContent = `${data.file_info.height_px} px`;
      metaChannels.textContent = `${data.file_info.channels}`;
      metaSize.textContent = `${data.file_info.size_kb} KB`;
      metadataSection.style.display = 'block';

      // Log AI integration status from backend
      console.info('Backend Response:', data);
      console.info('AI Integration Status:', data.ai_integration.message);
    } else {
      showAlert('error', `❌ Upload Failed: ${data.message || 'Unknown error occurred.'}`);
    }
  } catch (error) {
    showAlert(
      'error',
      `❌ Connection Error: Unable to reach Flask backend at ${BACKEND_BASE_URL}. Ensure 'python app.py' is running.`
    );
    console.error('Upload Error:', error);
  } finally {
    setLoadingState(false);
  }
});

// ==========================================================================
// 4. Reset & Helper Utilities
// ==========================================================================

resetBtn.addEventListener('click', resetAll);

function resetAll() {
  selectedFile = null;
  fileInput.value = '';
  imagePreview.src = '';
  dropZonePrompt.style.display = 'block';
  previewContainer.style.display = 'none';
  uploadBtn.disabled = true;
  metadataSection.style.display = 'none';
  hideAlert();
}

function setLoadingState(isLoading) {
  if (isLoading) {
    uploadBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    uploadBtn.querySelector('.btn-text').textContent = 'Validating...';
  } else {
    uploadBtn.disabled = !selectedFile;
    btnSpinner.style.display = 'none';
    uploadBtn.querySelector('.btn-text').textContent = 'Upload & Inspect';
  }
}

function showAlert(type, message) {
  feedbackAlert.className = `alert-box ${type}`;
  alertMessage.textContent = message;
  alertIcon.textContent = type === 'success' ? '✅' : '⚠️';
  feedbackAlert.style.display = 'block';
}

function hideAlert() {
  feedbackAlert.style.display = 'none';
}

// Initial health check on application load
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  // Periodically check every 15 seconds
  setInterval(checkBackendHealth, 15000);
});
