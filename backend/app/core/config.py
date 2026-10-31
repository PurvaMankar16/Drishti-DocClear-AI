import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BACKEND_DIR / "uploads"
PROCESSED_DIR = BACKEND_DIR / "processed"
ENHANCED_DIR = PROCESSED_DIR / "enhanced"
REPORTS_DIR = PROCESSED_DIR / "reports"

ML_CHECKPOINT_PATH = BACKEND_DIR / "models" / "best_resnet18.pth"
LAYOUT_CHECKPOINT_PATH = BACKEND_DIR / "models" / "best_fasterrcnn_doclaynet.pth"
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}
ALLOWED_PDF_MIME_TYPES = {"application/pdf"}

# Initial engineering thresholds only; tune with a validation dataset later.
BLUR_THRESHOLDS = {"good": 150.0, "warning": 50.0}
BRIGHTNESS_THRESHOLDS = {"good_min": 70.0, "good_max": 190.0, "warning_min": 45.0, "warning_max": 220.0}
CONTRAST_THRESHOLDS = {"good": 45.0, "warning": 25.0}
NOISE_THRESHOLDS = {"good": 8.0, "warning": 18.0}
SKEW_THRESHOLDS = {"good": 1.0, "warning": 4.0}
RESOLUTION_THRESHOLDS = {"good_megapixels": 1.0, "warning_megapixels": 0.4}

# Initial weights only; they are not scientifically validated.
QUALITY_WEIGHTS = {
    "blur": 0.25,
    "brightness": 0.15,
    "contrast": 0.15,
    "noise": 0.10,
    "skew": 0.10,
    "cropping": 0.15,
    "resolution": 0.10,
}
OVERALL_CLASSIFICATION_THRESHOLDS = {"acceptable": 70.0, "needs_enhancement": 40.0}

# Layout detection model configuration
LAYOUT_CONFIDENCE_THRESHOLD = 0.5
LAYOUT_NMS_IOU_THRESHOLD = 0.3
LAYOUT_INPUT_SIZE = 1025  # Model was trained with 1025x1025 images
