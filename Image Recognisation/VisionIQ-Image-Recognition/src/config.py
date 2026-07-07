"""
src/config.py — Central configuration for Image Recognition System
All tuneable constants live here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER  = os.path.join(BASE_DIR, "models")

# ── Upload Settings ────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB  = 10
MAX_CONTENT_LENGTH = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# ── Model Settings ─────────────────────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)          # MobileNetV2 input size
TOP_K           = 5                   # Number of top predictions to return
CONFIDENCE_MIN  = 0.01                # Minimum confidence to show (1%)

# Model options: "mobilenetv2" | "resnet50" | "efficientnetb0"
DEFAULT_MODEL   = "mobilenetv2"

MODEL_INFO = {
    "mobilenetv2": {
        "display_name": "MobileNetV2",
        "description":  "Fast & lightweight — best for real-time use",
        "size_mb":      14,
        "accuracy":     "71.8% Top-1 ImageNet",
        "speed":        "Fast",
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "description":  "Balanced accuracy & speed",
        "size_mb":      98,
        "accuracy":     "74.9% Top-1 ImageNet",
        "speed":        "Medium",
    },
    "efficientnetb0": {
        "display_name": "EfficientNet-B0",
        "description":  "Best accuracy for its size",
        "size_mb":      29,
        "accuracy":     "77.1% Top-1 ImageNet",
        "speed":        "Medium",
    },
}

# ── Flask Settings ─────────────────────────────────────────────────────────────
DEBUG        = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
SECRET_KEY   = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
HOST         = "0.0.0.0"
PORT         = 5000

# ── Preprocessing ──────────────────────────────────────────────────────────────
PREPROCESS_MAP = {
    "mobilenetv2":   "tensorflow.keras.applications.mobilenet_v2",
    "resnet50":      "tensorflow.keras.applications.resnet50",
    "efficientnetb0":"tensorflow.keras.applications.efficientnet",
}