"""
src/preprocessor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles image loading, validation, resizing, and preprocessing
for each supported model architecture.

Pipeline:
  raw bytes → PIL Image → resize → numpy array → model preprocess
"""

from __future__ import annotations
import io
import os
import uuid
import numpy as np
from PIL import Image, ImageOps
from src.config import IMAGE_SIZE, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB


class ImagePreprocessor:

    # ── Validate ──────────────────────────────────────────────────────────────
    @staticmethod
    def validate(file) -> tuple[bool, str]:
        """Validate uploaded file. Returns (ok, error_message)."""
        filename = getattr(file, "filename", "")
        if not filename:
            return False, "No file selected"

        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"File type '.{ext}' not supported. Use: {', '.join(ALLOWED_EXTENSIONS)}"

        # Check file size
        file.seek(0, 2)          # seek to end
        size_mb = file.tell() / (1024 * 1024)
        file.seek(0)             # reset
        if size_mb > MAX_FILE_SIZE_MB:
            return False, f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB"

        return True, ""

    # ── Save ──────────────────────────────────────────────────────────────────
    @staticmethod
    def save(file, upload_folder: str) -> tuple[str, str]:
        """
        Save uploaded file with a unique name.
        Returns (saved_filename, full_path)
        """
        os.makedirs(upload_folder, exist_ok=True)
        ext      = file.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        path     = os.path.join(upload_folder, filename)
        file.save(path)
        return filename, path

    # ── Load & Resize ─────────────────────────────────────────────────────────
    @staticmethod
    def load_image(path: str, target_size: tuple = IMAGE_SIZE) -> Image.Image:
        """Load image from path, convert to RGB, resize."""
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)       # fix rotation from EXIF
        img = img.convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        return img

    # ── Preprocess for model ──────────────────────────────────────────────────
    @staticmethod
    def preprocess(img: Image.Image, model_name: str) -> np.ndarray:
        """
        Convert PIL Image to model-ready numpy array.
        Each architecture expects different pixel normalization.
        """
        import tensorflow as tf

        arr = tf.keras.preprocessing.image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)         # shape: (1, 224, 224, 3)

        if model_name == "mobilenetv2":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        elif model_name == "resnet50":
            from tensorflow.keras.applications.resnet50 import preprocess_input
        elif model_name == "efficientnetb0":
            from tensorflow.keras.applications.efficientnet import preprocess_input
        else:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        return preprocess_input(arr)

    # ── Image Metadata ────────────────────────────────────────────────────────
    @staticmethod
    def get_metadata(path: str) -> dict:
        """Extract image metadata for display."""
        try:
            img = Image.open(path)
            size_kb = round(os.path.getsize(path) / 1024, 1)
            return {
                "width":    img.width,
                "height":   img.height,
                "format":   img.format or path.rsplit(".", 1)[-1].upper(),
                "mode":     img.mode,
                "size_kb":  size_kb,
            }
        except Exception:
            return {}

    # ── OpenCV Analysis (optional enrichment) ─────────────────────────────────
    @staticmethod
    def cv_analysis(path: str) -> dict:
        """
        Basic OpenCV image analysis:
        brightness, contrast, detected edges, dominant color.
        Returns empty dict if OpenCV unavailable.
        """
        try:
            import cv2

            img_bgr = cv2.imread(path)
            if img_bgr is None:
                return {}

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Brightness (mean pixel value 0-255)
            brightness = round(float(np.mean(gray)), 1)

            # Contrast (std deviation)
            contrast = round(float(np.std(gray)), 1)

            # Edge density (Canny)
            edges     = cv2.Canny(gray, 100, 200)
            edge_pct  = round(float(np.count_nonzero(edges)) / edges.size * 100, 1)

            # Dominant color (k-means k=1 on resized img)
            small = cv2.resize(img_bgr, (50, 50))
            pixels = small.reshape(-1, 3).astype(np.float32)
            _, _, centers = cv2.kmeans(
                pixels, 1, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                3, cv2.KMEANS_RANDOM_CENTERS
            )
            b, g, r = centers[0]
            dominant_hex = f"#{int(r):02x}{int(g):02x}{int(b):02x}"

            return {
                "brightness":    brightness,
                "contrast":      contrast,
                "edge_density":  edge_pct,
                "dominant_color": dominant_hex,
            }
        except Exception:
            return {}