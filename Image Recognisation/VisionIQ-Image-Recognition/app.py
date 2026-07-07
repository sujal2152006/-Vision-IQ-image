"""
app.py — Flask Application Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Routes:
  GET  /                  → Main UI
  POST /predict           → Upload image → get predictions
  POST /predict/url       → Predict from image URL
  GET  /api/models        → List available models
  POST /api/switch-model  → Switch active model
  GET  /api/health        → Health check
  GET  /uploads/<name>    → Serve uploaded images
"""

import os
import requests as req_lib
import tempfile
from flask import (Flask, render_template, request,
                   jsonify, send_from_directory, url_for)

from src.config      import (UPLOAD_FOLDER, MAX_CONTENT_LENGTH,
                              MODEL_INFO, DEFAULT_MODEL, SECRET_KEY,
                              DEBUG, HOST, PORT)
from src.preprocessor  import ImagePreprocessor
from src.model_manager import ModelManager

# ── App init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SECRET_KEY"]         = SECRET_KEY
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Warm up default model at startup ──────────────────────────────────────────
# @app.before_request
# def _warm_up():
#     """Load default model on first request so first prediction is fast."""
#     if not ModelManager.loaded_models():
#         try:
#             ModelManager.get_model(DEFAULT_MODEL)
#         except Exception as e:
#             print(f"[Startup] Model warmup skipped: {e}")

# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # return render_template(
    #     "index.html",
    #     model_info=MODEL_INFO,
    #     current_model=DEFAULT_MODEL,
    # )
    return render_template("index.html")

# ── Serve uploaded images ──────────────────────────────────────────────────────
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ── Main prediction endpoint ───────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts multipart/form-data with:
      - file:  image file
      - model: (optional) model name

    Returns JSON prediction results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file       = request.files["file"]
    model_name = request.form.get("model", DEFAULT_MODEL)

    # ── Validate ──
    ok, err = ImagePreprocessor.validate(file)
    if not ok:
        return jsonify({"error": err}), 400

    # ── Save ──
    try:
        filename, filepath = ImagePreprocessor.save(
            file, app.config["UPLOAD_FOLDER"]
        )
    except Exception as e:
        return jsonify({"error": f"Could not save file: {e}"}), 500

    # ── Preprocess + Predict ──
    try:
        img      = ImagePreprocessor.load_image(filepath)
        arr      = ImagePreprocessor.preprocess(img, model_name)
        result   = ModelManager.predict(arr, model_name)
        meta     = ImagePreprocessor.get_metadata(filepath)
        cv_data  = ImagePreprocessor.cv_analysis(filepath)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({
        **result,
        "image_url":  url_for("uploaded_file", filename=filename),
        "filename":   filename,
        "metadata":   meta,
        "cv_analysis": cv_data,
    })

# ── Predict from URL ───────────────────────────────────────────────────────────
@app.route("/predict/url", methods=["POST"])
def predict_url():
    """Download image from URL and run prediction."""
    body      = request.get_json(silent=True) or {}
    image_url = (body.get("url") or "").strip()
    model_name = body.get("model", DEFAULT_MODEL)

    if not image_url:
        return jsonify({"error": "URL is required"}), 400

    try:
        response = req_lib.get(image_url, timeout=10, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "image" not in content_type:
            return jsonify({"error": "URL does not point to an image"}), 400

        # Save to temp file
        ext = image_url.split(".")[-1].split("?")[0][:4] or "jpg"
        with tempfile.NamedTemporaryFile(
            suffix=f".{ext}", dir=UPLOAD_FOLDER, delete=False
        ) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        tmp_filename = os.path.basename(tmp_path)

    except Exception as e:
        return jsonify({"error": f"Could not download image: {e}"}), 400

    try:
        img     = ImagePreprocessor.load_image(tmp_path)
        arr     = ImagePreprocessor.preprocess(img, model_name)
        result  = ModelManager.predict(arr, model_name)
        meta    = ImagePreprocessor.get_metadata(tmp_path)
        cv_data = ImagePreprocessor.cv_analysis(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({
        **result,
        "image_url":   url_for("uploaded_file", filename=tmp_filename),
        "source_url":  image_url,
        "metadata":    meta,
        "cv_analysis": cv_data,
    })

# ── API: Models ────────────────────────────────────────────────────────────────
@app.route("/api/models")
def api_models():
    return jsonify({
        "models":        MODEL_INFO,
        "current_model": ModelManager._current_name,
        "loaded":        ModelManager.loaded_models(),
    })

@app.route("/api/switch-model", methods=["POST"])
def switch_model():
    body = request.get_json(silent=True) or {}
    name = body.get("model", "").strip()
    if name not in MODEL_INFO:
        return jsonify({"error": f"Unknown model: {name}. Choose from: {list(MODEL_INFO)}"}), 400
    try:
        ModelManager.get_model(name)
        return jsonify({"success": True, "active_model": name, "info": MODEL_INFO[name]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: Health ────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    import tensorflow as tf
    return jsonify({
        "status":        "ok",
        "tensorflow":    tf.__version__,
        "current_model": ModelManager._current_name,
        "loaded_models": ModelManager.loaded_models(),
        "gpu_available": bool(tf.config.list_physical_devices("GPU")),
    })

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Image Recognition System → http://localhost:5000")
    print(f"   Default model: {DEFAULT_MODEL}")
    app.run(debug=False, host=HOST, port=PORT)