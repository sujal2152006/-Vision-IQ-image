<div align="center">

# 🔍 VisionIQ — AI Image Recognition System

**A production-ready deep learning web app that identifies objects in any image using pre-trained CNN models**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/Bansaripatel02/VisionIQ-Image-Recognition/actions)

<br/>

*3 CNN Models · 1000 ImageNet Classes · Drag & Drop Upload · URL Prediction · Real-time Inference*

<br/>

[🎯 Problem Statement](#-problem-statement) · [✨ Features](#-features) · [⚡ Quick Start](#-quick-start) · [🔌 API Docs](#-api-reference) · [🏗 Architecture](#-architecture)

</div>

---

## 🎯 Problem Statement

Computer vision is one of the most impactful areas of AI — but building a working image classification system from scratch means wiring together TensorFlow, pretrained weights, image preprocessing pipelines, and a web server, all at once.

**VisionIQ** solves this by packaging three state-of-the-art pretrained CNN models behind a clean REST API and a drag-and-drop web interface — **zero training required, zero API key needed, runs entirely on your machine.**

### Real-World Use Cases
| Domain | Application |
|---|---|
| 🛒 **E-commerce** | Automatic product image categorization |
| 🏥 **Healthcare** | Pre-screening triage for medical image review |
| 🛡 **Content Moderation** | Automated object & scene detection |
| 🎓 **Education** | Visual demonstration of CNN architectures & transfer learning |
| 🔬 **Research** | Rapid prototyping for computer vision experiments |

---

## ✨ Features

### 🧠 3 Switchable CNN Models — Hot-swap at Runtime

| Model | Top-1 Accuracy | Size | Speed | Best For |
|---|---|---|---|---|
| **MobileNetV2** | 71.8% ImageNet | 14 MB | ⚡ Fast | Real-time demos, web apps |
| **ResNet-50** | 74.9% ImageNet | 98 MB | ⚖ Balanced | General purpose classification |
| **EfficientNet-B0** | 77.1% ImageNet | 29 MB | ⚖ Balanced | Maximum accuracy per parameter |

> All models use **ImageNet pretrained weights** — auto-downloaded by Keras on first use. No manual setup.

### 📤 Dual Input Methods
- **Drag & Drop / File Upload** — PNG, JPG, JPEG, GIF, BMP, WEBP · Max 10 MB
- **Image URL** — Paste any public image URL and predict instantly

### 🎯 Prediction Output
- **Top-5 predictions** with confidence scores and animated progress bars
- **Inference time** displayed in milliseconds
- **1000 ImageNet object classes** — animals, vehicles, food, furniture, electronics, and more

### 🔬 OpenCV Image Analysis Pipeline
Every uploaded image is automatically analysed for:
| Metric | Method |
|---|---|
| **Brightness** | Mean pixel intensity (0–255) |
| **Contrast** | Standard deviation of grayscale |
| **Edge Density** | Canny edge detection — % of edge pixels |
| **Dominant Colour** | K-Means clustering (k=1) on RGB pixels |

### 🖥 Professional Web UI
- Dark neural-themed interface with Space Grotesk typography
- **Scan-beam animation** over image during inference — visualises model processing
- Fully **responsive** — works on desktop and mobile
- Animated confidence bars with colour gradient per rank
- Model info panel with one-click switching

### 🔌 REST API
- `POST /predict` — multipart file upload
- `POST /predict/url` — JSON with image URL
- `GET /api/models` — list all models and current active
- `POST /api/switch-model` — hot-swap model without restart
- `GET /api/health` — TensorFlow version, GPU status, loaded models

---

## 🛠 Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Web Framework** | Flask | 3.1.1 | REST API + file upload + template rendering |
| **Deep Learning** | TensorFlow / Keras | 2.19.0 | Pretrained CNN models + ImageNet decode |
| **Image Processing** | Pillow | 11.2.1 | Image load, EXIF rotation fix, RGB convert, resize |
| **Computer Vision** | OpenCV | 4.11.0 | Brightness, contrast, Canny edges, K-Means colour |
| **Numerics** | NumPy | 1.26.4 | Array operations & preprocessing |
| **HTTP Client** | Requests | 2.32.3 | Download images from URLs for prediction |
| **Testing** | pytest | 8.2.2 | Unit + integration tests for all modules |
| **CI/CD** | GitHub Actions | — | Auto-runs test suite on every push to main |

---

## 📁 Project Structure

```
VisionIQ-Image-Recognition/
│
├── 📄 app.py                        # Flask entry point — all 7 routes
│
├── 📂 src/
│   ├── __init__.py
│   ├── config.py                    # Paths, model info, upload limits, Flask settings
│   ├── model_manager.py             # Lazy load · in-memory cache · predict · decode
│   └── preprocessor.py             # Validate · save · EXIF fix · resize · OpenCV analysis
│
├── 📂 templates/
│   └── index.html                   # Single-page web app with scan animation
│
├── 📂 static/
│   ├── css/
│   │   └── style.css                # Dark neural theme — Space Grotesk + grid background
│   └── js/
│       └── app.js                   # Drag-drop · file select · URL predict · model switch
│
├── 📂 tests/
│   └── test_system.py               # 9 pytest tests — config, preprocessor, Flask routes
│
├── 📂 uploads/                      # Auto-created — uploaded images stored here
│   └── .gitkeep
│
├── 📂 .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI pipeline
│
├── 📄 requirements.txt              # Pinned production dependencies
├── 📄 .gitignore
└── 📄 README.md
```

---

## ⚡ Quick Start

### Prerequisites
- **Python 3.12** (64-bit) — [Download here](https://www.python.org/downloads/release/python-31210/)
- **Git** — [Download here](https://git-scm.com/downloads)
- ~500 MB free disk space (TensorFlow installation)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Bansaripatel02/VisionIQ-Image-Recognition.git
cd VisionIQ-Image-Recognition

# 2. Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

**Open your browser → [http://localhost:5000](http://localhost:5000)** 🎉

> **First prediction note:** Keras automatically downloads pretrained model weights
> (~14 MB for MobileNetV2) on first use. This is a one-time download — subsequent
> predictions are instant from cache.

---

## 🖥 Usage Guide

### Upload & Predict
1. **Drag and drop** any image onto the upload zone — or click to browse
2. **Select your model** — MobileNetV2 (fast), ResNet-50 (balanced), EfficientNet-B0 (accurate)
3. Click **Analyze Image** — watch the scan animation as inference runs
4. View **Top-5 predictions** with confidence scores, image metadata, and OpenCV analysis

### Predict from URL
1. Paste any public image URL into the URL input box
2. Press **Enter** or click the arrow button
3. The app downloads the image and returns predictions instantly

### Switch Models
- Use the **model tabs** directly below the upload zone, or
- Click **⚙ Switch Model** in the navbar to open the model info panel
- Models are **hot-swapped** — no server restart required

---

## 🔌 API Reference

### `POST /predict` — File Upload
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/your/image.jpg" \
  -F "model=mobilenetv2"
```

### `POST /predict/url` — Image URL
```bash
curl -X POST http://localhost:5000/predict/url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
    "model": "resnet50"
  }'
```

### Sample Response
```json
{
  "top_label": "Golden Retriever",
  "top_conf": 91.4,
  "model_used": "ResNet-50",
  "model_key": "resnet50",
  "inference_ms": 87.3,
  "predictions": [
    { "rank": 1, "label": "Golden Retriever", "confidence": 91.4 },
    { "rank": 2, "label": "Labrador Retriever","confidence": 5.2 },
    { "rank": 3, "label": "Kuvasz",            "confidence": 1.1 },
    { "rank": 4, "label": "Great Pyrenees",    "confidence": 0.9 },
    { "rank": 5, "label": "Clumber Spaniel",   "confidence": 0.6 }
  ],
  "metadata": {
    "width": 320,
    "height": 240,
    "format": "JPEG",
    "mode": "RGB",
    "size_kb": 28.4
  },
  "cv_analysis": {
    "brightness": 158.3,
    "contrast": 52.7,
    "edge_density": 9.8,
    "dominant_color": "#c8a060"
  }
}
```

### `GET /api/health`
```bash
curl http://localhost:5000/api/health
```
```json
{
  "status": "ok",
  "tensorflow": "2.19.0",
  "current_model": "mobilenetv2",
  "loaded_models": ["mobilenetv2"],
  "gpu_available": false
}
```

### `POST /api/switch-model`
```bash
curl -X POST http://localhost:5000/api/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model": "efficientnetb0"}'
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

```
tests/test_system.py::TestConfig::test_model_info_keys          PASSED
tests/test_system.py::TestConfig::test_allowed_extensions       PASSED
tests/test_system.py::TestConfig::test_image_size               PASSED
tests/test_system.py::TestPreprocessor::test_load_image_shape   PASSED
tests/test_system.py::TestPreprocessor::test_get_metadata       PASSED
tests/test_system.py::TestPreprocessor::test_validate_bad_ext   PASSED
tests/test_system.py::TestPreprocessor::test_validate_no_file   PASSED
tests/test_system.py::TestFlaskRoutes::test_index_returns_200   PASSED
tests/test_system.py::TestFlaskRoutes::test_health_endpoint     PASSED

========================= 9 passed ✅ =========================
```

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Browser (Web UI)                              │
│         Drag & Drop Upload  ·  URL Input  ·  Model Selector          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  HTTP POST (multipart / JSON)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Flask  app.py                                │
│   /predict   /predict/url   /api/models   /api/switch-model          │
└──────┬──────────────────────────┬──────────────────────────────┬─────┘
       │                          │                              │
┌──────▼──────────┐   ┌───────────▼───────────┐   ┌────────────▼──────┐
│  ImagePreprocess│   │    ModelManager        │   │   OpenCV Analysis │
│                 │   │                        │   │                   │
│ validate()      │   │  get_model(name)       │   │  brightness       │
│ save() w/ UUID  │   │  _load() → Keras       │   │  contrast         │
│ EXIF rotation   │   │  _instances cache      │   │  Canny edges      │
│ RGB convert     │   │  predict() → decode    │   │  K-Means colour   │
│ resize 224×224  │   │  top-5 + confidence    │   │                   │
└─────────────────┘   └────────────────────────┘   └───────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   TensorFlow / Keras   │
                    │                        │
                    │   MobileNetV2  (14MB)  │
                    │   ResNet-50    (98MB)  │
                    │   EfficientNetB0(29MB) │
                    │                        │
                    │   ImageNet weights     │
                    │   1000 classes         │
                    └────────────────────────┘
```

### Key Design Decisions

**Why lazy loading + in-memory caching?**
> TensorFlow models take 2–5 seconds to load from disk. ModelManager loads each model
> only on first use and keeps it in RAM — subsequent predictions on the same model are
> instant. Switching models mid-session requires no server restart.

**Why separate `preprocessor.py` from `model_manager.py`?**
> Single Responsibility Principle. Preprocessor handles I/O (file system, image
> manipulation, OpenCV). ModelManager handles inference (TensorFlow, weights,
> prediction decoding). Each is independently testable and swappable.

**Why UUID filenames for uploads?**
> Prevents filename collisions, path traversal attacks, and overwrites.
> Every uploaded file gets a unique `hex_uuid.ext` name regardless of original filename.

**Why EXIF transpose before resize?**
> Photos taken on phones embed rotation metadata in EXIF. Without `ImageOps.exif_transpose()`,
> portrait photos appear sideways — causing wrong predictions.

---

## 🌐 Deployment

### Streamlit Cloud / Railway / Render (Free)
```bash
# Set environment variable for production
export SECRET_KEY="your-strong-secret-key-here"
export FLASK_DEBUG="false"

# Run with Gunicorn (production WSGI server)
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t visioniq .
docker run -p 5000:5000 visioniq
```

---

## 🗺 Roadmap

- [ ] **Grad-CAM heatmap** — highlight which pixels the model focused on
- [ ] **Batch prediction** — upload multiple images at once (ZIP file)
- [ ] **Custom model upload** — load your own `.keras` fine-tuned model
- [ ] **Object detection** — YOLO integration for bounding box output
- [ ] **Prediction history** — SQLite log with thumbnail gallery
- [ ] **YOLOv8 support** — real-time object detection with class counts
- [ ] **Docker + Gunicorn** — one-command production deployment

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

```bash
# 1. Fork the repository on GitHub
# 2. Create your feature branch
git checkout -b feature/grad-cam-heatmap

# 3. Make your changes and commit
git commit -m "feat: add Grad-CAM activation heatmap overlay"

# 4. Push and open a Pull Request
git push origin feature/grad-cam-heatmap
```

**Before submitting:**
- [ ] `pytest tests/ -v` — all 9 tests must pass
- [ ] New features should have corresponding unit tests in `tests/`
- [ ] Follow existing code style in `src/` modules

---

## 📜 License

Distributed under the **MIT License** — see [LICENSE](LICENSE) for details.
Free to use, modify, and distribute with attribution.

---

## 👤 Author

**Bansari Patel**

[![GitHub](https://img.shields.io/badge/GitHub-Bansaripatel02-181717?style=flat-square&logo=github)](https://github.com/Bansaripatel02)

---

<div align="center">

**⭐ If VisionIQ helped you learn or ship faster, please star the repo!**

*Built as part of a Deep Learning & Python portfolio*

*Model weights © ImageNet Large Scale Visual Recognition Challenge (ILSVRC)*

*⚠️ For educational and research purposes only · Not for clinical or safety-critical use*

</div>
