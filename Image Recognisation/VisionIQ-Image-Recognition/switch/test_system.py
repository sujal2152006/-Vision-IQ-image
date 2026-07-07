"""
tests/test_system.py
━━━━━━━━━━━━━━━━━━━━
Run: pytest tests/ -v
"""

import io, os, sys, types
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_image_path(tmp_path):
    """Create a real PNG test image using PIL."""
    from PIL import Image
    img  = Image.fromarray(np.uint8(np.random.rand(300, 400, 3) * 255))
    path = str(tmp_path / "test.jpg")
    img.save(path)
    return path


# ── Config Tests ──────────────────────────────────────────────────────────────
class TestConfig:
    def test_model_info_keys(self):
        from src.config import MODEL_INFO
        assert "mobilenetv2"    in MODEL_INFO
        assert "resnet50"       in MODEL_INFO
        assert "efficientnetb0" in MODEL_INFO

    def test_allowed_extensions(self):
        from src.config import ALLOWED_EXTENSIONS
        assert "jpg"  in ALLOWED_EXTENSIONS
        assert "png"  in ALLOWED_EXTENSIONS
        assert "webp" in ALLOWED_EXTENSIONS
        assert "pdf"  not in ALLOWED_EXTENSIONS

    def test_image_size(self):
        from src.config import IMAGE_SIZE
        assert IMAGE_SIZE == (224, 224)


# ── Preprocessor Tests ────────────────────────────────────────────────────────
class TestPreprocessor:
    def test_load_image_shape(self, sample_image_path):
        from src.preprocessor import ImagePreprocessor
        img = ImagePreprocessor.load_image(sample_image_path)
        assert img.size == (224, 224)
        assert img.mode == "RGB"

    def test_get_metadata(self, sample_image_path):
        from src.preprocessor import ImagePreprocessor
        meta = ImagePreprocessor.get_metadata(sample_image_path)
        assert meta["width"]  > 0
        assert meta["height"] > 0
        assert meta["size_kb"] > 0

    def test_validate_bad_extension(self):
        from src.preprocessor import ImagePreprocessor

        class FakeFile:
            filename = "document.pdf"
            def seek(self, *a): pass
            def tell(self): return 100

        ok, msg = ImagePreprocessor.validate(FakeFile())
        assert not ok
        assert "pdf" in msg.lower() or "not supported" in msg.lower()

    def test_validate_no_filename(self):
        from src.preprocessor import ImagePreprocessor

        class FakeFile:
            filename = ""

        ok, msg = ImagePreprocessor.validate(FakeFile())
        assert not ok

    def test_save_creates_file(self, sample_image_path, tmp_path):
        from src.preprocessor import ImagePreprocessor

        class FakeFile:
            filename = "photo.jpg"
            def save(self, path):
                import shutil; shutil.copy(sample_image_path, path)

        fname, fpath = ImagePreprocessor.save(FakeFile(), str(tmp_path))
        assert os.path.exists(fpath)
        assert fname.endswith(".jpg")


# ── Flask App Tests ───────────────────────────────────────────────────────────
class TestFlaskRoutes:
    @pytest.fixture
    def client(self, tmp_path):
        """Create Flask test client with temp upload folder."""
        os.environ["FLASK_TESTING"] = "1"
        import app as flask_app
        flask_app.app.config["TESTING"]       = True
        flask_app.app.config["UPLOAD_FOLDER"] = str(tmp_path)
        with flask_app.app.test_client() as c:
            yield c

    def test_index_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"VisionIQ" in res.data or b"Image" in res.data

    def test_health_endpoint(self, client):
        res  = client.get("/api/health")
        data = res.get_json()
        assert res.status_code == 200
        assert data["status"] == "ok"
        assert "tensorflow" in data

    def test_models_endpoint(self, client):
        res  = client.get("/api/models")
        data = res.get_json()
        assert "models" in data
        assert "mobilenetv2" in data["models"]

    def test_predict_no_file(self, client):
        res = client.post("/predict")
        assert res.status_code == 400
        assert "error" in res.get_json()

    def test_predict_invalid_file(self, client):
        res = client.post("/predict", data={
            "file": (io.BytesIO(b"not an image"), "test.pdf"),
        }, content_type="multipart/form-data")
        assert res.status_code == 400

    def test_switch_model_invalid(self, client):
        res = client.post("/api/switch-model",
                          json={"model": "does_not_exist"},
                          content_type="application/json")
        assert res.status_code == 400