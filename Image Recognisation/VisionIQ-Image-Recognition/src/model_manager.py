from __future__ import annotations
import time
import numpy as np
from src.config import IMAGE_SIZE, TOP_K, DEFAULT_MODEL, MODEL_INFO


class ModelManager:
    """Singleton-style model manager with lazy loading and caching."""

    _instances: dict = {}        # model_name → loaded model
    _current_name: str = DEFAULT_MODEL

    
    @classmethod
    def get_model(cls, model_name: str | None = None):
        """Return cached model or load it fresh."""
        name = model_name or cls._current_name
        if name not in cls._instances:
            cls._instances[name] = cls._load(name)
        cls._current_name = name
        return cls._instances[name], name

    @classmethod
    def _load(cls, name: str):
        """Load a pretrained Keras model by name."""
        import tensorflow as tf

        print(f"[ModelManager] Loading {name} …")
        t0 = time.time()

        if name == "mobilenetv2":
            model = tf.keras.applications.MobileNetV2(
                weights="imagenet", include_top=True
            )
        elif name == "resnet50":
            model = tf.keras.applications.ResNet50(
                weights="imagenet", include_top=True
            )
        elif name == "efficientnetb0":
            model = tf.keras.applications.EfficientNetB0(
                weights="imagenet", include_top=True
            )
        else:
            raise ValueError(f"Unknown model: {name}")

        print(f"[ModelManager] {name} loaded in {time.time()-t0:.1f}s")
        return model

    
    @classmethod
    def predict(cls, image_array: np.ndarray,
                model_name: str | None = None,
                top_k: int = TOP_K) -> dict:
        """
        Run inference on a preprocessed image array.

        Args:
            image_array: np.ndarray of shape (1, 224, 224, 3) — preprocessed
            model_name:  which model to use (switches if needed)
            top_k:       number of top predictions to return

        Returns:
            dict with predictions list and metadata
        """
        import tensorflow as tf

        model, name = cls.get_model(model_name)

        t0   = time.time()
        preds = model.predict(image_array, verbose=0)
        elapsed = (time.time() - t0) * 1000   # ms

        # Decode predictions to human labels
        if name == "mobilenetv2":
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
                preds, top=top_k
            )[0]
        elif name == "resnet50":
            decoded = tf.keras.applications.resnet50.decode_predictions(
                preds, top=top_k
            )[0]
        elif name == "efficientnetb0":
            decoded = tf.keras.applications.efficientnet.decode_predictions(
                preds, top=top_k
            )[0]
        else:
            decoded = []

        predictions = [
            {
                "rank":        i + 1,
                "class_id":    item[0],
                "label":       item[1].replace("_", " ").title(),
                "confidence":  round(float(item[2]) * 100, 2),
                "bar_width":   round(float(item[2]) * 100, 1),
            }
            for i, item in enumerate(decoded)
        ]

        return {
            "predictions":  predictions,
            "model_used":   MODEL_INFO[name]["display_name"],
            "model_key":    name,
            "inference_ms": round(elapsed, 1),
            "top_label":    predictions[0]["label"] if predictions else "Unknown",
            "top_conf":     predictions[0]["confidence"] if predictions else 0,
        }

    
    @classmethod
    def current_info(cls) -> dict:
        return MODEL_INFO.get(cls._current_name, {})

    @classmethod
    def loaded_models(cls) -> list[str]:
        return list(cls._instances.keys())

        

