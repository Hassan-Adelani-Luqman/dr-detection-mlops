"""Thread-safe model serving.

FastAPI serves requests concurrently and the retrain/rollback flows swap the
active model at runtime. A single reentrant lock guards load, reload, and
predict so a request never runs inference against a half-swapped model.

The model is loaded once at startup (via ``load_current_model``) and cached at
module level, not reloaded per request.
"""
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Keep TF import local-friendly: it is heavy but imported once here.
import tensorflow as tf

from app.config import CLASS_LABELS, CLINICAL_NOTES, CURRENT_MODEL_PATH
from app.services.preprocess import preprocess_batch
from app.services.registry import get_active_version

_lock = threading.RLock()
_model: Optional["tf.keras.Model"] = None


def load_current_model() -> None:
    """Load exp5_current.keras into the module cache. Call once at startup."""
    global _model
    with _lock:
        _model = tf.keras.models.load_model(CURRENT_MODEL_PATH)


def reload_model() -> None:
    """Swap the cached model after a retrain or rollback (thread-safe)."""
    global _model
    with _lock:
        _model = tf.keras.models.load_model(CURRENT_MODEL_PATH)


def is_loaded() -> bool:
    return _model is not None


def predict(image_path: Union[str, Path]) -> dict:
    """Run inference on one image and return a structured result dict."""
    x = preprocess_batch(image_path)  # (1, 224, 224, 3)
    with _lock:
        if _model is None:
            load_current_model()
        probs = _model.predict(x, verbose=0)[0]

    grade = int(np.argmax(probs))
    confidence_scores = {
        CLASS_LABELS[i]: round(float(probs[i]), 4) for i in range(len(probs))
    }
    return {
        "predicted_grade": grade,
        "grade_label": CLASS_LABELS[grade],
        "clinical_note": CLINICAL_NOTES[grade],
        "confidence": round(float(probs[grade]), 4),
        "confidence_scores": confidence_scores,
        "active_version": get_active_version(),
    }
