"""Thread-safe model serving.

FastAPI serves requests concurrently and the retrain/rollback flows swap the
active model at runtime. A single reentrant lock guards load, reload, and
inference so a request never runs against a half-swapped model.

The model is loaded once at startup (via ``load_current_model``) and cached at
module level, not reloaded per request. Image decoding happens outside the
lock; only the forward pass is serialised. The ML itself (preprocessing,
inference, result formatting) lives in ``src/``.
"""
import threading
from pathlib import Path
from typing import Optional, Union

import tensorflow as tf

from app.config import CURRENT_MODEL_PATH
from app.services.registry import get_active_version
from src.model import load_model
from src.prediction import format_prediction
from src.preprocessing import preprocess_batch

_lock = threading.RLock()
_model: Optional["tf.keras.Model"] = None


def load_current_model() -> None:
    """Load exp5_current.keras into the module cache. Call once at startup."""
    global _model
    with _lock:
        _model = load_model(CURRENT_MODEL_PATH)


def reload_model() -> None:
    """Swap the cached model after a retrain or rollback (thread-safe)."""
    load_current_model()


def is_loaded() -> bool:
    return _model is not None


def predict(image_path: Union[str, Path]) -> dict:
    """Run inference on one image and return a structured result dict."""
    x = preprocess_batch(image_path)  # (1, 224, 224, 3); raises ValueError if unreadable
    with _lock:
        if _model is None:
            load_current_model()
        probs = _model.predict(x, verbose=0)[0]

    result = format_prediction(probs)
    result["active_version"] = get_active_version()
    return result
