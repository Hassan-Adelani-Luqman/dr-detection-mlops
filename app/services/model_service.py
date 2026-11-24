"""Thread-safe model serving.

Prediction endpoints run in FastAPI's thread pool, and the retrain/rollback
flows swap the active model at runtime. A single reentrant lock guards load,
reload and inference, so a request never runs against a half-swapped model.
Image decoding happens outside the lock; only the forward pass is serialised
(one inference at a time per process; scale out with more containers).

The model is cached at module level and tagged with the registry version it
was loaded as. Before each prediction the registry's active version is checked:
if another process (another container behind the load balancer) retrained or
rolled back, this process reloads too, so every replica serves the same version.

The ML itself (preprocessing, result formatting) lives in ``src/``.
"""
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from app.config import CURRENT_MODEL_PATH
from app.services.registry import get_active_version
from src.model import load_model
from src.prediction import format_prediction
from src.preprocessing import preprocess_batch, preprocess_image_bytes

_lock = threading.RLock()
_model: Optional["tf.keras.Model"] = None
_loaded_version: Optional[str] = None


def load_current_model() -> None:
    """Load exp5_current.keras into the module cache, tagged with the active version."""
    global _model, _loaded_version
    with _lock:
        version = get_active_version()
        _model = load_model(CURRENT_MODEL_PATH)
        _loaded_version = version


def reload_model() -> None:
    """Swap the cached model after a retrain or rollback (thread-safe)."""
    load_current_model()


def is_loaded() -> bool:
    return _model is not None


def loaded_version() -> Optional[str]:
    """The registry version the cached model was loaded as."""
    return _loaded_version


def _infer(x: np.ndarray) -> dict:
    active = get_active_version()
    with _lock:
        if _model is None or _loaded_version != active:
            load_current_model()
        probs = _model.predict(x, verbose=0)[0]
        version = _loaded_version
    result = format_prediction(probs)
    result["active_version"] = version
    return result


def predict(image_path: Union[str, Path]) -> dict:
    """Grade one image file. Raises ValueError if it is not a readable image."""
    return _infer(preprocess_batch(image_path))


def predict_bytes(data: bytes) -> dict:
    """Grade one image held in memory (an upload). Raises ValueError if undecodable."""
    return _infer(np.expand_dims(preprocess_image_bytes(data), axis=0))
