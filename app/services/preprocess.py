"""Image preprocessing — MUST match exp5 training exactly.

exp5 was trained WITHOUT Ben Graham preprocessing, so we do NOT apply it here.
The pipeline is: read BGR -> convert to RGB -> resize -> scale to [0, 1] float32.
Verified correct in Phase 0.5 (grade-0 image predicted at 0.994 confidence).

Using any other preprocessing silently produces wrong predictions that are hard
to debug, so both the prediction and retraining paths import this single
function rather than reimplementing it.
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from app.config import TARGET_SIZE


def preprocess_image(path: Union[str, Path],
                     target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """Load and preprocess one image into a (H, W, 3) float32 array in [0, 1].

    Raises ValueError if the file cannot be read as an image.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.resize takes (width, height); TARGET_SIZE is (height, width)
    img = cv2.resize(img, (target_size[1], target_size[0]))
    return img.astype(np.float32) / 255.0


def preprocess_batch(path: Union[str, Path],
                     target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """Preprocess one image and add the batch dimension -> (1, H, W, 3)."""
    return np.expand_dims(preprocess_image(path, target_size), axis=0)
