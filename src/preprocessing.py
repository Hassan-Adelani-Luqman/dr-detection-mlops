"""Image preprocessing and the train/val/test split.

The image pipeline MUST match how exp5 was trained: read BGR, convert to RGB,
resize to 224x224, scale to [0, 1] float32. exp5 was trained WITHOUT Ben Graham
preprocessing, so it is not applied here. Prediction, retraining and
evaluation all call these functions, so the paths can never drift apart.
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SPLIT_SEED, TARGET_SIZE, TEST_SIZE, VAL_SIZE

PathLike = Union[str, Path]


def preprocess_image(path: PathLike, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """Load one image as a (H, W, 3) float32 RGB array scaled to [0, 1].

    Raises ValueError if the file cannot be read as an image.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.resize takes (width, height); target_size is (height, width).
    img = cv2.resize(img, (target_size[1], target_size[0]))
    return img.astype(np.float32) / 255.0


def preprocess_batch(path: PathLike, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """Preprocess one image and add the batch dimension -> (1, H, W, 3)."""
    return np.expand_dims(preprocess_image(path, target_size), axis=0)


def load_labels(csv_path: PathLike) -> pd.DataFrame:
    """Read the APTOS ``train.csv`` and check it has ``id_code`` and ``diagnosis``."""
    df = pd.read_csv(csv_path)
    missing = {"id_code", "diagnosis"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    return df


def stratified_split(df: pd.DataFrame, val_size: float = VAL_SIZE,
                     test_size: float = TEST_SIZE, seed: int = SPLIT_SEED):
    """Two-stage stratified train/val/test split, identical to the notebook's.

    Stage 1 separates the test set; stage 2 splits the rest into train and
    val. Both stratify on ``diagnosis``. On the 3,662 APTOS images this gives
    2,562 / 550 / 550.
    """
    trainval, test = train_test_split(
        df, test_size=test_size, stratify=df["diagnosis"], random_state=seed)
    train, val = train_test_split(
        trainval, test_size=val_size / (1.0 - test_size),
        stratify=trainval["diagnosis"], random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


def split_of_ids(aptos_dir: PathLike) -> dict:
    """Map every APTOS ``id_code`` to ``"train"``, ``"val"`` or ``"test"``."""
    train, val, test = stratified_split(load_labels(Path(aptos_dir) / "train.csv"))
    mapping = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        mapping.update(dict.fromkeys(part["id_code"], name))
    return mapping
