"""Retraining service: fine-tune, evaluate on held-out data, promote or keep.

The pipeline covers the three retraining requirements and then evaluates the
result the way a production system should:

  1. upload + save to the database   -> app/routes/upload.py
  2. preprocess the pooled images    -> load_images() (src.preprocessing)
  3. retrain from a pre-trained base -> src.model.build_finetune_model() + fine_tune()
  4. evaluate in production          -> score the candidate AND the model being served
                                        on the held-out evaluation set (source="eval")
  5. promote or keep                 -> app.services.promotion.should_promote()

Design decisions:
  * The current model is the pre-trained base; its convolutional backbone is
    FROZEN, so only the classifier head trains (minutes on CPU).
  * Training uses the balanced replay buffer (seed) TOGETHER with user uploads,
    preventing catastrophic forgetting. Evaluation images are never trained on.
  * Every candidate is saved as a version, promoted or not; a rejected one can
    still be activated by hand from Insights. exp5_base.keras is NEVER written.
"""
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from app.config import (CLASS_LABELS, CURRENT_MODEL_PATH, EARLY_STOP_PATIENCE,
                        MIN_SAMPLES, RETRAIN_BATCH_SIZE, RETRAIN_EPOCHS, RETRAIN_LR,
                        STAGING_MODEL_PATH, TARGET_SIZE, VERSIONS_DIR)
from app.services import model_service
from app.services.pool import eval_rows, resolve_image_path, training_rows
from app.services.promotion import should_promote
from app.services.registry import (get_active_version, get_registry,
                                   install_current_model, register_version)
from src.model import build_finetune_model, compute_metrics, fine_tune, load_model
from src.prediction import predict_proba
from src.preprocessing import preprocess_image

HEADLINE_METRICS = ("accuracy", "qwk", "auc", "weighted_f1", "grade34_recall")
EVAL_SOURCE_HELDOUT = "held-out evaluation set"
EVAL_SOURCE_VALIDATION = "validation split (no held-out set available)"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _load_one(row):
    try:
        x = preprocess_image(resolve_image_path(row.image_path))
    except (ValueError, FileNotFoundError):
        return None
    # preprocess_image gives k/255 floats, so uint8 storage is exact and 4x smaller.
    return np.rint(x * 255).astype(np.uint8), row.grade_label


def load_images(rows, workers: int | None = None):
    """Preprocess pooled images in parallel -> (X uint8 (N,224,224,3), y, n_skipped).

    Uses the same preprocess_image() as inference. Rows whose file is missing or
    unreadable are skipped rather than crashing the run.
    """
    workers = workers or min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = [r for r in pool.map(_load_one, rows) if r is not None]
    if not loaded:
        return np.empty((0, *TARGET_SIZE, 3), np.uint8), np.empty(0, np.int32), len(rows)
    X = np.stack([x for x, _ in loaded])
    y = np.array([g for _, g in loaded], dtype=np.int32)
    return X, y, len(rows) - len(loaded)


def split_train_val(y: np.ndarray, val_fraction: float = 0.2, seed: int = 42):
    """Indices for a stratified train/val split; random if a grade is too rare.

    Stratifying needs at least one validation (and one training) image per
    grade, so on a small pool the validation share is raised to one per grade.
    """
    idx = np.arange(len(y))
    n_classes = len(np.unique(y))
    n_val = max(1, int(round(len(y) * val_fraction)))
    counts = np.bincount(y)[np.unique(y)]           # images per grade present
    if counts.min() >= 2 and len(y) - max(n_val, n_classes) >= n_classes:
        n_val = max(n_val, n_classes)
        train_idx, val_idx = train_test_split(idx, test_size=n_val, stratify=y,
                                              random_state=seed)
        return train_idx, val_idx, "stratified"
    perm = np.random.default_rng(seed).permutation(idx)
    return perm[n_val:], perm[:n_val], "random"


@dataclass
class TrainingData:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    X_val: np.ndarray              # uint8, kept to evaluate on if there is no eval set
    y_val: np.ndarray
    n_train: int
    n_val: int
    skipped: int
    split: str
    counts_by_grade: dict


def build_dataset(db_session, batch_size: int = RETRAIN_BATCH_SIZE) -> TrainingData:
    """The "preprocess uploaded data" step: training pool -> tf.data datasets."""
    X, y, skipped = load_images(training_rows(db_session))
    if len(y) < MIN_SAMPLES:
        raise ValueError(f"Need at least {MIN_SAMPLES} usable images to retrain "
                         f"(found {len(y)}). Upload more images or seed the replay buffer.")
    train_idx, val_idx, split = split_train_val(y)

    def as_ds(ix, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((X[ix], y[ix]))
        if shuffle:
            ds = ds.shuffle(len(ix), seed=42)
        return (ds.batch(batch_size)
                  .map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b))
                  .prefetch(tf.data.AUTOTUNE))

    counts = {CLASS_LABELS[g]: int((y == g).sum()) for g in CLASS_LABELS}
    return TrainingData(as_ds(train_idx, True), as_ds(val_idx, False), X[val_idx], y[val_idx],
                        len(train_idx), len(val_idx), skipped, split, counts)


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #
def _stage(status: dict, stage: str, text: str) -> None:
    status["stage"] = stage
    status["stage_text"] = text


class _StatusCallback(tf.keras.callbacks.Callback):
    """Append per-epoch metrics to a shared status dict for live polling."""

    def __init__(self, status: dict, total_epochs: int):
        super().__init__()
        self.status = status
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.status["current_epoch"] = epoch + 1
        _stage(self.status, "training", f"Training: epoch {epoch + 1} of {self.total_epochs}")
        self.status["logs"].append({
            "epoch": epoch + 1,
            "loss": round(float(logs.get("loss", 0)), 4),
            "accuracy": round(float(logs.get("accuracy", 0)), 4),
            "val_loss": round(float(logs.get("val_loss", 0)), 4),
            "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4),
        })


def _headline(metrics: dict) -> dict:
    return {k: (round(metrics[k], 4) if metrics[k] is not None else None)
            for k in HEADLINE_METRICS} | {"n": metrics["n"]}


# --------------------------------------------------------------------------- #
# The full retrain
# --------------------------------------------------------------------------- #
def run_retraining(db_session, epochs: int = RETRAIN_EPOCHS,
                   status: dict | None = None) -> dict:
    """Fine-tune, evaluate both models on held-out data, then promote or keep.

    ``status`` (optional) is a shared dict updated live for the status endpoint.
    Returns a summary dict; the registry records the new version either way.
    """
    status = status if status is not None else {}
    status.setdefault("logs", [])
    status["total_epochs"] = epochs
    previous_version = get_active_version()

    # 1. Pre-trained base (the model being served) with its backbone frozen.
    _stage(status, "loading", f"Loading {previous_version} as the pre-trained base")
    model = build_finetune_model(CURRENT_MODEL_PATH, learning_rate=RETRAIN_LR)

    # 2. Preprocess the training pool and the held-out evaluation set.
    pool_rows, held_rows = training_rows(db_session), eval_rows(db_session)
    _stage(status, "preprocessing",
           f"Preprocessing {len(pool_rows)} training and {len(held_rows)} evaluation images")
    data = build_dataset(db_session)
    X_eval, y_eval, eval_skipped = load_images(held_rows)

    # 3. Fine-tune the head (early stopping restores the best epoch).
    _stage(status, "training", f"Training: epoch 1 of {epochs}")
    history = fine_tune(model, data.train_ds, data.val_ds, epochs=epochs,
                        early_stop_patience=EARLY_STOP_PATIENCE,
                        callbacks=[_StatusCallback(status, epochs)])

    # 4. Evaluate the candidate and the served model on the same held-out images.
    if len(y_eval):
        eval_source, X_cmp, y_cmp = EVAL_SOURCE_HELDOUT, X_eval, y_eval
    else:
        eval_source, X_cmp, y_cmp = EVAL_SOURCE_VALIDATION, data.X_val, data.y_val
    _stage(status, "evaluating",
           f"Evaluating the new model and {previous_version} on {len(y_cmp)} "
           f"{'held-out' if len(y_eval) else 'validation'} images")
    candidate_metrics = compute_metrics(y_cmp, predict_proba(model, X_cmp))
    served = load_model(CURRENT_MODEL_PATH)
    previous_metrics = compute_metrics(y_cmp, predict_proba(served, X_cmp))
    del served

    # 5. Decide, then save the candidate as a version either way.
    promote, reason = should_promote(candidate_metrics, previous_metrics)
    _stage(status, "saving", "Promoting the new version" if promote
           else "Saving the new version (not promoted)")
    registry = get_registry()
    now = datetime.now(timezone.utc)
    version = f"v{registry['retrain_count'] + 2}_retrained_{now.strftime('%Y%m%d_%H%M%S')}"
    versioned_path = VERSIONS_DIR / f"{version}.keras"
    model.save(STAGING_MODEL_PATH)                       # transient write target
    shutil.copyfile(STAGING_MODEL_PATH, versioned_path)  # permanent snapshot
    STAGING_MODEL_PATH.unlink(missing_ok=True)

    hist = history.history
    entry = {
        "version": version,
        "type": "retrained",
        "base_version": previous_version,
        "created_at": now.isoformat(),
        "samples_trained_on": data.n_train + data.n_val,
        "train_counts_by_grade": data.counts_by_grade,
        "split": data.split,
        "epochs_run": len(hist.get("loss", [])),
        "final_loss": round(float(hist["loss"][-1]), 4),
        "train_accuracy": round(float(hist["accuracy"][-1]), 4),
        "final_val_accuracy": round(float(hist["val_accuracy"][-1]), 4),
        "eval_source": eval_source,
        "eval_n": int(len(y_cmp)),
        "eval_candidate": _headline(candidate_metrics),
        "eval_previous": _headline(previous_metrics),
        "promoted": promote,
        "decision_reason": reason,
    }
    if promote:
        install_current_model(versioned_path)            # atomic swap of the served file
    register_version(entry, promote=promote)
    if promote:
        model_service.reload_model()                     # hot-swap in this process
    _stage(status, "done", "Promoted" if promote else "Kept the current model")

    return {
        "version": version,
        "previous_version": previous_version,
        "active_version": version if promote else previous_version,
        "promoted": promote,
        "decision_reason": reason,
        "eval": {"source": eval_source, "n": entry["eval_n"],
                 "skipped": eval_skipped if len(y_eval) else 0,
                 "candidate": entry["eval_candidate"], "previous": entry["eval_previous"]},
        "epochs_run": entry["epochs_run"],
        "samples": entry["samples_trained_on"],
        "n_train": data.n_train,
        "n_val": data.n_val,
        "skipped": data.skipped,
        "split": data.split,
        "train_counts_by_grade": data.counts_by_grade,
        "final_loss": entry["final_loss"],
        "train_accuracy": entry["train_accuracy"],
        "final_val_accuracy": entry["final_val_accuracy"],
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in hist.items()},
    }
