"""Retraining service — fine-tune the classifier head on the training pool.

This module implements the three retraining sub-requirements:
  1. upload + save          -> handled in routes/upload.py (Phase 4)
  2. preprocess uploaded data -> build_dataset() below
  3. retrain from a pre-trained base -> build_retrain_model() + run_retraining()

Design decisions (see IMPLEMENTATION_PLAN.md Part A):
  * The current model is loaded as the pre-trained base and its convolutional
    backbone is FROZEN using a cutoff derived from the model itself, so only the
    classifier head trains. This makes CPU retraining feasible in minutes.
  * Training runs on the balanced replay buffer (seed images) TOGETHER with new
    user uploads, preventing catastrophic forgetting.
  * The original weights (exp5_base.keras) are NEVER overwritten. New versions
    are written via staging -> versioned snapshot -> current (active).
"""
import shutil
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

from app.config import (CURRENT_MODEL_PATH, EARLY_STOP_PATIENCE, FREEZE_CUTOFF,
                        MIN_SAMPLES, RETRAIN_BATCH_SIZE, RETRAIN_EPOCHS,
                        RETRAIN_LR, STAGING_MODEL_PATH, VERSIONS_DIR)
from app.database.models import UploadedImage
from app.services import model_service
from app.services.preprocess import preprocess_image
from app.services.registry import get_registry, update_registry


def freeze_backbone(model, cutoff: int = FREEZE_CUTOFF) -> int:
    """Freeze layers[:cutoff+1] (the conv backbone up to global pooling).

    Setting trainable=False on a BatchNormalization layer also forces it into
    inference mode in Keras 3, so the frozen BN running statistics stay fixed.
    Asserts that trainable weights remain — converting a bad cutoff from a
    silent no-op into a loud failure.
    """
    for layer in model.layers[:cutoff + 1]:
        layer.trainable = False
    assert model.trainable_weights, "freeze cutoff left no trainable weights"
    return cutoff


def build_retrain_model(base_path=CURRENT_MODEL_PATH):
    """Load the current model as a pre-trained base and freeze its backbone."""
    model = tf.keras.models.load_model(base_path)
    freeze_backbone(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(RETRAIN_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dataset(db_session, batch_size: int = RETRAIN_BATCH_SIZE):
    """Build train/val tf.data.Datasets from ALL pooled images (seed + user).

    This is the "preprocess uploaded data" step: every image is run through the
    same preprocess_image() used at inference time. Rows whose file is missing
    are skipped rather than crashing the run. Returns
    (train_ds, val_ds, n_train, n_val).
    """
    rows = db_session.query(UploadedImage).all()

    X, y = [], []
    skipped = 0
    for r in rows:
        try:
            X.append(preprocess_image(r.image_path))
            y.append(r.grade_label)
        except (ValueError, FileNotFoundError):
            skipped += 1
    n = len(X)
    if n < MIN_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_SAMPLES} usable images to retrain "
            f"(found {n}). Upload more images or seed the replay buffer."
        )

    X = np.stack(X)
    y = np.array(y, dtype=np.int32)

    # Stratified-ish shuffle + 80/20 split.
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    n_val = max(1, int(round(n * 0.2)))
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    return train_ds, val_ds, len(X_train), n_val, skipped


class _StatusCallback(tf.keras.callbacks.Callback):
    """Append per-epoch metrics to a shared status dict for live polling."""

    def __init__(self, status: dict, total_epochs: int):
        super().__init__()
        self.status = status
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.status["current_epoch"] = epoch + 1
        self.status["logs"].append({
            "epoch": epoch + 1,
            "loss": round(float(logs.get("loss", 0)), 4),
            "accuracy": round(float(logs.get("accuracy", 0)), 4),
            "val_loss": round(float(logs.get("val_loss", 0)), 4),
            "val_accuracy": round(float(logs.get("val_accuracy", 0)), 4),
        })


def run_retraining(db_session, epochs: int = RETRAIN_EPOCHS,
                   status: dict | None = None) -> dict:
    """Full retrain flow. Returns a summary dict; updates the registry.

    status (optional) is a shared dict updated live for the status endpoint.
    """
    status = status if status is not None else {"logs": [], "current_epoch": 0}
    status.setdefault("logs", [])

    # 1. Pre-trained base with frozen backbone.
    model = build_retrain_model(CURRENT_MODEL_PATH)

    # 2. Preprocess the pooled images into datasets.
    train_ds, val_ds, n_train, n_val, skipped = build_dataset(db_session)
    n_samples = n_train + n_val
    status["total_epochs"] = epochs

    # 3. Fine-tune the head.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=EARLY_STOP_PATIENCE,
                                         restore_best_weights=True),
        _StatusCallback(status, epochs),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, verbose=0)

    # 4. Version + register (NEVER overwrite base).
    registry = get_registry()
    now = datetime.now(timezone.utc)
    version = (f"v{registry['retrain_count'] + 2}_retrained_"
               f"{now.strftime('%Y%m%d_%H%M%S')}")
    versioned_path = VERSIONS_DIR / f"{version}.keras"

    model.save(STAGING_MODEL_PATH)                    # transient write target
    shutil.copy2(STAGING_MODEL_PATH, versioned_path)  # permanent snapshot
    shutil.copy2(versioned_path, CURRENT_MODEL_PATH)  # promote to active
    STAGING_MODEL_PATH.unlink(missing_ok=True)

    # 5. Update registry.
    hist = history.history
    entry = {
        "version": version,
        "type": "retrained",
        "samples_trained_on": n_samples,
        "created_at": now.isoformat(),
        "epochs_run": len(hist.get("loss", [])),
        "final_loss": round(float(hist["loss"][-1]), 4),
        "final_accuracy": round(float(hist["accuracy"][-1]), 4),
        "final_val_accuracy": round(float(hist["val_accuracy"][-1]), 4),
    }
    update_registry(new_entry=entry, active_version=version)

    # 6. Hot-swap the served model.
    model_service.reload_model()

    return {
        "version": version,
        "epochs_run": entry["epochs_run"],
        "samples": n_samples,
        "n_train": n_train,
        "n_val": n_val,
        "skipped": skipped,
        "final_loss": entry["final_loss"],
        "final_accuracy": entry["final_accuracy"],
        "final_val_accuracy": entry["final_val_accuracy"],
        "history": {k: [round(float(v), 4) for v in vals]
                    for k, vals in hist.items()},
    }
