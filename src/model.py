"""The Keras model: loading, fine-tuning from the pre-trained base, evaluation.

Fine-tuning freezes the convolutional backbone (layers up to and including the
global average pooling layer) so only the dense classifier head trains. That
keeps retraining feasible on CPU and limits catastrophic forgetting.
"""
import math
import warnings
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, roc_auc_score)

from src.config import DEFAULT_MODEL_PATH, FREEZE_CUTOFF, NUM_CLASSES
from src.prediction import predict_proba

PathLike = Union[str, Path]


def load_model(path: PathLike = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """Load a saved ``.keras`` model."""
    return tf.keras.models.load_model(path)


def freeze_backbone(model: tf.keras.Model, cutoff: int = FREEZE_CUTOFF) -> int:
    """Freeze ``model.layers[:cutoff + 1]`` and return the cutoff.

    In Keras 3, setting ``trainable=False`` on a BatchNormalization layer also
    runs it in inference mode, so the frozen running statistics stay fixed.
    Fails loudly if the cutoff would leave nothing to train.
    """
    for layer in model.layers[:cutoff + 1]:
        layer.trainable = False
    if not model.trainable_weights:
        raise ValueError(f"Freeze cutoff {cutoff} left no trainable weights.")
    return cutoff


def build_finetune_model(base_path: PathLike = DEFAULT_MODEL_PATH,
                         learning_rate: float = 1e-5,
                         cutoff: int = FREEZE_CUTOFF) -> tf.keras.Model:
    """Load a pre-trained model, freeze its backbone, and compile it for fine-tuning."""
    model = load_model(base_path)
    freeze_backbone(model, cutoff)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune(model: tf.keras.Model, train_ds, val_ds, epochs: int,
              early_stop_patience: int = 3, callbacks: Sequence = ()):
    """Train the unfrozen head with early stopping on validation loss.

    The best epoch's weights are restored. Returns the Keras ``History``.
    """
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stop_patience, restore_best_weights=True)
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=[stop, *callbacks], verbose=0)


def _clean(value):
    """Plain float for JSON, with NaN turned into None."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def compute_metrics(y_true, probs) -> dict:
    """Classification metrics for 5-grade predictions.

    ``probs`` has shape (N, 5). ``grade34_recall`` is the mean of the class-3
    and class-4 recalls, the same definition the training project uses. AUC is
    one-vs-rest macro AUC, or None when a grade is missing from ``y_true``.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    y_pred = probs.argmax(axis=1)
    labels = list(range(NUM_CLASSES))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    per_class_recall = [cm[i, i] / support[i] if support[i] else None for i in labels]
    severe = [r for r in per_class_recall[3:5] if r is not None]

    # A grade missing from y_true makes AUC undefined: sklearn returns NaN with
    # a warning (or raises, depending on version). Either way report None.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        try:
            auc = roc_auc_score(y_true, probs, multi_class="ovr", labels=labels)
        except ValueError:
            auc = None

    return {
        "n": int(len(y_true)),
        "accuracy": _clean(accuracy_score(y_true, y_pred)),
        "qwk": _clean(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
        "auc": _clean(auc),
        "weighted_f1": _clean(f1_score(y_true, y_pred, labels=labels,
                                       average="weighted", zero_division=0)),
        "macro_f1": _clean(f1_score(y_true, y_pred, labels=labels,
                                    average="macro", zero_division=0)),
        "grade34_recall": _clean(np.mean(severe)) if severe else None,
        "per_class_recall": [_clean(r) for r in per_class_recall],
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(model: tf.keras.Model, images, y_true, batch_size: int = 32) -> dict:
    """Predict ``images`` (paths or a preprocessed array) and score against ``y_true``."""
    return compute_metrics(y_true, predict_proba(model, images, batch_size=batch_size))
