"""Tests for the core ML package in src/.

Everything here runs on a fresh clone (synthetic images, the committed base
model) except the split test, which needs the APTOS train.csv in data/raw.
"""
import json

import cv2
import numpy as np
import pytest

from src.config import (CLASS_LABELS, DEFAULT_MODEL_PATH, FREEZE_CUTOFF,
                        REPO_ROOT, TARGET_SIZE)
from src.model import compute_metrics, freeze_backbone, load_model
from src.prediction import format_prediction, main, predict_image, predict_proba
from src.preprocessing import (load_labels, preprocess_batch, preprocess_image,
                               stratified_split)

APTOS_CSV = REPO_ROOT / "data" / "raw" / "train.csv"
RESULT_KEYS = {"predicted_grade", "grade_label", "clinical_note",
               "confidence", "confidence_scores"}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_preprocess_image_shape_dtype_range(synthetic_image):
    x = preprocess_image(synthetic_image)
    assert x.shape == (*TARGET_SIZE, 3)
    assert x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_preprocess_image_converts_bgr_to_rgb(tmp_path):
    bgr_blue = np.zeros((64, 64, 3), np.uint8)
    bgr_blue[..., 0] = 255                      # OpenCV stores blue in channel 0
    path = tmp_path / "blue.png"
    cv2.imwrite(str(path), bgr_blue)
    x = preprocess_image(path)
    assert x[..., 2].mean() == pytest.approx(1.0)   # blue is channel 2 in RGB
    assert x[..., 0].mean() == pytest.approx(0.0)


def test_preprocess_batch_adds_batch_dimension(synthetic_image):
    assert preprocess_batch(synthetic_image).shape == (1, *TARGET_SIZE, 3)


def test_preprocess_image_rejects_non_image(tmp_path):
    path = tmp_path / "notes.png"
    path.write_text("not an image")
    with pytest.raises(ValueError):
        preprocess_image(path)


@pytest.mark.skipif(not APTOS_CSV.exists(), reason="APTOS train.csv not in data/raw")
def test_stratified_split_reproduces_notebook_split():
    df = load_labels(APTOS_CSV)
    train, val, test = stratified_split(df)
    assert (len(train), len(val), len(test)) == (2562, 550, 550)
    ids = [set(part["id_code"]) for part in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    # Stratified: each grade's share of the test set is within 1 point of the whole.
    full = df["diagnosis"].value_counts(normalize=True)
    part = test["diagnosis"].value_counts(normalize=True)
    assert (full - part).abs().max() < 0.01


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def test_format_prediction_picks_highest_probability():
    result = format_prediction(np.array([0.1, 0.2, 0.5, 0.1, 0.1]))
    assert set(result) == RESULT_KEYS
    assert result["predicted_grade"] == 2
    assert result["grade_label"] == CLASS_LABELS[2]
    assert result["confidence"] == 0.5
    assert list(result["confidence_scores"]) == list(CLASS_LABELS.values())


def test_predict_image_returns_full_result(base_model, synthetic_image):
    result = predict_image(base_model, synthetic_image)
    assert set(result) == RESULT_KEYS
    assert 0 <= result["predicted_grade"] <= 4
    assert sum(result["confidence_scores"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_proba_from_paths_matches_array(base_model, synthetic_image):
    from_paths = predict_proba(base_model, [synthetic_image, synthetic_image], batch_size=1)
    from_array = predict_proba(base_model, np.stack([preprocess_image(synthetic_image)] * 2))
    assert from_paths.shape == (2, 5)
    np.testing.assert_allclose(from_paths, from_array, atol=1e-5)


def test_cli_prints_prediction(synthetic_image, capsys):
    assert main([str(synthetic_image), "--json"]) == 0
    result = json.loads(capsys.readouterr().out.strip())
    assert RESULT_KEYS <= set(result)
    assert result["image"] == str(synthetic_image)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_compute_metrics_perfect_predictions():
    y = np.repeat(np.arange(5), 4)
    m = compute_metrics(y, np.eye(5)[y])
    for key in ("accuracy", "qwk", "auc", "weighted_f1", "macro_f1", "grade34_recall"):
        assert m[key] == pytest.approx(1.0), key
    assert m["n"] == 20
    assert np.array_equal(np.diag(m["confusion_matrix"]), [4] * 5)


def test_grade34_recall_is_mean_of_class3_and_class4_recall():
    y = [3, 3, 3, 3, 4, 4, 0, 1, 2]
    pred = [3, 3, 3, 0, 4, 0, 0, 1, 2]          # class 3: 3/4, class 4: 1/2
    m = compute_metrics(y, np.eye(5)[pred])
    assert m["per_class_recall"][3] == pytest.approx(0.75)
    assert m["per_class_recall"][4] == pytest.approx(0.5)
    assert m["grade34_recall"] == pytest.approx(0.625)


def test_compute_metrics_missing_grades_give_none():
    y = [0, 0, 1, 1]
    m = compute_metrics(y, np.eye(5)[y])
    assert m["auc"] is None
    assert m["per_class_recall"][3] is None and m["grade34_recall"] is None
    json.dumps(m)                                 # stays JSON-serialisable


# --------------------------------------------------------------------------- #
# Fine-tuning setup
# --------------------------------------------------------------------------- #
def test_freeze_backbone_freezes_up_to_global_pooling():
    model = load_model(DEFAULT_MODEL_PATH)        # fresh copy: freezing mutates it
    freeze_backbone(model)
    assert "global_average_pooling" in model.layers[FREEZE_CUTOFF].name
    assert not any(layer.trainable for layer in model.layers[:FREEZE_CUTOFF + 1])
    assert any(layer.trainable_weights for layer in model.layers[FREEZE_CUTOFF + 1:])


def test_freeze_backbone_rejects_cutoff_that_leaves_nothing_to_train():
    model = load_model(DEFAULT_MODEL_PATH)
    with pytest.raises(ValueError):
        freeze_backbone(model, cutoff=len(model.layers))
