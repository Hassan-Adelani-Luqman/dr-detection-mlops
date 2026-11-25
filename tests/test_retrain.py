"""Retraining: data split, readiness trigger, and end-to-end promote / keep.

The end-to-end tests run a real (1-epoch) fine-tune on a small synthetic pool.
Model state lives in the temp MODELS_RUNTIME_DIR set by conftest, so nothing
under models/ is written. Deselect with: pytest -m "not slow".
"""
import hashlib
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import partial

import cv2
import numpy as np
import pytest

from app.config import BASE_MODEL_PATH, CURRENT_MODEL_PATH, UPLOADS_DIR, VERSIONS_DIR
from app.database.db import SessionLocal
from app.database.models import UploadedImage
from app.routes import retrain as retrain_routes
from app.services import registry, retrain_service


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_images(db, source: str, per_grade: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for grade in range(5):
        for _ in range(per_grade):
            img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            img[..., 1] = 40 * grade                     # a little grade signal
            name = f"syn_{source}_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(UPLOADS_DIR / name), img)
            db.add(UploadedImage(filename=name, image_path=name, grade_label=grade, source=source))
    db.commit()


@pytest.fixture(scope="module")
def synthetic_pool(client):
    db = SessionLocal()
    _add_images(db, "seed", per_grade=3, seed=1)
    _add_images(db, "eval", per_grade=2, seed=2)
    db.close()


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_split_is_stratified_when_every_grade_has_two_images():
    y = np.repeat(np.arange(5), 4)
    train, val, kind = retrain_service.split_train_val(y)
    # 20% of 20 is 4 < 5 grades, so validation grows to one image per grade.
    assert kind == "stratified" and len(val) == 5 and len(train) == 15
    assert sorted(set(y[val])) == [0, 1, 2, 3, 4]


def test_split_keeps_twenty_percent_on_larger_pools():
    y = np.repeat(np.arange(5), 40)                       # like the 200-image replay buffer
    train, val, kind = retrain_service.split_train_val(y)
    assert kind == "stratified" and len(val) == 40
    assert np.bincount(y[val]).tolist() == [8] * 5


def test_split_falls_back_to_random_for_rare_grades():
    y = np.array([0, 0, 0, 0, 1])                         # grade 1 has a single image
    train, val, kind = retrain_service.split_train_val(y)
    assert kind == "random" and len(train) + len(val) == 5


def test_readiness_counts_uploads_since_last_retrain(client, monkeypatch):
    db = SessionLocal()
    try:
        _add_images(db, "user", per_grade=1, seed=3)       # 5 new user uploads
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        monkeypatch.setattr(retrain_routes, "get_registry", lambda: {"last_retrained": future})
        assert retrain_routes.readiness(db)["new_since_last_retrain"] == 0
        monkeypatch.setattr(retrain_routes, "get_registry", lambda: {"last_retrained": past})
        ready = retrain_routes.readiness(db)
        assert ready["new_since_last_retrain"] >= 5
        assert ready["recommended"] == (ready["new_since_last_retrain"] >= ready["threshold"])
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def _retrain(monkeypatch, decision: bool):
    monkeypatch.setattr(retrain_service, "should_promote",
                        lambda cand, cur: (decision, "forced by test"))
    db = SessionLocal()
    try:
        status = {}
        return retrain_service.run_retraining(db, epochs=1, status=status), status
    finally:
        db.close()


@pytest.mark.slow
def test_retrain_promotes_and_hot_swaps(synthetic_pool, monkeypatch):
    base_before = _sha(BASE_MODEL_PATH)
    before = registry.get_active_version()
    result, status = _retrain(monkeypatch, decision=True)

    assert result["promoted"] and result["active_version"] == result["version"]
    assert registry.get_active_version() == result["version"]
    assert (VERSIONS_DIR / f"{result['version']}.keras").exists()
    assert _sha(CURRENT_MODEL_PATH) == _sha(VERSIONS_DIR / f"{result['version']}.keras")
    assert status["stage"] == "done"

    ev = result["eval"]
    assert ev["source"] == retrain_service.EVAL_SOURCE_HELDOUT and ev["n"] >= 10
    for key in ("qwk", "accuracy", "grade34_recall"):
        assert key in ev["candidate"] and key in ev["previous"]
    entry = registry.find_version(result["version"])
    assert entry["promoted"] is True and entry["base_version"] == before
    assert entry["eval_candidate"] == ev["candidate"]
    # Evaluation images are never trained on.
    assert sum(entry["train_counts_by_grade"].values()) == result["samples"]
    db = SessionLocal()
    try:
        n_training = len(retrain_service.training_rows(db))
    finally:
        db.close()
    assert result["samples"] <= n_training
    assert _sha(BASE_MODEL_PATH) == base_before


@pytest.mark.slow
def test_rejected_candidate_is_saved_but_not_served(synthetic_pool, monkeypatch):
    serving = registry.get_active_version()
    current_sha = _sha(CURRENT_MODEL_PATH)
    result, _ = _retrain(monkeypatch, decision=False)

    assert not result["promoted"] and result["active_version"] == serving
    assert registry.get_active_version() == serving
    assert _sha(CURRENT_MODEL_PATH) == current_sha
    assert (VERSIONS_DIR / f"{result['version']}.keras").exists()
    entry = registry.find_version(result["version"])
    assert entry["promoted"] is False and entry["decision_reason"] == "forced by test"


@pytest.mark.slow
def test_retrain_through_the_api(client, synthetic_pool, monkeypatch):
    monkeypatch.setattr(retrain_service, "run_retraining",
                        partial(retrain_service.run_retraining, epochs=1))
    assert client.post("/api/retrain").status_code == 200
    for _ in range(240):
        status = client.get("/api/retrain/status").json()
        if status["status"] != "running":
            break
        time.sleep(0.5)
    assert status["status"] == "complete", status.get("error")
    assert status["result"]["eval"]["n"] >= 10
    assert client.get("/api/retrain/readiness").json()["running"] is False
    history = client.get("/api/insights").json()["history"]
    assert history[-1]["heldout"] is not None and "promoted" in history[-1]


def test_retraining_can_be_switched_off(client, monkeypatch):
    monkeypatch.setattr(retrain_routes, "RETRAIN_ENABLED", False)
    resp = client.post("/api/retrain")
    assert resp.status_code == 503 and "turned off" in resp.json()["detail"]
    ready = client.get("/api/retrain/readiness").json()
    assert ready["retrain_enabled"] is False and "RETRAIN_ENABLED" in ready["disabled_message"]
    assert retrain_routes.start_retraining() is False
