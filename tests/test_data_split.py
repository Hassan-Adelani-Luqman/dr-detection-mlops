"""The training pool and the held-out evaluation set must never overlap."""
import csv
from collections import Counter

import pytest

from app.config import DATA_DIR, UPLOADS_DIR
from app.database.db import SessionLocal
from app.database.models import UploadedImage
from app.services.pool import eval_rows, resolve_image_path, training_rows
from scripts.prepare_data import pick_seed_ids
from src.config import REPO_ROOT
from src.preprocessing import load_labels, stratified_split

MANIFEST = REPO_ROOT / "data" / "test" / "manifest.csv"
APTOS_CSV = REPO_ROOT / "data" / "raw" / "train.csv"
needs_aptos = pytest.mark.skipif(not APTOS_CSV.exists(), reason="APTOS train.csv not in data/raw")


def _manifest() -> list[tuple[str, int]]:
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return [(r["id_code"], int(r["diagnosis"])) for r in csv.DictReader(fh)]


def test_manifest_lists_the_550_held_out_images():
    rows = _manifest()
    assert len(rows) == len({i for i, _ in rows}) == 550
    assert Counter(g for _, g in rows) == {0: 271, 1: 56, 2: 150, 3: 29, 4: 44}


@needs_aptos
def test_manifest_matches_the_notebook_test_split():
    _, _, test = stratified_split(load_labels(APTOS_CSV))
    assert {i for i, _ in _manifest()} == set(test["id_code"])


@needs_aptos
def test_seed_ids_come_only_from_the_training_split():
    train, val, test = stratified_split(load_labels(APTOS_CSV))
    picks = pick_seed_ids(train, per_grade=40)
    ids = {i for grade_ids in picks.values() for i in grade_ids}
    assert all(len(v) == 40 for v in picks.values()) and len(ids) == 200
    assert ids <= set(train["id_code"])
    assert ids.isdisjoint(test["id_code"]) and ids.isdisjoint(val["id_code"])


def test_training_rows_never_include_eval(client):
    db = SessionLocal()
    added = [UploadedImage(filename=f"t_{s}.png", image_path=f"t_{s}.png",
                           grade_label=0, source=s) for s in ("seed", "user", "eval")]
    db.add_all(added)
    db.commit()
    try:
        assert {r.source for r in training_rows(db)} >= {"seed", "user"}
        assert all(r.source != "eval" for r in training_rows(db))
        assert eval_rows(db) and all(r.source == "eval" for r in eval_rows(db))
    finally:
        for row in added:
            db.delete(row)
        db.commit()
        db.close()


def test_resolve_image_path_rules():
    assert resolve_image_path("train/abc.png") == DATA_DIR / "train" / "abc.png"
    assert resolve_image_path("test/abc.png") == DATA_DIR / "test" / "abc.png"
    assert resolve_image_path("abc.png") == UPLOADS_DIR / "abc.png"
    # Rows written on another machine fall back to the file name in uploads.
    assert resolve_image_path("C:\\Users\\me\\uploads\\seed_1.png") == UPLOADS_DIR / "seed_1.png"
    assert resolve_image_path("/no/such/dir/x.png") == UPLOADS_DIR / "x.png"
