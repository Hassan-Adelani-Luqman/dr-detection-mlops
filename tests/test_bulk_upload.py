"""Bulk and ZIP uploads into the training pool or the evaluation set."""
import io
import zipfile

import cv2
import numpy as np
import pytest

from app.config import UPLOADS_DIR
from app.database.db import SessionLocal
from app.database.models import UploadedImage


def _png(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    ok, buf = cv2.imencode(".png", rng.integers(0, 255, (32, 40, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _zip(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _count(source: str) -> int:
    db = SessionLocal()
    try:
        return db.query(UploadedImage).filter_by(source=source).count()
    finally:
        db.close()


def _bulk(client, files, grade="2", destination="train"):
    return client.post("/api/upload/bulk",
                       data={"grade_label": grade, "destination": destination},
                       files=[("files", f) for f in files])


def test_bulk_upload_saves_every_image(client):
    before = _count("user")
    resp = _bulk(client, [(f"img{i}.png", _png(i), "image/png") for i in range(5)])
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["saved"]) == 5 and body["rejected"] == []
    assert all(s["grade_label"] == 2 for s in body["saved"])
    assert _count("user") == before + 5
    assert body["pools"]["user"]["Moderate DR"] >= 5


def test_bulk_upload_reports_bad_files_and_keeps_good_ones(client):
    files = [("good.png", _png(1), "image/png"),
             ("notes.png", b"definitely not an image", "image/png"),
             ("huge.png", b"\x89PNG\r\n\x1a\n" + b"\0" * (10 * 1024 * 1024), "image/png")]
    body = _bulk(client, files).json()
    assert [s["filename"] for s in body["saved"]] == ["good.png"]
    reasons = {r["filename"]: r["reason"] for r in body["rejected"]}
    assert "not a JPEG" in reasons["notes.png"] and "larger than" in reasons["huge.png"]


def test_bulk_upload_to_eval_set(client):
    before_eval, before_user = _count("eval"), _count("user")
    body = _bulk(client, [("e.png", _png(7), "image/png")], destination="eval").json()
    assert body["destination"] == "eval" and len(body["saved"]) == 1
    assert _count("eval") == before_eval + 1 and _count("user") == before_user


@pytest.mark.parametrize("data,expected", [
    ({"grade_label": "7", "destination": "train"}, "grade_label"),
    ({"grade_label": "1", "destination": "somewhere"}, "destination"),
])
def test_bulk_upload_rejects_bad_form_values(client, data, expected):
    resp = client.post("/api/upload/bulk", data=data,
                       files=[("files", ("a.png", _png(), "image/png"))])
    assert resp.status_code == 400 and expected in resp.json()["detail"]


def test_bulk_upload_limits_file_count(client):
    files = [(f"{i}.png", b"x", "image/png") for i in range(101)]
    resp = _bulk(client, files)
    assert resp.status_code == 400 and "At most 100" in resp.json()["detail"]


def test_zip_upload_takes_grades_from_folders(client):
    archive = _zip({
        "0/a.png": _png(1), "0/b.png": _png(2),
        "grade_3/c.png": _png(3),
        "batch/4/d.png": _png(4),               # grade folder deeper in the path
        "e.png": _png(5),                       # not in a grade folder
        "2/readme.txt": b"hello",               # not an image
        "__MACOSX/0/._a.png": b"meta", "0/.DS_Store": b"meta",
    })
    resp = client.post("/api/upload/zip", data={"destination": "train"},
                       files={"file": ("pack.zip", archive, "application/zip")})
    assert resp.status_code == 200
    body = resp.json()
    saved = sorted((s["filename"], s["grade_label"]) for s in body["saved"])
    assert saved == [("a.png", 0), ("b.png", 0), ("c.png", 3), ("d.png", 4)]
    rejected = {r["filename"] for r in body["rejected"]}
    assert rejected == {"e.png", "2/readme.txt"}
    assert body["ignored"] == 2


def test_zip_entry_cannot_escape_the_uploads_folder(client):
    outside = UPLOADS_DIR.parent / "evil.png"
    archive = _zip({"0/../../evil.png": _png(9)})
    body = client.post("/api/upload/zip", files={"file": ("x.zip", archive, "application/zip")}).json()
    assert [s["filename"] for s in body["saved"]] == ["evil.png"]
    assert not outside.exists()
    assert any(p.name.endswith("_evil.png") for p in UPLOADS_DIR.iterdir())


def test_zip_upload_rejects_non_zip(client):
    resp = client.post("/api/upload/zip", files={"file": ("x.zip", b"not a zip", "application/zip")})
    assert resp.status_code == 400 and "ZIP" in resp.json()["detail"]


def test_single_upload_accepts_destination(client):
    before = _count("eval")
    resp = client.post("/api/upload", data={"grade_label": "1", "destination": "eval"},
                       files={"file": ("one.png", _png(11), "image/png")})
    assert resp.status_code == 200 and resp.json()["destination"] == "eval"
    assert _count("eval") == before + 1


def test_upload_page_shows_all_three_pools(client):
    html = client.get("/upload").text
    for marker in ('id="img-input"', "multiple", 'id="zip-input"', 'value="eval"',
                   'id="c-seed-0"', 'id="c-user-0"', 'id="c-eval-0"'):
        assert marker in html, marker
