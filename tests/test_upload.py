"""Upload endpoint tests."""
from conftest import requires_samples

from app.database.db import SessionLocal
from app.database.models import UploadedImage


@requires_samples
def test_upload_creates_db_row(client, grade0_image):
    before = SessionLocal().query(UploadedImage).filter_by(source="user").count()

    with open(grade0_image, "rb") as fh:
        resp = client.post("/api/upload", data={"grade_label": "0"},
                           files={"file": ("g0.png", fh, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["grade_label"] == 0
    assert data["total_user_uploads"] == before + 1

    # a row was actually persisted with source="user"
    db = SessionLocal()
    row = db.query(UploadedImage).order_by(UploadedImage.id.desc()).first()
    assert row.grade_label == 0
    assert row.source == "user"
    db.close()


@requires_samples
def test_upload_rejects_invalid_grade(client, grade0_image):
    with open(grade0_image, "rb") as fh:
        resp = client.post("/api/upload", data={"grade_label": "9"},
                           files={"file": ("g0.png", fh, "image/png")})
    assert resp.status_code == 400


def test_upload_rejects_non_image(client):
    resp = client.post("/api/upload", data={"grade_label": "0"},
                       files={"file": ("bad.txt", b"nope", "text/plain")})
    assert resp.status_code == 400
