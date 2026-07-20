"""Prediction endpoint tests."""
from conftest import requires_samples


@requires_samples
def test_predict_returns_valid_grade(client, grade0_image):
    with open(grade0_image, "rb") as fh:
        resp = client.post("/api/predict",
                           files={"file": ("g0.png", fh, "image/png")})
    assert resp.status_code == 200
    data = resp.json()

    # grade is in the valid range
    assert 0 <= data["predicted_grade"] <= 4
    assert data["grade_label"]
    assert data["clinical_note"]

    # all five class probabilities present and summing to ~1.0
    scores = data["confidence_scores"]
    assert len(scores) == 5
    assert abs(sum(scores.values()) - 1.0) < 1e-3

    # active version is reported
    assert data["active_version"]


@requires_samples
def test_predict_grade0_image_is_correct(client, grade0_image):
    """The known grade-0 sample must predict grade 0 (prediction criterion)."""
    with open(grade0_image, "rb") as fh:
        resp = client.post("/api/predict",
                           files={"file": ("g0.png", fh, "image/png")})
    assert resp.status_code == 200
    assert resp.json()["predicted_grade"] == 0


def test_predict_rejects_non_image(client):
    resp = client.post("/api/predict",
                       files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400


def test_predict_rejects_fake_image_bytes(client):
    resp = client.post("/api/predict",
                       files={"file": ("fake.png", b"XXXX not png", "image/png")})
    assert resp.status_code == 400
