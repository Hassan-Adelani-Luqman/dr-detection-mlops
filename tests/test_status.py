"""Up-time, monitoring and model-version sync."""
from app.services import model_service, monitor

STATUS_KEYS = {"status", "model_loaded", "active_version", "loaded_version",
               "retrain_count", "last_retrained", "hostname", "started_at",
               "uptime_seconds", "uptime", "requests_served", "server_errors",
               "predictions_served", "prediction_latency_ms"}


def test_health_reports_uptime(client):
    first = client.get("/health").json()
    assert first["status"] == "ok" and first["model_loaded"]
    assert first["uptime_seconds"] >= 0
    assert first["hostname"] == monitor.HOSTNAME and first["started_at"]
    assert client.get("/health").json()["uptime_seconds"] >= first["uptime_seconds"]


def test_status_has_all_fields(client):
    body = client.get("/api/status").json()
    assert STATUS_KEYS <= set(body)
    assert set(body["prediction_latency_ms"]) == {"window", "mean", "p50", "p95", "max"}


def test_responses_say_which_container_served_them(client):
    resp = client.get("/health")
    assert resp.headers["X-Served-By"] == monitor.HOSTNAME
    assert float(resp.headers["X-Process-Time-ms"]) >= 0


def test_prediction_is_counted_and_timed(client, synthetic_image):
    before = client.get("/api/status").json()["predictions_served"]
    with open(synthetic_image, "rb") as fh:
        resp = client.post("/api/predict", files={"file": ("s.png", fh, "image/png")})
    assert resp.status_code == 200
    after = client.get("/api/status").json()
    assert after["predictions_served"] == before + 1
    assert after["prediction_latency_ms"]["window"] >= 1


def test_predict_rejects_oversized_image(client):
    big = b"\x89PNG\r\n\x1a\n" + b"\0" * (10 * 1024 * 1024)
    resp = client.post("/api/predict", files={"file": ("big.png", big, "image/png")})
    assert resp.status_code == 400
    assert "limit" in resp.json()["detail"]


def test_model_reloads_when_active_version_changes(client, synthetic_image, monkeypatch):
    """Another replica retrained or rolled back -> this one reloads before predicting."""
    loads = []
    real_load = model_service.load_model
    monkeypatch.setattr(model_service, "load_model",
                        lambda path: loads.append(path) or real_load(path))
    monkeypatch.setattr(model_service, "get_active_version", lambda: "v_from_other_replica")

    result = model_service.predict(synthetic_image)
    assert len(loads) == 1 and result["active_version"] == "v_from_other_replica"
    model_service.predict(synthetic_image)            # same version: no reload
    assert len(loads) == 1


def test_format_duration():
    assert monitor.format_duration(5) == "5s"
    assert monitor.format_duration(65) == "1m 5s"
    assert monitor.format_duration(3725) == "1h 2m"
    assert monitor.format_duration(90000) == "1d 1h"


def test_every_page_shows_the_uptime_pill(client):
    for path in ("/", "/predict", "/upload", "/retrain", "/insights"):
        assert 'id="status-pill"' in client.get(path).text, path
    assert 'id="model-status"' in client.get("/insights").text
