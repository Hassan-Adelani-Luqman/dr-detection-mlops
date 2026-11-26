"""The Insights page's dataset features (the brief's "interpret 3 features")."""
import json

from app.config import DATASET_INSIGHTS_PATH

CHARTED = ("megapixels", "brightness", "contrast", "green")


def test_dataset_insights_file_has_per_grade_features():
    data = json.loads(DATASET_INSIGHTS_PATH.read_text(encoding="utf-8"))
    assert sum(data["counts"]) == 3662 and len(data["counts"]) == 5
    assert len(data["square_share"]) == 5
    assert len(data["features"]) >= 3
    for key in CHARTED:
        feature = data["features"][key]
        assert len(feature["by_grade"]) == 5
        assert -1 <= feature["spearman"] <= 1
        for b in feature["by_grade"]:
            assert b["q1"] <= b["median"] <= b["q3"]


def test_api_insights_serves_dataset_features(client):
    body = client.get("/api/insights").json()
    features = body["dataset_features"]["features"]
    assert set(CHARTED) <= set(features)


def test_insights_page_has_feature_cards_and_interpretations(client):
    html = client.get("/insights").text
    for canvas in ("dist-chart", "res-chart", "square-chart", "bright-chart",
                   "contrast-chart", "green-chart"):
        assert f'id="{canvas}"' in html, canvas
    assert html.count("What it tells us.") == 4
    assert "%%" not in html                     # no unfilled placeholders
