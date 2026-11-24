"""Central configuration for the web app: paths, UI metadata, retraining settings.

Model and dataset constants (class labels, input size, freeze cutoff) live in
``src/config.py`` and are re-exported here so app code has one import point.

Values that the container may need to override (database location, uploads
directory) are read from the environment with sensible local defaults, so the
same code runs unchanged locally and in Docker.
"""
import json
import os
from pathlib import Path

from src.config import (CLASS_LABELS, CLINICAL_NOTES, FREEZE_CUTOFF,  # noqa: F401
                        NUM_CLASSES, TARGET_SIZE)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent          # repo root
APP_DIR = BASE_DIR / "app"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
# Writable model state (current model, versions, registry). Defaults to
# models/; tests point it at a temp folder so they never touch real state.
MODELS_RUNTIME_DIR = Path(os.getenv("MODELS_RUNTIME_DIR", str(MODELS_DIR)))
VERSIONS_DIR = MODELS_RUNTIME_DIR / "versions"

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(DATA_DIR / "uploads")))
TRAIN_DIR = DATA_DIR / "train"     # replay buffer (seed images from the training split)
TEST_DIR = DATA_DIR / "test"       # held-out evaluation set (test split)

BASE_MODEL_PATH = MODELS_DIR / "exp5_base.keras"           # ORIGINAL — never written
CURRENT_MODEL_PATH = MODELS_RUNTIME_DIR / "exp5_current.keras"     # active served model
STAGING_MODEL_PATH = MODELS_RUNTIME_DIR / "exp5_staging.keras"     # transient write target
REGISTRY_PATH = MODELS_RUNTIME_DIR / "model_registry.json"

# Ensure runtime directories exist (safe if already present)
for _d in (MODELS_DIR, MODELS_RUNTIME_DIR, DATA_DIR, VERSIONS_DIR, UPLOADS_DIR,
           TRAIN_DIR, TEST_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}")

# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
# UI colour per grade (green -> red severity ramp)
GRADE_COLORS = {
    0: "#2e7d32",   # green
    1: "#f9a825",   # yellow
    2: "#ef6c00",   # orange
    3: "#c62828",   # red
    4: "#7b1fa2",   # purple (proliferative)
}

# --------------------------------------------------------------------------- #
# Retraining (low LR + frozen backbone + replay buffer to avoid catastrophic
# forgetting; the freeze cutoff itself is in src/config.py)
# --------------------------------------------------------------------------- #
# Minimum images required to retrain. Kept low (5) so the public demo can
# retrain on a handful of images uploaded live, since the APTOS replay buffer
# cannot be redistributed to the hosted environment. Locally, the seeded buffer
# (~200 images) means retraining trains on far more than this floor. The low LR
# and frozen backbone keep even a tiny retrain from collapsing the model.
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "5"))
RETRAIN_EPOCHS = 8
RETRAIN_BATCH_SIZE = 8
RETRAIN_LR = 1e-5                 # low LR: fine-tune the head without wrecking it
EARLY_STOP_PATIENCE = 3

# Promotion gate: a retrained model replaces the current one only if, on the
# held-out evaluation set, its QWK is no more than PROMOTION_QWK_TOLERANCE below
# the current model's and its Grade 3+4 recall no more than
# PROMOTION_RECALL_TOLERANCE below. Otherwise it is saved but not served.
PROMOTION_QWK_TOLERANCE = float(os.getenv("PROMOTION_QWK_TOLERANCE", "0.01"))
PROMOTION_RECALL_TOLERANCE = float(os.getenv("PROMOTION_RECALL_TOLERANCE", "0.05"))

# Retraining trigger: recommend a retrain once this many new labelled images
# have been uploaded since the last one; with AUTO_RETRAIN=true, start it.
RETRAIN_TRIGGER_THRESHOLD = int(os.getenv("RETRAIN_TRIGGER_THRESHOLD", "20"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "false").lower() in ("1", "true", "yes")

# --------------------------------------------------------------------------- #
# Reference data for the Insights page
# --------------------------------------------------------------------------- #
# Original APTOS 2019 train-set class distribution (from the notebook)
APTOS_DISTRIBUTION = {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}

# Held-out metrics of the base model, measured by scripts/evaluate_model.py on
# the 550 test-split images and committed as models/baseline_metrics.json.
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.json"


def load_baseline_metrics() -> dict | None:
    """Headline metrics for the Insights page, or None if the file is missing."""
    try:
        report = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    m = report["metrics"]
    return {
        "accuracy": m["accuracy"],
        "qwk": m["qwk"],
        "auc": m["auc"],
        "weighted_f1": m["weighted_f1"],
        "grade34_recall": m["grade34_recall"],
        "n": m["n"],
        "evaluated_at": report.get("evaluated_at"),
        "source": (f"Measured on {m['n']} held-out test images (APTOS 2019, "
                   f"stratified split, seed {report['split']['seed']}) by "
                   "scripts/evaluate_model.py"),
    }


# Per-grade image-feature summaries of APTOS (scripts/compute_dataset_insights.py).
DATASET_INSIGHTS_PATH = APP_DIR / "static" / "data" / "dataset_insights.json"


def load_dataset_insights() -> dict | None:
    """Feature summaries for the Insights page, or None if the file is missing."""
    try:
        return json.loads(DATASET_INSIGHTS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# Number of samples the base model was originally trained on (for the registry)
BASE_TRAINING_SAMPLES = 2562
