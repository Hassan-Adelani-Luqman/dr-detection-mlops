"""Central configuration: paths, class metadata, and retraining constants.

Values that the container may need to override (database location, uploads
directory) are read from the environment with sensible local defaults, so the
same code runs unchanged locally and in Docker.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent          # repo root
APP_DIR = BASE_DIR / "app"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
VERSIONS_DIR = MODELS_DIR / "versions"

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(DATA_DIR / "uploads")))

BASE_MODEL_PATH = MODELS_DIR / "exp5_base.keras"           # ORIGINAL — never written
CURRENT_MODEL_PATH = MODELS_DIR / "exp5_current.keras"     # active served model
STAGING_MODEL_PATH = MODELS_DIR / "exp5_staging.keras"     # transient write target
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# Ensure runtime directories exist (safe if already present)
for _d in (MODELS_DIR, DATA_DIR, VERSIONS_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}")

# --------------------------------------------------------------------------- #
# Model / inference
# --------------------------------------------------------------------------- #
TARGET_SIZE = (224, 224)          # (height, width) — matches exp5 training
NUM_CLASSES = 5

CLASS_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

CLINICAL_NOTES = {
    0: "No diabetic retinopathy detected. Routine annual screening recommended.",
    1: "Mild non-proliferative DR. Re-screen in 6–12 months.",
    2: "Moderate non-proliferative DR. Specialist review within 3–6 months.",
    3: "Severe non-proliferative DR. URGENT specialist referral required.",
    4: "Proliferative DR. IMMEDIATE ophthalmologist referral — sight-threatening.",
}

# UI colour per grade (green -> red severity ramp)
GRADE_COLORS = {
    0: "#2e7d32",   # green
    1: "#f9a825",   # yellow
    2: "#ef6c00",   # orange
    3: "#c62828",   # red
    4: "#7b1fa2",   # purple (proliferative)
}

# --------------------------------------------------------------------------- #
# Retraining (tuned in Part A of the plan to avoid catastrophic forgetting)
# --------------------------------------------------------------------------- #
FREEZE_CUTOFF = 30                # freeze layers[:FREEZE_CUTOFF+1] (verified Phase 0.5)
MIN_SAMPLES = 20                  # refuse to retrain below this many images
RETRAIN_EPOCHS = 8
RETRAIN_BATCH_SIZE = 8
RETRAIN_LR = 1e-5                 # low LR: fine-tune the head without wrecking it
EARLY_STOP_PATIENCE = 3

# --------------------------------------------------------------------------- #
# Reference data for the Insights page
# --------------------------------------------------------------------------- #
# Original APTOS 2019 train-set class distribution (from the notebook)
APTOS_DISTRIBUTION = {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}

# Held-out test metrics from the Intro-to-ML summative evaluation of exp5.
# Labelled clearly as baseline (pre-retrain) so they are not read as live metrics.
BASELINE_METRICS = {
    "quadratic_weighted_kappa": 0.711,
    "grade_3_4_recall": 0.372,
    "auc": 0.846,
    "accuracy": 0.673,
    "source": "Intro to ML summative — exp5 held-out test set",
}

# Number of samples the base model was originally trained on (for the registry)
BASE_TRAINING_SAMPLES = 2562
