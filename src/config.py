"""Model and dataset constants shared by training, evaluation and serving."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "exp5_base.keras"

# Input size the exp5 model was trained on: (height, width).
TARGET_SIZE = (224, 224)
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

# Train/val/test split used by the training notebook. Reproducing it exactly
# lets the app and scripts evaluate on the same held-out test images.
SPLIT_SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Layers [0, FREEZE_CUTOFF] form the convolutional backbone of exp5, ending at
# the global average pooling layer (index 30). Only the dense head after it is
# trained when fine-tuning.
FREEZE_CUTOFF = 30
