"""Prediction: turn fundus images into DR grades.

Library use::

    from src.model import load_model
    from src.prediction import predict_image
    result = predict_image(load_model(), "fundus.png")

Command line::

    python -m src.prediction path/to/image.png [more.png ...] [--model models/exp5_base.keras]
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from src.config import CLASS_LABELS, CLINICAL_NOTES, DEFAULT_MODEL_PATH
from src.preprocessing import preprocess_batch, preprocess_image

PathLike = Union[str, Path]


def predict_proba(model, images: Union[np.ndarray, Sequence[PathLike]],
                  batch_size: int = 32) -> np.ndarray:
    """Return class probabilities, shape (N, 5).

    ``images`` is either a preprocessed array (N, 224, 224, 3) or a sequence of
    image paths. Paths are loaded batch by batch, so large sets never sit in
    memory all at once.
    """
    if isinstance(images, np.ndarray):
        return model.predict(images, batch_size=batch_size, verbose=0)
    paths = list(images)
    out = []
    for start in range(0, len(paths), batch_size):
        batch = np.stack([preprocess_image(p) for p in paths[start:start + batch_size]])
        out.append(model.predict(batch, batch_size=batch_size, verbose=0))
    return np.concatenate(out) if out else np.empty((0, len(CLASS_LABELS)))


def format_prediction(probs: np.ndarray) -> dict:
    """Build the result dict for one image from its 5 class probabilities."""
    grade = int(np.argmax(probs))
    return {
        "predicted_grade": grade,
        "grade_label": CLASS_LABELS[grade],
        "clinical_note": CLINICAL_NOTES[grade],
        "confidence": round(float(probs[grade]), 4),
        "confidence_scores": {
            CLASS_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
    }


def predict_image(model, path: PathLike) -> dict:
    """Preprocess one image, run the model, and return the formatted result."""
    probs = model.predict(preprocess_batch(path), verbose=0)[0]
    return format_prediction(probs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict the DR grade of fundus images.")
    parser.add_argument("images", nargs="+", type=Path, help="Image file(s) to grade.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help=f"Keras model file (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--json", action="store_true", help="Print raw JSON results.")
    args = parser.parse_args(argv)

    from src.model import load_model  # deferred: keeps the module import light

    model = load_model(args.model)
    for path in args.images:
        result = {"image": str(path), **predict_image(model, path)}
        if args.json:
            print(json.dumps(result))
        else:
            print(f"{path}: grade {result['predicted_grade']} "
                  f"({result['grade_label']}), confidence {result['confidence']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
