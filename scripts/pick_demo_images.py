"""Pick held-out demo images that the base model grades correctly.

Scores the held-out test images (data/test/manifest.csv) with the base model
and, for each grade, copies the most confident CORRECT predictions to
demo_images/heldout_grade{g}_{id}.png. The best one per grade also goes to
tests/sample_images/grade_{g}.png for the test suite. Both folders are
gitignored (APTOS licence).

Run scripts.prepare_data first. Images are read from data/test/, or from
data/raw/train_images/ when data/test/ has not been filled.

Usage:
    python -m scripts.pick_demo_images [--per-grade 3]
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

# Make the repo packages importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_LABELS, DEFAULT_MODEL_PATH, REPO_ROOT  # noqa: E402
from src.model import load_model  # noqa: E402
from src.prediction import predict_proba  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "test" / "manifest.csv"
DEMO_DIR = REPO_ROOT / "demo_images"
SAMPLES_DIR = REPO_ROOT / "tests" / "sample_images"


def _image_path(id_code: str) -> Path:
    prepared = REPO_ROOT / "data" / "test" / f"{id_code}.png"
    return prepared if prepared.exists() else REPO_ROOT / "data" / "raw" / "train_images" / f"{id_code}.png"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pick correctly graded held-out demo images.")
    ap.add_argument("--per-grade", type=int, default=3)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = ap.parse_args(argv)

    if not MANIFEST.exists():
        raise SystemExit("data/test/manifest.csv not found; run `python -m scripts.prepare_data` first.")
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = [(r["id_code"], int(r["diagnosis"])) for r in csv.DictReader(fh)]

    print(f"Scoring {len(rows)} held-out images with {args.model.name}...")
    probs = predict_proba(load_model(args.model), [_image_path(i) for i, _ in rows])

    DEMO_DIR.mkdir(exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in DEMO_DIR.glob("heldout_grade*.png"):
        old.unlink()

    print(f"\n{'grade':<22}{'correct':>9}   top picks (id, confidence)")
    for grade, label in CLASS_LABELS.items():
        correct = [(probs[k][grade], id_code) for k, (id_code, true) in enumerate(rows)
                   if true == grade and probs[k].argmax() == grade]
        total = sum(1 for _, true in rows if true == grade)
        correct.sort(reverse=True)
        picks = correct[:args.per_grade]
        for conf, id_code in picks:
            shutil.copyfile(_image_path(id_code), DEMO_DIR / f"heldout_grade{grade}_{id_code}.png")
        if picks:
            shutil.copyfile(_image_path(picks[0][1]), SAMPLES_DIR / f"grade_{grade}.png")
        shown = ", ".join(f"{i} ({c:.2f})" for c, i in picks)
        print(f"{grade} {label:<20}{len(correct):>4}/{total:<4}   {shown}")

    print(f"\nWrote demo images to demo_images/ and test samples to tests/sample_images/.")
    print("For the video, prefer images with confidence >= 0.6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
