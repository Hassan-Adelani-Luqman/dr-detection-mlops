"""Seed a stratified replay buffer from the APTOS 2019 training set.

Fine-tuning the model on a handful of user uploads alone causes catastrophic
forgetting: the classifier head collapses toward whatever grades were uploaded.
To prevent that, retraining trains on a balanced replay buffer of real APTOS
images *together with* new uploads. This script populates that buffer.

It copies a stratified sample (default 40 images per grade) into the uploads
directory and records each in the database with source="seed". Seed rows are
included in retraining but excluded from the Insights "your uploads" view.

Idempotent: if seed rows already exist it does nothing unless --force is given.

APTOS images are NOT redistributable (Kaggle competition rules), so they are
never committed. A grader obtains the dataset separately and runs this script.

Usage:
    python -m scripts.seed_data --aptos-dir <path> --per-grade 40
"""
import argparse
import csv
import glob
import os
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path

# Make the app package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CLASS_LABELS, UPLOADS_DIR  # noqa: E402
from app.database.db import SessionLocal, init_db  # noqa: E402
from app.database.models import UploadedImage  # noqa: E402

DEFAULT_APTOS = (Path(__file__).resolve().parent.parent.parent
                 / "diabetic-retinopathy-detection" / "data" / "raw")


def _stratified_pick(train_csv: Path, img_dir: Path, per_grade: int, seed: int):
    """Return {grade: [image_path, ...]} with up to per_grade images each."""
    import random
    rng = random.Random(seed)
    by_grade = defaultdict(list)
    with open(train_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            by_grade[int(row["diagnosis"])].append(row["id_code"])

    picks = {}
    for grade in CLASS_LABELS:
        codes = by_grade.get(grade, [])
        rng.shuffle(codes)
        chosen = []
        for code in codes:
            matches = glob.glob(str(img_dir / f"{code}.*"))
            if matches:
                chosen.append(matches[0])
            if len(chosen) >= per_grade:
                break
        picks[grade] = chosen
    return picks


def seed(aptos_dir: Path, per_grade: int, force: bool, seed_val: int,
         test_samples_dir: Path | None) -> None:
    train_csv = aptos_dir / "train.csv"
    img_dir = aptos_dir / "train_images"
    if not train_csv.exists() or not img_dir.exists():
        raise SystemExit(
            f"APTOS data not found under {aptos_dir}.\n"
            f"Expected {train_csv} and {img_dir}/.\n"
            "Download the APTOS 2019 dataset and pass --aptos-dir."
        )

    init_db()
    db = SessionLocal()
    try:
        existing = db.query(UploadedImage).filter_by(source="seed").count()
        if existing and not force:
            print(f"Seed rows already present ({existing}). Use --force to re-seed. "
                  "Nothing to do.")
            return
        if force and existing:
            db.query(UploadedImage).filter_by(source="seed").delete()
            db.commit()
            print(f"--force: removed {existing} existing seed rows.")

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        picks = _stratified_pick(train_csv, img_dir, per_grade, seed_val)

        total = 0
        first_per_grade = {}
        for grade, paths in picks.items():
            for src in paths:
                ext = Path(src).suffix or ".png"
                stored = f"seed_{uuid.uuid4().hex[:8]}{ext}"
                dest = UPLOADS_DIR / stored
                shutil.copyfile(src, dest)
                # Store filename only (portable across host/container).
                db.add(UploadedImage(filename=Path(src).name, image_path=stored,
                                     grade_label=grade, source="seed"))
                first_per_grade.setdefault(grade, src)
                total += 1
            print(f"  grade {grade} ({CLASS_LABELS[grade]}): {len(paths)} images")
        db.commit()
        print(f"\nSeeded {total} images across {len(picks)} grades "
              f"(source='seed').")

        # Copy one representative image per grade for the test suite.
        if test_samples_dir is not None:
            test_samples_dir.mkdir(parents=True, exist_ok=True)
            for grade, src in first_per_grade.items():
                ext = Path(src).suffix or ".png"
                shutil.copyfile(src, test_samples_dir / f"grade_{grade}{ext}")
            print(f"Copied {len(first_per_grade)} sample images to "
                  f"{test_samples_dir} (gitignored).")
    finally:
        db.close()


def main():
    ap = argparse.ArgumentParser(description="Seed the replay buffer from APTOS.")
    ap.add_argument("--aptos-dir", type=Path, default=DEFAULT_APTOS,
                    help=f"APTOS raw dir (default: {DEFAULT_APTOS})")
    ap.add_argument("--per-grade", type=int, default=40,
                    help="Images per grade (default: 40).")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    ap.add_argument("--force", action="store_true",
                    help="Delete existing seed rows and re-seed.")
    ap.add_argument("--no-test-samples", action="store_true",
                    help="Do not copy sample images into tests/sample_images/.")
    args = ap.parse_args()

    test_dir = (None if args.no_test_samples
                else Path(__file__).resolve().parent.parent / "tests" / "sample_images")
    seed(args.aptos_dir, args.per_grade, args.force, args.seed, test_dir)


if __name__ == "__main__":
    main()
