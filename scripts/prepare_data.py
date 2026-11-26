"""Prepare the local training and evaluation pools from a copy of APTOS 2019.

It builds two pools that never overlap, both drawn from the training notebook's
own split (src.preprocessing.stratified_split):

* Replay buffer -> data/train/, database source "seed".
  A stratified sample of the TRAINING split (40 per grade by default).
  Retraining uses it together with user uploads, so fine-tuning on a few new
  images does not make the model forget the grades it already knows.
* Evaluation set -> data/test/, database source "eval".
  All 550 images of the held-out TEST split. Never trained on; used to measure
  the deployed model and to compare retrained models with it.

data/test/manifest.csv (id_code, diagnosis) is written for reference and is
committed. The images are not committed: APTOS is Kaggle competition data and
cannot be redistributed.

Usage:
    python -m scripts.prepare_data                       # build both pools (skips any that exist)
    python -m scripts.prepare_data --force               # rebuild both
    python -m scripts.prepare_data --train-per-grade 0   # evaluation set only
    python -m scripts.prepare_data --no-eval             # replay buffer only
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

# Make the repo packages importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (CLASS_LABELS, DATA_DIR, TEST_DIR, TRAIN_DIR,  # noqa: E402
                        UPLOADS_DIR)
from app.database.db import SessionLocal, init_db  # noqa: E402
from app.database.models import UploadedImage  # noqa: E402
from app.services.pool import resolve_image_path  # noqa: E402
from src.config import REPO_ROOT, SPLIT_SEED  # noqa: E402
from src.preprocessing import load_labels, stratified_split  # noqa: E402

MANIFEST = TEST_DIR / "manifest.csv"
KEEP = {"README.md", ".gitkeep", "manifest.csv"}   # never deleted from data/train|test


def pick_seed_ids(train_df, per_grade: int, seed: int = SPLIT_SEED) -> dict:
    """``{grade: [id_code, ...]}``: up to ``per_grade`` random ids per grade."""
    picks = {}
    for grade in CLASS_LABELS:
        rows = train_df[train_df["diagnosis"] == grade]
        n = min(per_grade, len(rows))
        picks[grade] = sorted(rows.sample(n=n, random_state=seed)["id_code"]) if n else []
    return picks


def _empty_folder(folder: Path) -> int:
    removed = 0
    for f in folder.iterdir():
        if f.is_file() and f.name not in KEEP:
            f.unlink()
            removed += 1
    return removed


def _clear_seed(db) -> None:
    """Remove every seed row and file, including the old data/uploads/seed_* layout."""
    rows = db.query(UploadedImage).filter_by(source="seed").all()
    for r in rows:
        path = resolve_image_path(r.image_path)
        if path.exists() and DATA_DIR.resolve() in path.resolve().parents:
            path.unlink()
        db.delete(r)
    legacy = list(UPLOADS_DIR.glob("seed_*"))
    for f in legacy:
        f.unlink()
    db.commit()
    files = _empty_folder(TRAIN_DIR)
    print(f"  removed {len(rows)} seed rows, {files} files in data/train, "
          f"{len(legacy)} legacy files in data/uploads")


def _clear_prepared_eval(db) -> None:
    """Remove the prepared eval rows (data/test/...). User-uploaded eval images stay."""
    rows = (db.query(UploadedImage)
              .filter(UploadedImage.source == "eval",
                      UploadedImage.image_path.like("test/%"))
              .all())
    for r in rows:
        db.delete(r)
    db.commit()
    files = _empty_folder(TEST_DIR)
    print(f"  removed {len(rows)} prepared eval rows and {files} files in data/test")


def _add_images(db, items, img_dir: Path, dest: Path, source: str) -> None:
    """Copy ``(id_code, grade)`` images into ``dest`` and record them in the database."""
    dest.mkdir(parents=True, exist_ok=True)
    rel = dest.relative_to(DATA_DIR).as_posix()
    for id_code, grade in items:
        name = f"{id_code}.png"
        shutil.copyfile(img_dir / name, dest / name)
        db.add(UploadedImage(filename=name, image_path=f"{rel}/{name}",
                             grade_label=int(grade), source=source))
    db.commit()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prepare the replay buffer and evaluation set.")
    ap.add_argument("--aptos-dir", type=Path, default=REPO_ROOT / "data" / "raw",
                    help="Folder with APTOS train.csv and train_images/ (default: data/raw).")
    ap.add_argument("--train-per-grade", type=int, default=40,
                    help="Replay-buffer images per grade; 0 skips the buffer (default: 40).")
    ap.add_argument("--no-eval", action="store_true", help="Do not build the evaluation set.")
    ap.add_argument("--force", action="store_true", help="Rebuild pools that already exist.")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED, help="Sampling seed (default: 42).")
    args = ap.parse_args(argv)

    csv_path, img_dir = args.aptos_dir / "train.csv", args.aptos_dir / "train_images"
    if not csv_path.exists() or not img_dir.is_dir():
        raise SystemExit(f"APTOS data not found: expected {csv_path} and {img_dir}/.\n"
                         "Download APTOS 2019 from Kaggle and pass --aptos-dir.")

    train, _, test = stratified_split(load_labels(csv_path))
    init_db()
    db = SessionLocal()
    try:
        if args.train_per_grade > 0:
            existing = db.query(UploadedImage).filter_by(source="seed").count()
            if existing and not args.force:
                print(f"Replay buffer: {existing} seed rows already present (use --force).")
            else:
                print("Replay buffer (training split only):")
                if existing:
                    _clear_seed(db)
                picks = pick_seed_ids(train, args.train_per_grade, args.seed)
                items = [(i, g) for g, ids in picks.items() for i in ids]
                _add_images(db, items, img_dir, TRAIN_DIR, "seed")
                for g, ids in picks.items():
                    print(f"  grade {g} ({CLASS_LABELS[g]}): {len(ids)}")
                print(f"  -> {len(items)} images in data/train")

        if not args.no_eval:
            existing = (db.query(UploadedImage)
                          .filter(UploadedImage.source == "eval",
                                  UploadedImage.image_path.like("test/%"))
                          .count())
            if existing and not args.force:
                print(f"Evaluation set: {existing} prepared eval rows already present "
                      "(use --force).")
            else:
                print("Evaluation set (held-out test split):")
                if existing:
                    _clear_prepared_eval(db)
                items = list(zip(test["id_code"], test["diagnosis"]))
                _add_images(db, items, img_dir, TEST_DIR, "eval")
                with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh, lineterminator="\n")
                    writer.writerow(["id_code", "diagnosis"])
                    writer.writerows(items)
                counts = test["diagnosis"].value_counts().sort_index()
                print("  " + ", ".join(f"grade {g}: {n}" for g, n in counts.items()))
                print(f"  -> {len(items)} images in data/test, manifest at data/test/manifest.csv")

        # The whole point: no evaluation image may sit in the training pool.
        seed_ids = {Path(r.filename).stem for r in db.query(UploadedImage).filter_by(source="seed")}
        overlap = seed_ids & set(test["id_code"])
        if overlap:
            raise SystemExit(f"ERROR: {len(overlap)} replay-buffer images are in the "
                             f"held-out test split, e.g. {sorted(overlap)[0]}")
        print("Check passed: the replay buffer and the evaluation set do not overlap.")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
