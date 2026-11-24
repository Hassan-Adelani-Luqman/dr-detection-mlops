"""Measure a model on the held-out APTOS test split and save the metrics.

The test split is rebuilt exactly as the training notebook made it (stratified,
seed 42, 15% val / 15% test -> 550 images), and images go through the same
preprocessing the app uses, so the numbers describe the model as deployed.

Usage:
    python -m scripts.evaluate_model                       # base model -> models/baseline_metrics.json
    python -m scripts.evaluate_model --model models/exp5_current.keras --out /tmp/current.json
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the repo packages importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (CLASS_LABELS, DEFAULT_MODEL_PATH, REPO_ROOT,  # noqa: E402
                        SPLIT_SEED, TEST_SIZE, VAL_SIZE)
from src.model import evaluate_model, load_model  # noqa: E402
from src.preprocessing import load_labels, stratified_split  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "models" / "baseline_metrics.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _display(path: Path) -> str:
    """Repo-relative path when possible (keeps the JSON machine-independent)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--aptos-dir", type=Path, default=REPO_ROOT / "data" / "raw",
                    help="Folder with APTOS train.csv and train_images/ (default: data/raw).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    _, _, test = stratified_split(load_labels(args.aptos_dir / "train.csv"))
    paths = [args.aptos_dir / "train_images" / f"{i}.png" for i in test["id_code"]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test images missing, e.g. {missing[0]}")

    print(f"Evaluating {_display(args.model)} on {len(paths)} held-out test images...")
    metrics = evaluate_model(load_model(args.model), paths, test["diagnosis"].to_numpy())

    report = {
        "model_file": _display(args.model),
        "model_sha256": _sha256(args.model),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "APTOS 2019 Blindness Detection, train.csv",
        "split": {
            "subset": "test",
            "method": "two-stage stratified split, as in the training notebook",
            "seed": SPLIT_SEED, "val_size": VAL_SIZE, "test_size": TEST_SIZE,
        },
        "preprocessing": "BGR->RGB, resize 224x224, scale to [0, 1] (src.preprocessing)",
        "class_labels": [CLASS_LABELS[g] for g in sorted(CLASS_LABELS)],
        "notes": "grade34_recall is the mean of the class-3 and class-4 recall; "
                 "auc is one-vs-rest macro AUC.",
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    m = metrics
    print(f"  accuracy        {m['accuracy']:.4f}")
    print(f"  QWK             {m['qwk']:.4f}")
    print(f"  AUC (OvR)       {m['auc']:.4f}")
    print(f"  weighted F1     {m['weighted_f1']:.4f}")
    print(f"  grade 3/4 recall {m['grade34_recall']:.4f}")
    print(f"Wrote {_display(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
