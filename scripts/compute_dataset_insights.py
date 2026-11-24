"""Measure simple per-image features of APTOS 2019 and summarise them by grade.

For every labelled training image it records:
  * size: width, height, megapixels, aspect ratio (width / height), square or not
  * inside the fundus (pixels brighter than a near-black threshold, i.e. the
    round retina without the black border): mean brightness, contrast
    (standard deviation of brightness), and mean red / green / blue
  * sharpness: variance of the Laplacian on a 512-px-wide copy
  * fundus fraction: share of the frame covered by the retina

It then writes per-grade summaries (median, quartiles, 5th/95th percentiles,
mean) and each feature's Spearman correlation with grade to
app/static/data/dataset_insights.json, which the Insights page charts. The
JSON holds no images, so it is committed and works on the hosted app.

Usage:
    python -m scripts.compute_dataset_insights [--aptos-dir data/raw] [--workers 4]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Make the repo packages importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_LABELS, REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "app" / "static" / "data" / "dataset_insights.json"
MASK_THRESHOLD = 10          # grey level separating the retina from the black border

FEATURES = {
    "megapixels": ("Resolution", "megapixels"),
    "aspect": ("Aspect ratio (width / height)", ""),
    "brightness": ("Brightness inside the fundus", "grey level 0–255"),
    "contrast": ("Contrast inside the fundus", "std of grey level"),
    "red": ("Red channel inside the fundus", "0–255"),
    "green": ("Green channel inside the fundus", "0–255"),
    "blue": ("Blue channel inside the fundus", "0–255"),
    "sharpness": ("Sharpness", "variance of Laplacian"),
    "fundus_fraction": ("Share of the frame covered by the retina", "fraction"),
}


def measure(args) -> dict:
    path, grade = args
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > MASK_THRESHOLD
    small = cv2.resize(gray, (512, max(1, round(512 * h / w))))
    return {
        "grade": grade,
        "width": w, "height": h,
        "megapixels": w * h / 1e6,
        "aspect": w / h,
        "square": float(w == h),
        "brightness": float(gray[mask].mean()),
        "contrast": float(gray[mask].std()),
        "red": float(img[..., 2][mask].mean()),
        "green": float(img[..., 1][mask].mean()),
        "blue": float(img[..., 0][mask].mean()),
        "sharpness": float(cv2.Laplacian(small, cv2.CV_64F).var()),
        "fundus_fraction": float(mask.mean()),
    }


def summarise(values: pd.Series) -> dict:
    q = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    r = lambda x: round(float(x), 4)  # noqa: E731
    return {"n": int(values.size), "mean": r(values.mean()), "median": r(q[0.5]),
            "q1": r(q[0.25]), "q3": r(q[0.75]), "p5": r(q[0.05]), "p95": r(q[0.95])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarise APTOS image features by grade.")
    ap.add_argument("--aptos-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    labels = pd.read_csv(args.aptos_dir / "train.csv")
    jobs = [(args.aptos_dir / "train_images" / f"{i}.png", int(g))
            for i, g in zip(labels["id_code"], labels["diagnosis"])]
    print(f"Measuring {len(jobs)} images with {args.workers} workers...")
    with Pool(args.workers) as pool:
        rows = pool.map(measure, jobs, chunksize=16)
    df = pd.DataFrame(rows)

    grades = sorted(CLASS_LABELS)
    features = {}
    for key, (label, unit) in FEATURES.items():
        features[key] = {
            "label": label,
            "unit": unit,
            "spearman": round(float(df[key].corr(df["grade"], method="spearman")), 3),
            "by_grade": [summarise(df.loc[df["grade"] == g, key]) for g in grades],
        }
    square_share = [round(float(df.loc[df["grade"] == g, "square"].mean()), 4) for g in grades]
    counts = [int((df["grade"] == g).sum()) for g in grades]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"APTOS 2019 train.csv ({len(df):,} labelled images)",
        "fundus_mask": f"pixels with grey level > {MASK_THRESHOLD} (excludes the black border)",
        "grades": grades,
        "labels": [CLASS_LABELS[g] for g in grades],
        "counts": counts,
        "square_share": square_share,
        "features": features,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'feature':<16}{'rho':>7}   median by grade 0..4")
    for key, f in features.items():
        meds = "  ".join(f"{b['median']:8.3f}" for b in f["by_grade"])
        print(f"{key:<16}{f['spearman']:>7.3f}   {meds}")
    print(f"{'square share':<16}{'':>7}   " + "  ".join(f"{s:8.3f}" for s in square_share))
    print(f"counts: {counts}")
    print(f"Wrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
