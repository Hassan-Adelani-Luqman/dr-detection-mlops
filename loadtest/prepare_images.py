"""Copy held-out fundus images into loadtest/images/ for the Locust test.

Picks a fixed, stratified sample (4 per grade by default) from the held-out
evaluation set in data/test/ (built by `python -m scripts.prepare_data`),
falling back to data/raw/train_images/. Full-size images are used, so the
measured latency includes decoding a realistic multi-megabyte upload.
The folder is gitignored (APTOS licence).

    python loadtest/prepare_images.py [--per-grade 4]
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "images"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-grade", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(ROOT / "data" / "test" / "manifest.csv", newline="") as fh:
        rows = [(r["id_code"], int(r["diagnosis"])) for r in csv.DictReader(fh)]
    rng = random.Random(args.seed)
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()

    total_mb = 0.0
    for grade in range(5):
        ids = sorted(i for i, g in rows if g == grade)
        for id_code in rng.sample(ids, min(args.per_grade, len(ids))):
            src = ROOT / "data" / "test" / f"{id_code}.png"
            if not src.exists():
                src = ROOT / "data" / "raw" / "train_images" / f"{id_code}.png"
            shutil.copyfile(src, OUT / f"grade{grade}_{id_code}.png")
            total_mb += src.stat().st_size / 1e6
    n = len(list(OUT.glob("*.png")))
    print(f"Copied {n} held-out images to {OUT.relative_to(ROOT)} "
          f"(mean size {total_mb / max(n, 1):.2f} MB)")


if __name__ == "__main__":
    main()
