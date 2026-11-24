# Diabetic Retinopathy Detection — MLOps Deployment

A web application that grades retinal fundus images for diabetic retinopathy
(DR) across five severity levels, accepts new labelled images from the field,
and **retrains the model on demand** using the existing model as a pre-trained
base. Built to support scalable, accessible DR screening across Africa, where
specialist ophthalmologists are scarce.

### 🔗 Links

| | |
|---|---|
| **Live app** | https://dr-detection-mlops-69ui.onrender.com |
| **Video demo** | https://youtu.be/5iXWeM2hiyc |
| **Notebook** | [`notebook/diabetic_retinopathy_detection.ipynb`](notebook/diabetic_retinopathy_detection.ipynb) |

> ⚠️ Clinical screening aid only — not a substitute for professional diagnosis.
> The live app sleeps when idle (free tier); the first request after inactivity
> can take 30–60 s to wake.

---

## Features

- **Predict** — upload a fundus image and get its DR grade (0–4) with per-class
  confidence and a clinical action note.
- **Upload** — contribute labelled images to the training pool, saved to a
  database for future retraining.
- **Retrain** — fine-tune a new model version from the pool using the active
  model as a pre-trained base. The convolutional backbone is frozen so only the
  classifier head trains — fast enough for CPU. **The original weights are never
  overwritten.**
- **Rollback** — restore any previous model version with one click.
- **Insights** — dataset class distribution, baseline metrics, and full model
  version history.

## Grade scale

| Grade | Label | Clinical action |
|------:|-------|-----------------|
| 0 | No DR | Routine annual screening |
| 1 | Mild DR | Re-screen in 6–12 months |
| 2 | Moderate DR | Specialist review within 3–6 months |
| 3 | Severe DR | URGENT specialist referral |
| 4 | Proliferative DR | IMMEDIATE ophthalmologist referral |

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │            FastAPI + Uvicorn             │
  Browser  ──HTTP──▶     │  ┌───────────────────────────────────┐  │
  (Jinja2 +              │  │ Routes: predict / upload / retrain │  │
   Chart.js)             │  │         rollback / insights        │  │
                         │  └───────────────────────────────────┘  │
                         │        │            │            │        │
                         │        ▼            ▼            ▼        │
                         │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
                         │  │  model   │ │ retrain  │ │ registry │  │
                         │  │ service  │ │ service  │ │  (JSON)  │  │
                         │  │ (TF/Keras│ │ (freeze +│ │ versions │  │
                         │  │  + lock) │ │ fine-tune│ │ + active │  │
                         │  └────┬─────┘ └────┬─────┘ └────┬─────┘  │
                         └───────┼────────────┼────────────┼────────┘
                                 │            │            │
                          ┌──────▼─────┐ ┌────▼─────┐ ┌────▼─────────┐
                          │ exp5_*.keras│ │ SQLite   │ │ models/      │
                          │ base/current│ │ (uploads)│ │ versions/*   │
                          └────────────┘ └──────────┘ └──────────────┘
```

- **Backend:** FastAPI + Uvicorn
- **Model runtime:** TensorFlow 2.18 (Keras 3), thread-safe reload under a lock
- **Database:** SQLite via SQLAlchemy
- **Frontend:** Jinja2 server-side templates + Chart.js (no build step)
- **Image processing:** OpenCV (headless) + NumPy
- **Container:** Docker + docker-compose
- **Hosting:** Render (Docker runtime)

---

## How retraining works

Retraining demonstrates the full MLOps loop while protecting the original model.

1. **Collect** — labelled fundus images are uploaded via the app and saved to the
   SQLite database, joining a balanced *replay buffer* of seed images. Images in
   the held-out evaluation set are never part of the training pool.
2. **Preprocess** — every image in the pool is run through the **exact same**
   preprocessing the model was trained on (BGR→RGB, resize to 224×224,
   scale to `[0, 1]`). The prediction and retraining paths share one function so
   they can never drift apart.
3. **Fine-tune from a pre-trained base** — the current model is loaded as the
   pre-trained base and its convolutional backbone (layers up to global pooling)
   is **frozen**; only the classifier head trains. This keeps retraining fast on
   CPU and, combined with a low learning rate and the balanced replay buffer,
   prevents catastrophic forgetting.
4. **Version & promote** — the retrained model is written via
   `staging → versioned snapshot → current (active)`. A new entry is appended to
   `models/model_registry.json` and promoted to active. **`exp5_base.keras` is
   never written to** — verified by an automated base-model-integrity test.
5. **Hot-swap** — the served model is reloaded under a lock so in-flight
   predictions are never served a half-swapped model.

### Why a replay buffer?

Fine-tuning a 5-class classifier on only a handful of uploads collapses the head
toward whatever grades were uploaded (catastrophic forgetting). To prevent this,
retraining trains on a **balanced buffer of real images** (40 per grade, drawn
only from the notebook's *training* split) *together with* new uploads.

### Training pool vs. held-out evaluation set

`scripts/prepare_data.py` builds two pools from the notebook's own stratified
split (seed 42), and checks that they never overlap:

| Pool | Folder | Database `source` | Contents | Used for |
|------|--------|-------------------|----------|----------|
| Replay buffer | `data/train/` | `seed` | 40 images per grade from the **training** split | retraining |
| User uploads | `data/uploads/` | `user` | images uploaded in the app | retraining |
| Evaluation set | `data/test/` | `eval` | all 550 images of the held-out **test** split | measuring models, never trained on |

`data/test/manifest.csv` lists the evaluation images and their true grades.

## Model versioning & rollback

- `exp5_base.keras` — the original trained model. **Committed, never modified.**
- `exp5_current.keras` — the active served model (copied from base on first run).
- `models/versions/*.keras` — an immutable snapshot per retrain.
- `models/model_registry.json` — tracks the active version, retrain count, and
  full history (samples trained on, accuracy, timestamps).

Any version — including the original `v1_base` — can be promoted back to active
from the **Insights** page with one click.

---

## Local setup

### With Docker (recommended — matches production)

```bash
docker compose up --build
# open http://localhost:8000
```

Compose mounts `./models` and `./data` as volumes, so uploads, the database, and
retrained versions persist across restarts.

### Without Docker

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   Unix: source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime + tests, Locust, notebook tools
uvicorn app.main:app --reload
# open http://localhost:8000
```

### Getting the data (needed for retraining and evaluation)

Prediction works out of the box. Retraining and the evaluation scripts need the
APTOS 2019 images, which cannot be redistributed:

1. Accept the competition rules at
   <https://www.kaggle.com/competitions/aptos2019-blindness-detection>.
2. Download and unzip into `data/raw/` (so that `data/raw/train.csv` and
   `data/raw/train_images/` exist), for example with the Kaggle CLI:
   `kaggle competitions download -c aptos2019-blindness-detection -p data/raw`.
3. Build the replay buffer and the held-out evaluation set, re-measure the model,
   and pick demo images it grades correctly:

```bash
python -m scripts.prepare_data          # data/train (200) + data/test (550)
python -m scripts.evaluate_model        # -> models/baseline_metrics.json
python -m scripts.pick_demo_images      # -> demo_images/, tests/sample_images/
```

### Running the tests

```bash
pytest
```

Covers preprocessing, prediction, the metrics, the backbone freeze, upload, the
separation of training and evaluation data, and a **base-model integrity** test
asserting the original weights are never modified by registry operations. Tests
that need APTOS images are skipped when `data/raw` or `tests/sample_images` is
empty.

---

## Deployment

### Live: Render (Docker runtime)

The app is deployed on Render from this repository's `Dockerfile`, serving a
public HTTPS URL. Render injects `PORT`; the container binds it automatically.

**Prediction and Insights** run comfortably on the live URL. **Retraining** is
memory-intensive (TensorFlow + `model.fit`) and exceeds Render's free-tier
512 MB limit, so it is demonstrated on the **containerised app running locally**
(`docker compose up`), which also carries the seed replay buffer. This split is
shown in the video demo. For live retraining, deploy to a host with ≥ 2 GB RAM.

`render.yaml` is included for reproducible Render deployment.

### Note on storage & cold starts

Free-tier storage is **ephemeral**: uploaded images, the database, and retrained
versions reset on restart/redeploy. This is acceptable for the demo — the base
model is baked into the image, so prediction always works. Free instances also
sleep when idle; the first request after inactivity takes 30–60 s.

---

## Model & notebook

- `models/exp5_base.keras` — the trained 5-class model (1.24 M params, input
  224×224×3, output 5 classes). Committed; copied to `exp5_current.keras` at
  startup.
- `notebook/diabetic_retinopathy_detection.ipynb` — the notebook that created
  and evaluated the model.

### Measured performance of the deployed model

Measured on the **550 held-out test images** (the notebook's stratified split,
seed 42) with the app's own preprocessing, by `scripts/evaluate_model.py`.
Full results, including the confusion matrix, are in
[`models/baseline_metrics.json`](models/baseline_metrics.json).

| Accuracy | QWK | AUC (one-vs-rest) | Weighted F1 | Grade 3+4 recall |
|---------:|----:|------------------:|------------:|-----------------:|
| 0.618 | 0.588 | 0.839 | 0.625 | 0.390 |

Grade 3+4 recall is the mean of the Severe (0.621) and Proliferative (0.159)
recalls. These numbers come from the committed model file itself; they differ
from the Exp 5 row printed in the notebook, which came from a different training
session than the saved model.

## Project structure

```
src/
  preprocessing.py     image preprocessing + the notebook's train/val/test split
  model.py             load, freeze backbone, fine-tune, evaluate (metrics)
  prediction.py        single-image / batch prediction (also a CLI)
app/
  main.py              FastAPI app, lifespan startup, routers
  config.py            paths, UI metadata, retraining constants
  routes/              predict, upload, retrain, rollback, insights
  services/            model_service, retrain_service, registry
  database/            SQLAlchemy engine + UploadedImage model
  templates/ static/   Jinja2 pages + CSS
data/
  train/               replay buffer (seed images; not committed)
  test/                held-out evaluation set (not committed)
models/                exp5_base.keras (committed) + runtime versions
notebook/              training + evaluation notebook
scripts/
  prepare_data.py      build data/train (replay buffer) + data/test (eval set)
  evaluate_model.py    measure a model on the held-out test split
  pick_demo_images.py  held-out images the model grades correctly
tests/                 src, prediction, upload, base-model integrity
Dockerfile  docker-compose.yml  render.yaml
```

## Known limitations

- **Ephemeral hosted storage** — uploads/retrains reset on redeploy (free tier).
- **Live retraining** needs ≥ 2 GB RAM; demonstrated locally via Docker.
- **Grade 3+4 recall (0.390)** — the model under-detects sight-threatening DR,
  especially Proliferative (recall 0.159), and confuses the middle grades (it
  over-predicts Severe). A clinical deployment would require much higher
  sensitivity on Grades 3 and 4.
- **APTOS images are not redistributed** (competition licensing); build the
  training pool and evaluation set from your own copy via
  `scripts/prepare_data.py`.
