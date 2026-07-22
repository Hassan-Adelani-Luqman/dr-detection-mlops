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
| **Notebook** | [`notebooks/diabetic_retinopathy_detection.ipynb`](notebooks/diabetic_retinopathy_detection.ipynb) |

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
   SQLite database, joining a balanced *replay buffer* of seed images.
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
retraining trains on a **balanced buffer of real images** (40 per grade)
*together with* new uploads. Seed the buffer from your own APTOS copy:

```bash
python -m scripts.seed_data --aptos-dir /path/to/aptos/raw --per-grade 40
```

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
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000
```

### Running the tests

```bash
pytest
```

Includes prediction, upload, and a **base-model integrity** test asserting the
original weights are never modified by registry operations.

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
- `notebooks/diabetic_retinopathy_detection.ipynb` — the notebook that created
  and evaluated the model. Baseline held-out metrics: QWK **0.711**,
  accuracy **0.673**, AUC **0.846**, Grade 3+4 recall **0.372**.

## Project structure

```
app/
  main.py              FastAPI app, lifespan startup, routers
  config.py            paths, class labels, retraining constants
  routes/              predict, upload, retrain, rollback, insights
  services/            model_service, retrain_service, registry, preprocess
  database/            SQLAlchemy engine + UploadedImage model
  templates/ static/   Jinja2 pages + CSS
models/                exp5_base.keras (committed) + runtime versions
notebooks/             evaluation notebook
scripts/seed_data.py   stratified APTOS replay-buffer seeder
tests/                 prediction, upload, base-model integrity
Dockerfile  docker-compose.yml  render.yaml
```

## Known limitations

- **Ephemeral hosted storage** — uploads/retrains reset on redeploy (free tier).
- **Live retraining** needs ≥ 2 GB RAM; demonstrated locally via Docker.
- **Grade 3/4 recall (0.372)** — the model under-detects severe/proliferative DR
  due to class imbalance in the source data; a clinical deployment would require
  higher sensitivity on sight-threatening grades.
- **APTOS images are not redistributed** (competition licensing); seed the
  training pool from your own copy via `scripts/seed_data.py`.
