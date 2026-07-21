---
title: DR Detection MLOps
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Diabetic Retinopathy Detection — MLOps Deployment

A web application that grades retinal fundus images for diabetic retinopathy
(DR) across five severity levels, accepts new labelled images from the field,
and **retrains the model on demand** using the existing model as a pre-trained
base. Built to support scalable, accessible DR screening across Africa, where
specialist ophthalmologists are scarce.

> **Public URL:** _<add after deploying — see Deployment below>_
>
> **Video demo:** _<add after recording>_

> ⚠️ Screening aid only — not a substitute for clinical diagnosis.

---

## Features

- **Predict** — upload a fundus image, get its DR grade (0–4) with per-class
  confidence and a clinical action note.
- **Upload** — contribute labelled images to the training pool (saved to a
  database).
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

## Deployment

### Primary: Hugging Face Spaces (Docker SDK)

Chosen for its free tier's 16 GB RAM, which comfortably runs the TensorFlow
image and on-demand retraining. HF builds the image from the `Dockerfile` and
serves it at a public HTTPS URL. The `app_port: 8000` in the metadata above
matches the Dockerfile.

1. Create a new Space → SDK: **Docker** → **Blank**.
2. Push this repository to the Space's git remote.
3. HF builds and deploys automatically; the URL appears at the top of the Space.

### Alternative: Render

`render.yaml` is included, but Render's free tier (512 MB RAM) will likely be
OOM-killed during retraining. Suitable for prediction-only demos or a paid
instance (≥ 2 GB RAM). See comments in `render.yaml`.

### Note on storage (both hosts)

Free tiers use **ephemeral** storage: uploaded images, the SQLite database, and
retrained model versions are lost on restart/redeploy. This is acceptable for
the demo — the base model is baked into the image, so prediction always works.
For persistence, attach a Render Disk or HF persistent storage (paid).

### Cold starts

Free tiers spin down when idle; the first request after inactivity can take
30–60 s. Load the page a minute before demoing.

---

## Local setup

### With Docker (recommended)

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

### Seeding the training pool (for retraining locally)

Retraining trains on a balanced replay buffer of real images plus any uploads.
APTOS images are **not** redistributed here (Kaggle competition rules), so seed
the buffer from your own copy of the APTOS 2019 dataset:

```bash
python -m scripts.seed_data --aptos-dir /path/to/aptos/raw --per-grade 40
```

On the hosted demo (no seed buffer), upload ~5 images across grades via the
Upload page, then retrain — `MIN_SAMPLES` is set low to allow this.

---

## Model & notebook

- `models/exp5_base.keras` — the trained 5-class model (1.24 M params, input
  224×224×3). Committed; the app copies it to `exp5_current.keras` at startup.
- `notebooks/diabetic_retinopathy_detection.ipynb` — the notebook that created
  and evaluated the model (satisfies the model-evaluation criterion).

_Full architecture, tech stack, and how-retraining-works detail: see below
(expanded in the documentation section)._
