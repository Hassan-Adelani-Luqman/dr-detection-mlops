# DR Detection MLOps — Execution Plan

**Project:** Diabetic Retinopathy Detection — MLOps Deployment
**Marks:** 45 (5 video + 10 retraining + 10 prediction + 10 evaluation + 10 deployment)
**Extends:** `coursework/diabetic-retinopathy-detection` (Intro to ML summative)

---

## Part A — Preflight Findings

Verified against the actual machine and filesystem on 2026-07-20. Four items
diverge from the original brief and are resolved below.

### A.1 Assets confirmed present

| Asset | Location | Size | Status |
|---|---|---|---|
| `exp5_deep_cnn.keras` | project root | 15.06 MB | ✅ present, commit-safe (<100 MB) |
| `diabetic_retinopathy_detection.ipynb` | project root | 2.15 MB | ✅ present |
| APTOS `train_images/` + `train.csv` | `../diabetic-retinopathy-detection/data/raw/` | — | ✅ available locally |
| Docker | 29.5.2 | — | ✅ |
| git | 2.53.0 | — | ✅ |

### A.2 BLOCKER — Python 3.14.3 cannot run TensorFlow

System Python is **3.14.3**. TensorFlow 2.18 ships wheels for **3.9–3.12 only**.
`pip install tensorflow-cpu` will fail with "no matching distribution". No
project venv exists yet.

**Decision — install Python 3.11 and create a project venv.**
Docker-only development is the fallback, but every code change would need a
30–60 s image rebuild, which makes Phases 2–8 painful. A local 3.11 venv gives
fast iteration; Docker then validates parity once at Phase 10.

```powershell
winget install Python.Python.3.11
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pin `python-3.11` in the Dockerfile so local and container runtimes match.

### A.3 RISK — the `layers[:30]` freeze cutoff is unverified

The brief hardcodes freezing `model.layers[:30]`. The model has only 1.24 M
params; if it has fewer than 30 layers, this freezes **every** layer, leaving
zero trainable weights — `model.fit()` then either errors or silently trains
nothing, and the retraining criterion fails while appearing to work.

**Decision — derive the cutoff, never hardcode it.** Phase 0.5 interrogates the
model; `retrain_service` splits at the last flattening layer:

```python
def freeze_backbone(model):
    """Freeze everything up to and including the last Flatten/GlobalPool layer.
    Everything after it (the classifier head) stays trainable."""
    cut = max(
        (i for i, l in enumerate(model.layers)
         if isinstance(l, (tf.keras.layers.Flatten,
                           tf.keras.layers.GlobalAveragePooling2D,
                           tf.keras.layers.GlobalMaxPooling2D))),
        default=len(model.layers) - 2,
    )
    for layer in model.layers[:cut + 1]:
        layer.trainable = False
    assert model.trainable_weights, "freeze cutoff left no trainable weights"
    return cut
```

The assertion is the gate — it converts a silent failure into a loud one.

### A.4 RISK — retraining on uploads alone will destroy the model

The brief retrains on uploaded images only, with `MIN_SAMPLES = 4`. Fine-tuning
a 5-class classifier on ~10 images for 15 epochs causes catastrophic forgetting:
the head collapses toward whatever grades were uploaded. The video demo shows a
prediction *after* retraining (step 5) — a collapsed model returns a wrong grade
there and jeopardises the 10-mark prediction criterion at the exact moment the
grader is watching.

**Decision — seed a stratified replay buffer.** Phase 4.5 loads ~200 APTOS
images (40 per grade) into `data/uploads/` and the DB, flagged
`source="seed"`. Retraining then trains on replay + new uploads together, so
the gradient signal stays balanced and the model genuinely improves rather than
collapsing.

Supporting changes: `MIN_SAMPLES = 20`, `epochs = 8`, LR `1e-5` (not `1e-4`),
and a `source` column on `UploadedImage` to distinguish seed from user data.
Insights charts filter to `source="user"` so the demo still shows the user's own
uploads clearly.

### A.5 RISK — Render free tier has 512 MB RAM; TF needs more

`tensorflow-cpu` alone is ~400–600 MB resident before the model or FastAPI
load. Retraining allocates a further several hundred MB. Render's free instance
will almost certainly OOM-kill the container, most likely mid-retrain.

**Decision — deploy to Hugging Face Spaces (Docker SDK) as primary.**
Free tier gives 16 GB RAM and 2 vCPU, takes the same Dockerfile, and issues a
public HTTPS URL — it satisfies the rubric identically. Keep `render.yaml` in
the repo as a documented alternative. Ship whichever is live first; do not
burn time fighting a 512 MB ceiling.

HF Spaces note: it expects port **7860** by default, and the container runs as
a non-root user, so `data/` and `models/` need write permissions set in the
Dockerfile.

### A.6 Resolved — APTOS sample image licensing

Kaggle competition rules restrict redistribution, so **do not commit fundus
images**. `scripts/seed_data.py` copies a stratified sample from the local
APTOS path at setup time; the README documents how a grader obtains the data.
`tests/sample_images/` stays gitignored, and tests skip gracefully when it is
empty so CI never fails on missing data.

### A.7 Open question for you

**Does the grader need to reproduce retraining without the APTOS dataset?**
If yes, commit ~10 heavily-downscaled images as a fallback seed and note the
provenance. If no, the seed script alone is sufficient. Plan assumes **no**.

---

## Part B — Phase Map

Each phase ends with a **verification gate** and exactly one commit. Do not
proceed past a red gate. Estimates assume the A.2 venv exists.

| Phase | Deliverable | Est. |
|---|---|---|
| 0 | Repo skeleton, assets, deps | 45 m |
| 0.5 | Model interrogation report | 20 m |
| 1 | Config + SQLite layer | 45 m |
| 2 | Model service + registry | 1.5 h |
| 3 | FastAPI app + prediction | 2 h |
| 4 | Upload + persistence | 1 h |
| 4.5 | Replay-buffer seed script | 45 m |
| 5 | Retraining + versioning | 3 h |
| 6 | Rollback | 45 m |
| 7 | Insights + charts | 2 h |
| 8 | Landing page + polish | 1.5 h |
| 9 | Tests | 1 h |
| 10 | Docker | 1 h |
| 11 | Public deployment | 1.5 h |
| 12 | README | 1 h |
| 13 | Video + submission | 1.5 h |
| | **Total** | **~21 h** |

---

## PHASE 0 — Repository Initialisation

**Tasks**
1. `git init`; set default branch `main`.
2. Create the full tree (Section 3 of the brief) plus `scripts/`.
3. `exp5_deep_cnn.keras` → `models/exp5_base.keras` (copy, keep the original).
4. `diabetic_retinopathy_detection.ipynb` → `notebooks/` (copy, never edit).
5. `.gitignore` per brief, plus `tests/sample_images/*` and `!.gitkeep`.
6. `.dockerignore` = gitignore + `.git/`, `tests/`, `notebooks/`, `.venv/`,
   `*.md`. Excluding the 2.15 MB notebook and venv keeps build context small.
7. `requirements.txt` per brief + `pytest==8.3.3`, `httpx==0.27.2`
   (needed by FastAPI's `TestClient`), `pandas==2.2.3` (seed script).
8. Create the venv per A.2 and install.

**Gate** — `python -c "import tensorflow, cv2, fastapi, sqlalchemy"` exits 0.
Confirm `git status` shows `models/exp5_base.keras` staged but
`exp5_current.keras` ignored.

✅ `chore: initialise MLOps repo structure with base model and notebook`

---

## PHASE 0.5 — Model Interrogation

Cheap, and it de-risks Phases 2 and 5. Writes a throwaway report to the
scratchpad, not the repo.

**Tasks**
1. Load `models/exp5_base.keras`; print `model.summary()`, layer count, each
   layer's type/name/param count, `input_shape`, `output_shape`.
2. Confirm output is `(None, 5)` and input `(None, 224, 224, 3)`.
3. Locate the last Flatten/GlobalPool index — this is the A.3 freeze cutoff.
4. Predict on one known-grade APTOS image using the Section 2 preprocessing;
   confirm the softmax sums to ~1.0 and the argmax is plausible.
5. Record whether the model contains BatchNorm layers — if it does, note that
   frozen BN layers must stay in inference mode during fine-tuning
   (`training=False`), or the running statistics drift and wreck accuracy.

**Gate** — a documented freeze cutoff index and one verified correct
prediction. If step 4 gives a wrong grade, **stop**: preprocessing does not
match training, and every downstream phase inherits the bug.

*No commit — investigation only.*

---

## PHASE 1 — Config and Database Layer

**Tasks**
1. `app/config.py` — path constants, `CLASS_LABELS`, `CLINICAL_NOTES`,
   `TARGET_SIZE`, `MIN_SAMPLES = 20`, `RETRAIN_EPOCHS = 8`,
   `RETRAIN_LR = 1e-5`, `APTOS_DISTRIBUTION`, `BASELINE_METRICS`.
   Read `DATABASE_URL` / `UPLOADS_DIR` from env with these as defaults so the
   container can redirect them without a code change.
2. `app/database/db.py` — engine with
   `connect_args={"check_same_thread": False}` (required: the retrain thread
   touches the DB from outside the request thread), `SessionLocal`, `Base`,
   `init_db()`, `get_db()`.
3. `app/database/models.py` — `UploadedImage` per brief **plus**
   `source = Column(String, default="user")` per A.4.

**Gate** — a scratch script runs `init_db()`, inserts and reads back one row,
then deletes the db file.

✅ `feat: add config constants and SQLite database layer`

---

## PHASE 2 — Model Service and Registry

**Tasks**
1. `app/services/registry.py` — `init_registry()` (creates the JSON and copies
   base→current on first run), `get_registry()`, `update_registry()`,
   `get_active_version()`, `get_history()`. Write via temp-file + `os.replace`
   so a crash mid-write cannot corrupt the registry.
2. `app/services/preprocess.py` — `preprocess_image()` exactly per Section 2.
   No Ben Graham. Add a docstring stating exp5 was trained without it.
3. `app/services/model_service.py` — module-level cache + `threading.RLock`,
   `load_current_model()`, `reload_model()`, `predict()` returning the brief's
   dict shape. Load once at startup.

**Gate** — `predict()` on the Phase 0.5 image returns the same grade and
confidence vector as the raw interrogation script. Divergence means a bug in
the service wrapper.

✅ `feat: add model service with inference and version registry`

---

## PHASE 3 — FastAPI Skeleton and Prediction

**Tasks**
1. `app/main.py` — app factory, `lifespan` handler running `init_db()`,
   `init_registry()`, `load_current_model()`, static mount, Jinja2 setup,
   routers, `/` landing route, `/health` returning active version.
   Use `lifespan`, not the deprecated `@app.on_event("startup")`.
2. `app/routes/predict.py` — `GET /predict`; `POST /api/predict` accepting
   `UploadFile`, writing to a temp path, predicting, cleaning up in a
   `finally`. Validate content-type and magic bytes; 400 on non-images with a
   readable message.
3. `app/templates/base.html` — nav, dark clinical theme, Chart.js CDN.
   **Vendor Chart.js into `static/js/` instead of the CDN** if the deployment
   target has a restrictive CSP; decide at Phase 11.
4. `app/templates/predict.html` — drag-and-drop, colour-coded grade badge,
   Chart.js confidence bars, clinical note, loading spinner.
5. `app/static/css/style.css`.

**Gate** — `uvicorn app.main:app --reload`, upload a known-grade fundus image
through the browser, get the **correct** grade. This is the 10-mark prediction
criterion; do not move on until it is right.

✅ `feat: add FastAPI app skeleton and image prediction endpoint + page`

---

## PHASE 4 — Upload and Persistence

**Tasks**
1. `app/routes/upload.py` — `GET /upload`; `POST /api/upload` with
   `file: UploadFile` and `grade_label: int = Form(...)`. Validate 0–4, cap
   file size (~10 MB), save as `data/uploads/<uuid8>_<safe_name>`, insert the
   row with `source="user"`, return the new total.
   Sanitise the filename — never trust the client-supplied path component.
2. `app/templates/upload.html` — form, grade dropdown with labels, live
   per-grade counts, success toast.

**Gate** — upload 4 images across grades; files exist on disk and rows exist in
the DB with correct labels.

✅ `feat: add image upload endpoint with database persistence`

---

## PHASE 4.5 — Replay Buffer Seed (per A.4)

**Tasks**
1. `scripts/seed_data.py` — CLI taking `--aptos-dir` (default: the local
   sibling path) and `--per-grade 40`. Reads `train.csv`, samples stratified by
   `diagnosis`, copies images into `data/uploads/`, inserts rows with
   `source="seed"`. Idempotent: skip if seed rows already exist.
2. Document it in the README as a **required setup step** for retraining.
3. Copy 5 images (one per grade) to `tests/sample_images/` for Phase 9 —
   gitignored per A.6.

**Gate** — DB holds ~200 seed rows with a roughly even grade spread;
`/upload` counts reflect them.

✅ `feat: add stratified APTOS replay-buffer seed script`

---

## PHASE 5 — Retraining Service (core criterion)

Must visibly demonstrate all three sub-requirements: **upload+save** (Phase 4),
**preprocess** (`build_dataset`), **retrain from pre-trained base**
(`build_retrain_model`).

**Tasks**
1. `app/services/retrain_service.py`
   - `build_retrain_model()` — load `exp5_current.keras`, apply the A.3
     derived-cutoff freeze, compile with Adam @ `RETRAIN_LR`,
     `sparse_categorical_crossentropy`.
   - `build_dataset()` — query all rows (seed + user), preprocess each via the
     shared `preprocess_image`, stratified 80/20 train/val split, shuffle,
     batch 8, prefetch. Raise a friendly error below `MIN_SAMPLES`.
     Skip rows whose file is missing rather than crashing the run.
   - `run_retraining()` — fit with `EarlyStopping(monitor="val_loss",
     patience=3, restore_best_weights=True)`, then **staging → versioned →
     current** copy. Never `save()` to `BASE_MODEL_PATH`.
   - Guard against concurrent runs with a module-level flag; a second POST
     while running returns 409.
2. `app/routes/retrain.py` — `GET /retrain`; `POST /api/retrain` spawning a
   `threading.Thread` (not `BackgroundTasks` — that still runs in the request
   lifecycle); `GET /api/retrain/status` returning
   `{status, current_epoch, total_epochs, logs[], result}`.
   A Keras callback appends per-epoch metrics to the shared status dict.
   Wrap the thread body in try/except so a crash sets `status="error"` with the
   message instead of hanging on "running" forever.
3. `app/templates/retrain.html` — start button, 3 s polling, live epoch log,
   completion summary with the new version name.

**Gate** — record `Get-FileHash models/exp5_base.keras` before and after a full
retrain. Then verify: new file in `models/versions/`, `exp5_current.keras`
updated, **base hash identical**, registry `active_version` advanced, and a
post-retrain prediction still returns the correct grade (the A.4 check).

✅ `feat: add retraining service with frozen-backbone fine-tuning and model versioning`

---

## PHASE 6 — Rollback

**Tasks**
1. `app/routes/rollback.py` — `POST /api/rollback/{version}`. Validate the
   version against the registry **and** that the file exists; reject path
   traversal in the version string. Copy → current, update registry, call
   `reload_model()` under the lock, return confirmation.
2. Special-case `v1_base` — it lives at `models/exp5_base.keras`, not in
   `versions/`. This is the version most likely demoed, so get it right.

**Gate** — retrain to v2, roll back to `v1_base`, confirm the registry updates
and predictions match pre-retrain output.

✅ `feat: add model version rollback endpoint`

---

## PHASE 7 — Insights Page (deployment criterion)

**Tasks**
1. `app/routes/insights.py` — `GET /insights`; `GET /api/insights` returning
   user-upload distribution, seed distribution, hardcoded APTOS distribution
   `{0:1805, 1:370, 2:999, 3:193, 4:295}`, model metadata, baseline metrics
   (QWK 0.711, Grade 3+4 recall 0.372, AUC 0.846, accuracy 0.673), and full
   version history.
2. `app/templates/insights.html` — grouped bar chart (uploads vs APTOS,
   normalised to % so the 1805-vs-handful scale gap doesn't flatten the
   uploads), metrics cards, version-history table with per-row Rollback.
3. Label the baseline metrics clearly as *Intro to ML held-out test set* so
   they are not mistaken for live post-retrain metrics.

**Gate** — charts render with real data; rollback button works end-to-end.

✅ `feat: add insights page with dataset visualisations and model history`

---

## PHASE 8 — Landing Page and Polish

**Tasks**
1. `app/templates/index.html` — title, Africa DR-screening mission line, four
   feature cards, active-version indicator.
2. Responsive nav, consistent spacing, spinners on predict and retrain, error
   states on every fetch call.
3. Full manual pass: predict → upload → retrain → predict → insights →
   rollback → predict.

**Gate** — the complete loop runs clean with no console errors.

✅ `feat: add landing page and polish UI across all pages`

---

## PHASE 9 — Tests

**Tasks**
1. `tests/conftest.py` — fixtures for a temp DB and a `TestClient`; a
   `skipif` marker when `tests/sample_images/` is empty.
2. `test_prediction.py` — grade in 0–4, confidences sum ≈ 1.0, all 5 classes
   present, correct grade on a known sample.
3. `test_upload.py` — POST creates a DB row; invalid grade → 400; non-image
   → 400.
4. `test_registry.py` — **base-model integrity**: hash `exp5_base.keras`, run a
   registry update + version promotion, assert the hash is unchanged. This is
   the assertion that protects the highest-risk requirement.

**Gate** — `pytest` green.

✅ `test: add prediction, upload, and base-model integrity tests`

---

## PHASE 10 — Dockerisation

**Tasks**
1. `Dockerfile` on `python:3.11-slim`, `libglib2.0-0` for headless OpenCV,
   requirements layer cached before `COPY . .`, non-root user, `EXPOSE 8000`.
2. `docker-compose.yml` with `./models` and `./data` volumes for persistence.
3. Build, run, and re-test predict + retrain **inside** the container — this is
   where a Windows/Linux path or permissions bug surfaces.

**Gate** — `docker compose up --build`, prediction works at `localhost:8000`,
image size noted (expect 1–1.5 GB).

✅ `build: add Dockerfile and docker-compose for containerised deployment`

---

## PHASE 11 — Public Deployment (per A.5)

**Tasks**
1. Push to a **public** GitHub repo.
2. Primary: Hugging Face Space, Docker SDK. Add the port-7860 and
   writable-directory adjustments; keep them env-driven so local compose is
   unaffected.
3. Keep `render.yaml` committed and document the 512 MB constraint honestly in
   the README.
4. Test the public URL end-to-end, including a full retrain.
5. Note cold-start behaviour for the video.

**Gate** — public HTTPS URL serving correct predictions and completing a
retrain without OOM.

✅ `deploy: add container deployment config and public hosting`

---

## PHASE 12 — README

**Tasks** — title and mission; **public URL near the top**; ASCII architecture
diagram; features; local setup (venv **and** compose); the seed-data step;
how retraining works (pre-trained base → derived freeze cutoff → fine-tune head
→ version → promote — state the pre-trained-model concept explicitly, the
rubric grades it); versioning and rollback; replay-buffer rationale; link to
`notebooks/` noting it satisfies the evaluation criterion; tech stack;
video link placeholder; known limitations (ephemeral storage, cold starts,
Grade 3/4 recall 0.372).

**Gate** — a fresh reader can clone and run it from the README alone. Confirm
both the model file and notebook are committed.

✅ `docs: add comprehensive README with setup, architecture, and usage`

---

## PHASE 13 — Video and Submission

Warm the URL 2 minutes before recording. Camera on throughout, 5–7 min:
landing page → **prediction** (correct grade + confidence + note) → **upload**
(4–5 labelled images, count rises) → **retraining** (live epochs; pause
recording during fit; resume for the new version) → **prediction again** on the
retrained model → **insights** charts and history → rollback → notebook in
repo.

Then: add the video link to the README, final commit, ZIP the repo, submit.

✅ `docs: add video demo link`

---

## Part C — Risk Register

| Risk | Impact | Mitigation | Phase |
|---|---|---|---|
| Python 3.14 blocks TF | Cannot start | Install 3.11 venv | 0 |
| Freeze cutoff freezes all layers | Retraining silently no-ops | Derived cutoff + assertion | 0.5, 5 |
| Catastrophic forgetting | Wrong grade in demo | Replay buffer, low LR, fewer epochs | 4.5, 5 |
| Render 512 MB OOM | No public URL | HF Spaces primary | 11 |
| Base model overwritten | Rubric failure | staging→versioned→current + hash test | 5, 9 |
| Preprocessing mismatch | Wrong predictions, hard to debug | Single shared `preprocess_image`; verified Phase 0.5 | 2 |
| Retrain blocks request | Timeout on host | Thread + status polling | 5 |
| Frozen BatchNorm drift | Accuracy collapse | Check in 0.5; `training=False` if present | 5 |
| APTOS redistribution | Licensing | Seed script, images gitignored | 4.5 |

## Part D — Submission Checklist

- [ ] Public URL live and tested
- [ ] `models/exp5_base.keras` committed
- [ ] `notebooks/diabetic_retinopathy_detection.ipynb` committed, unmodified
- [ ] All phase commits present, sequential
- [ ] Prediction returns correct grades (before **and** after retrain)
- [ ] Retraining creates a new version; base hash unchanged
- [ ] Rollback works
- [ ] Insights page shows visualisations
- [ ] `pytest` green
- [ ] README has public URL + video link
- [ ] Video recorded, camera on, prediction + retraining both shown
- [ ] Repo public on GitHub
- [ ] ZIP + URL submitted
