# Implementation Plan — Summative Fixes (DR Detection MLOps)

**Goal:** close every gap between this repo and (a) the MLOPs rubric and (b) the full
assignment brief, then submit. The full brief is the page saved at
`~/Downloads/Summative assignment - MLOP_files/pub.html`. It asks for more than the rubric
does: Locust load tests, uptime in the UI, explanations of at least 3 dataset features,
bulk upload, and a specific repo layout.

**Deadline:** Thursday 26 Nov 2026, 23:59 (Africa/Harare). There are two submission
attempts: **attempt 1 = zip of the GitHub repo**, **attempt 2 = GitHub repo URL**.

**How to use this plan:** do one phase at a time, in order. Each phase lists its tasks,
the files it touches, how to check it is done, and a commit message. Tick the boxes as
you go and update the phase's **Status** line.

---

## 0. Where things stand (audit of 2026-11-21)

**Already done and working:** prediction page and API, single-image upload saved to SQLite,
shared preprocessing, retraining from the existing model with a frozen backbone, model
versioning and rollback, the notebook (preprocessing, optimisation, at least 4 metrics),
the live Render URL, Docker, the YouTube video link, and 11 passing tests.

**Gaps this plan fixes:**

| # | Gap | Source | Phase |
|---|-----|--------|-------|
| G1 | No `src/preprocessing.py`, `src/model.py`, `src/prediction.py`, no `data/train` or `data/test`, folder named `notebooks/` | Brief (repo layout) | 1 |
| G2 | Baseline metrics shown in the app and README (0.673 / 0.711 / 0.846 / 0.372) are not this model's. The deployed file actually scores accuracy 0.618, QWK 0.588, AUC 0.839, weighted F1 0.625, Grade 3/4 recall 0.390 | Integrity | 2 |
| G3 | The seed buffer contains 34 of 200 images from the held-out test split | Integrity | 2 |
| G4 | All grade-0 demo images come from the training split, and most grade 1–4 demo images predict wrong | Rubric (correct prediction) | 2 |
| G5 | Model uptime is not shown anywhere | Brief (UI) | 3 |
| G6 | Inference runs on the event loop, so one prediction blocks the whole server | Prerequisite for G7 | 3 |
| G7 | No Locust flood test, no multi-container setup, no results in the README | Brief | 4 |
| G8 | Only one dataset chart and no written interpretation; the brief needs at least 3 features explained | Brief | 5 |
| G9 | Upload takes one image at a time; the brief requires bulk upload | Brief | 6 |
| G10 | Retrained models are never compared with the current model on held-out data, and are always promoted | Brief ("evaluate in production") | 7 |
| G11 | The Insights "Accuracy" column shows training accuracy (v2 appears as 41.6%) | Correctness | 7 |
| G12 | Retraining on Render will crash the 512 MB instance; it only runs in local Docker | Brief (deploy) / risk | 8 |
| G13 | 4 preprocessing/EDA visualisation cells in the notebook were never run; there is no "load saved model → predict" cell | Rubric (notebook) | 9 |
| G14 | README is missing flood results, dataset download steps and correct metrics; stale comments; video needs updating | Brief (README) | 10 |

**Priority legend:** **P0** = required by the brief or rubric, or currently wrong.
**P1** = strengthens the grade. **P2** = polish, do only if time allows.

---

## Ground rules (all phases)

- **Never write to `models/exp5_base.keras`.** The base-integrity test must stay green.
- **Never commit APTOS images** (Kaggle competition licence). Images live in gitignored
  folders and are recreated by scripts; only ids and labels (`manifest.csv`) are committed.
- `pytest` must pass at the end of every phase.
- After every phase, check the app manually: `uvicorn app.main:app --reload`, then open
  Predict, Upload, Retrain and Insights.
- Prediction on the Render URL must keep working after every deploy.
- Commit at the end of each phase, using the same commit-message style as the existing
  history (`feat:`, `fix:`, `docs:` and so on).
- Work on branch `fixes` and merge into `main` before submission. Graders see the
  default branch.

---

## Decisions (with defaults, so no phase is blocked)

| ID | Decision | Default | Change if… |
|----|----------|---------|------------|
| D1 | Where retraining runs in the cloud | Turn retraining off on Render with a clear message (Phase 8.1), then **try** Hugging Face Spaces (Phase 8.2) | You'd rather pay for a Render instance with at least 2 GB |
| D2 | Held-out evaluation set | All 550 test-split images locally; in the cloud, an "Evaluation set" upload | — |
| D3 | When a retrained model gets promoted | Candidate QWK ≥ current QWK − 0.01 **and** Grade 3/4 recall ≥ current − 0.05 | You want a stricter or looser gate |
| D4 | Also ship a `.h5` copy of the model | No; the README explains that `.keras` is TensorFlow's native format | You want the brief's file types matched literally (+15 MB) |
| D5 | Re-record the video | Yes, after Phase 10, so it shows the evaluation step, bulk upload and correct held-out predictions | The current video already shows the camera throughout and you are short on time |
| D6 | Commit sample APTOS images | No; tests generate synthetic images | — |
| D7 | Load-test matrix | 1, 2 and 3 containers × 10, 25 and 50 users, 2 minutes each | This machine is 4 cores / about 5 GB free, so 4 containers won't fit |

---

## Phase 0 — Setup and safety net  (P0, ~30 min)

**Status:** ☑ done 2026-11-24

- [x] 0.1 Create branch `fixes`.
- [x] 0.2 Linux virtualenv at `.venv` (Python 3.12.3), with `requirements-dev.txt` installed.
- [x] 0.3 Split the requirements:
  - `requirements.txt` (what the Docker image needs): the previous list, plus
    `scikit-learn==1.9.1`; `pytest` and `httpx` moved out.
  - New `requirements-dev.txt`: `-r requirements.txt` plus pinned `pytest`, `httpx`,
    `locust==2.47.0`, `matplotlib`, `nbclient`, `ipykernel`, `seaborn`, `tqdm`,
    `imbalanced-learn`, `scikit-image`, `kagglehub`. numpy stays at 1.26.4 and `pip check`
    is clean.
  - Note: Locust must be imported before anything that loads `ssl`, so always run it through
    the `locust` CLI and never import it next to the app or TensorFlow.
- [x] 0.4 `data/raw` is a symlink to `../../diabetic-retinopathy-detection/data/raw`
  (3,662 train and 1,928 test images plus `train.csv`). Ignored in `.gitignore` (no trailing
  slash, so the pattern matches the symlink) and in `.dockerignore`. Scripts default to
  `--aptos-dir data/raw`.
  - That folder also holds a second, identical 9.6 GB copy in
    `aptos2019-blindness-detection/` (same file names and sizes). It isn't used and can be
    deleted.
- [x] 0.5 Local runtime state backed up to `backup/pre-fixes/` (gitignored, with a
  README.txt): registry (active `v2_retrained_20260722_202339`), `exp5_current.keras`, the
  `v2` and `v3` versions, `app.db`, and all 212 files in `data/uploads/`.
- [x] 0.6 Baseline check: `pytest` → 11 passed.

**Done when:** the new venv runs `pytest` green, `data/raw/train.csv` resolves, and the backup exists. ✔
**Commit:** `chore: split dev requirements and link local APTOS data`

---

## Phase 1 — Repo layout and shared ML code in `src/`  (P0, ~3 h) — G1

**Status:** ☐ not started

The brief requires this layout. Move the pure ML code into `src/`. `app/` keeps only the
web layer, the database and the registry, and imports from `src/`. `src/` must never
import from `app/`.

- [ ] 1.1 `src/__init__.py`, plus `src/config.py` holding `TARGET_SIZE`, `NUM_CLASSES`,
  `CLASS_LABELS`, `CLINICAL_NOTES`, and the split constants (`SPLIT_SEED=42`,
  `VAL_SIZE=0.15`, `TEST_SIZE=0.15`). `app/config.py` imports these instead of
  defining them.
- [ ] 1.2 `src/preprocessing.py`
  - `preprocess_image()` and `preprocess_batch()`, moved verbatim from
    `app/services/preprocess.py` (BGR→RGB, resize to 224, /255). Delete the old module.
  - `load_labels(csv_path)` and `stratified_split(df)`. These must reproduce the notebook's
    two-stage split **exactly**: `train_test_split(test_size=0.15, stratify, random_state=42)`,
    then the same again with `test_size=0.15/0.85` → 2562 / 550 / 550.
  - `split_of_ids(aptos_dir) -> dict[id_code, "train"|"val"|"test"]`.
- [ ] 1.3 `src/model.py`
  - `load_model(path)` and `freeze_backbone(model, cutoff=30)`. Layer 30 is
    `global_average_pooling2d_2`, so the dense head (layers 31+) is what trains.
  - `build_finetune_model(base_path, lr)` and `fine_tune(model, train_ds, val_ds, epochs, callbacks)`.
  - `evaluate_model(model, paths_or_array, y, batch_size=32) -> dict`. It returns accuracy,
    weighted F1, macro F1, QWK, AUC (one-vs-rest), per-class recall, the confusion matrix,
    and `grade34_recall`. Define `grade34_recall` as the **mean of the class-3 and class-4
    recall**, which is how the original project defines it. Load images in batches so 550
    images don't all sit in memory at once.
- [ ] 1.4 `src/prediction.py`
  - `predict_proba(model, paths, batch_size)` and
    `predict_image(model, path) -> {predicted_grade, grade_label, clinical_note, confidence, confidence_scores}`.
  - A command-line entry point: `python -m src.prediction <image> [--model models/exp5_base.keras]`.
- [ ] 1.5 Rewire the app:
  - `app/services/model_service.py` → uses `src.prediction.predict_image`.
  - `app/services/retrain_service.py` → uses `src.model` and `src.preprocessing`.
  - Fix the imports in `tests/`.
  - Remove the dangling `IMPLEMENTATION_PLAN.md` / "Phase 0.5" / "Part A" references in
    `app/config.py`, `retrain_service.py` and the old preprocess docstring.
- [ ] 1.6 Rename `notebooks/` to `notebook/` with `git mv`. Then:
  - In notebook cell 8, change `PROJECT_ROOT.name == 'notebooks'` to
    `in ('notebooks', 'notebook')`.
  - Update README links and `.dockerignore`.
- [ ] 1.7 Create `data/train/` and `data/test/`, each with a `README.md` and `.gitkeep`:
  - `train/` holds the replay buffer, `test/` the held-out evaluation set (Phase 2),
    `uploads/` user data, and `raw/` the APTOS source.
  - Gitignore the images in both folders but keep the README and `manifest.csv`.
- [ ] 1.8 New `tests/test_src.py`, using synthetic images so it runs on a fresh clone:
  - Preprocessing returns shape (224, 224, 3), float32, values in [0, 1].
  - `predict_image` returns all its keys and the probabilities sum to 1.
  - `evaluate_model` on perfect predictions gives QWK = 1 and accuracy = 1.
  - `stratified_split` sizes are 2562 / 550 / 550 (skip if `data/raw` is absent).
- [ ] 1.9 (P2, only if D4 = yes) `scripts/export_h5.py` writes `models/exp5_base.h5` and
  checks that its predictions match the `.keras` model (max absolute difference < 1e-5).

**Done when:**
- `pytest` passes.
- `python -m src.prediction demo_images/grade0_1.png` prints grade 0.
- `docker compose build` succeeds, and the app predicts through the UI.

**Commit:** `refactor: move ML core into src/ (preprocessing, model, prediction) to match required layout`

---

## Phase 2 — Correct metrics, data separation and held-out demo images  (P0, ~3 h) — G2, G3, G4

**Status:** ☐ not started

- [ ] 2.1 `scripts/evaluate_model.py --model models/exp5_base.keras --aptos-dir data/raw --out models/baseline_metrics.json`
  - Rebuild the test split with `src.preprocessing.stratified_split` and score it with
    `src.model.evaluate_model`.
  - Write to the JSON file: the metrics, the confusion matrix, `n_test`, a description of
    the split, the model's SHA-256, and a timestamp. **Commit this JSON.**
  - **Must reproduce, within ±0.001:** accuracy 0.6182, QWK 0.5879, AUC 0.8386, weighted
    F1 0.6252, `grade34_recall` 0.390, n = 550.
- [ ] 2.2 `app/config.py`: load `BASELINE_METRICS` from `models/baseline_metrics.json`
  instead of hard-coding it. The Insights metric cards then show accuracy, QWK, AUC,
  weighted F1 and Grade 3/4 recall, with the note "measured on 550 held-out images by
  `scripts/evaluate_model.py`".
- [ ] 2.3 README: replace 0.711 / 0.673 / 0.846 / 0.372 (lines ~193–194 and ~217) with the
  measured values and point to `baseline_metrics.json`.
- [ ] 2.4 Rename `scripts/seed_data.py` to `scripts/prepare_data.py` and extend it:
  - `--train-per-grade 40`: a stratified replay buffer drawn **only from training-split
    ids**. Copy the images to `data/train/` and add database rows with `source="seed"`.
  - `--eval`: all 550 **test-split** images go to `data/test/` with database rows
    `source="eval"`. Write `data/test/manifest.csv` (`id_code,diagnosis`) and **commit it**.
  - `--force`: delete the old seed and eval rows and files, including the legacy
    `data/uploads/seed_*`.
  - Copy one test-split image per grade into `tests/sample_images/`.
  - Store `image_path` **relative to `DATA_DIR`** (for example `train/seed_ab12.png`).
  - In `retrain_service` resolve paths like this: relative to `DATA_DIR` first; for older
    rows that hold only a filename, fall back to `UPLOADS_DIR`.
- [ ] 2.5 Training must **exclude `source="eval"`** rows (filter in `build_dataset`).
  Insights counts must report seed, user and eval rows separately.
- [ ] 2.6 New `tests/test_data_split.py`:
  - No `id_code` in `data/test/manifest.csv` appears among the seed rows (the `filename`
    column holds the original APTOS file name).
  - The training query never returns eval rows.
- [ ] 2.7 Reset local runtime state: delete the registry, `exp5_current.keras`,
  `models/versions/*.keras` and `data/app.db` (they are already backed up in 0.5). Restart
  the app, which re-initialises at `v1_base`. Then run
  `python -m scripts.prepare_data --train-per-grade 40 --eval --force`.
- [ ] 2.8 New `scripts/pick_demo_images.py`:
  - Scores the test split with the base model.
  - For each grade, copies the top-3 **correctly predicted held-out** images to
    `demo_images/heldout_grade{g}_{id}.png` (still gitignored).
  - Deletes the old training-split demo images.
  - Reference picks from the audit: grade 0 `cd314653a4d8` (1.00), grade 1 `05a5183c92d0`
    (0.69), grade 3 `e1fb532f55df` and `b019a49787c1` (0.84). Grades 2 and 4 peak at about
    0.36 and 0.33 confidence, so **don't demo those grades**.

**Done when:**
- `baseline_metrics.json` matches the numbers in 2.1.
- The disjointness test passes and Insights shows the measured metrics.
- The registry is at `v1_base`, with 200 seed rows (all from the training split) and 550
  eval rows.

**Commit:** `fix: report measured held-out metrics for the deployed model; separate train and eval data`

---

## Phase 3 — Stop inference blocking the server; add uptime and monitoring  (P0, ~3 h) — G5, G6

**Status:** ☐ not started

- [ ] 3.1 Keep inference off the event loop. Change `api_predict`, `api_upload` and
  `api_rollback` from `async def` to plain `def`, so FastAPI runs them in its threadpool;
  read uploads with `file.file.read()`. The model lock stays: it is deliberate, and it
  limits each container to one inference at a time, which is exactly what the load test
  in Phase 4 measures.
- [ ] 3.2 Move the image magic-byte and size checks into `app/utils/validation.py`, shared
  by predict and upload. Add a 10 MB limit to predict.
- [ ] 3.3 Keep every container on the right model version. `model_service` remembers
  `_loaded_version`. On each predict, if the registry's `active_version` differs, it
  reloads under the lock. This keeps all replicas in sync after a retrain or rollback.
- [ ] 3.4 New `app/services/monitor.py`:
  - `STARTED_AT` is set in `lifespan`. It counts requests, predictions and errors, and keeps
    the last 1000 prediction latencies (a `deque`).
  - `snapshot()` returns `uptime_seconds`, `started_at`, `hostname` (the container id),
    `active_version`, `model_loaded`, `predictions_served`, `latency_ms`
    (mean, p50, p95), `retrain_count` and `last_retrained`.
  - A middleware adds the response headers `X-Served-By: <hostname>` and
    `X-Process-Time-ms`. Phase 4 uses these to prove the load is spread across containers.
- [ ] 3.5 API: extend `/health` with `uptime_seconds` and `hostname`; add `GET /api/status`
  returning the full snapshot.
- [ ] 3.6 UI:
  - **A status pill in the header of every page** (`base.html`). It polls `/health` every
    15 s and shows "● Online · up 2h 14m · v1_base", or a red "Offline" when the call fails.
  - **A "Model status" card at the top of Insights**: uptime, start time, active version,
    predictions served, mean and p95 latency, which container answered, last retrained.
- [ ] 3.7 Tests:
  - `/health` has `uptime_seconds >= 0`.
  - `/api/status` returns all its keys.
  - The `X-Served-By` header is present.
  - After one predict, `predictions_served` goes up.

**Done when:** you fire 20 predictions in parallel
(`seq 20 | xargs -P 20 -I{} curl -s -F file=@demo_images/... localhost:8000/api/predict`)
and `curl localhost:8000/health` still answers in under 200 ms, with uptime visible in the UI.
**Commit:** `feat: add uptime and monitoring status; serve inference off the event loop`

---

## Phase 4 — Locust flood test across multiple containers  (P0, ~5 h incl. runs) — G7

**Status:** ☐ not started

- [ ] 4.1 Rework `docker-compose.yml`:
  - `web`: **no fixed host port** (`expose: ["8000"]` only, so `--scale` works),
    `cpus: "1.0"`, `mem_limit: 1500m`, and environment
    `TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1`, `OMP_NUM_THREADS=1`. Keep the
    existing volumes and healthcheck.
  - New `lb` service: `nginx:1.27-alpine`, `ports: ["8000:80"]`, mounting
    `loadtest/nginx.conf`, with `depends_on: web: condition: service_healthy`.
  - With these settings, adding a container adds a whole CPU. Otherwise one TensorFlow
    process grabs all cores and extra containers barely help.
  - In `model_service`, also apply the thread settings with `tf.config.threading.*`
    **before** loading the model.
- [ ] 4.2 `loadtest/nginx.conf`:
  - `upstream web_pool { server web:8000; }`. nginx resolves `web` to every replica's IP
    when it starts.
  - `client_max_body_size 20m;` (the 1 MB default would reject fundus images),
    `proxy_read_timeout 600s;`, and pass `X-Served-By` through.
  - **After scaling, run `docker compose restart lb`** so nginx picks up the new replicas.
  - `docker compose up` with no flags still gives 1 replica on port 8000, so the README
    instructions stay valid.
- [ ] 4.3 `loadtest/locustfile.py`:
  - `PredictUser(HttpUser)` with `wait_time = between(0.5, 1.5)`.
  - Task weight 10: `POST /api/predict` with a random image from `loadtest/images/`. Use
    `catch_response`, and fail on anything other than 200 or a response missing
    `predicted_grade`.
  - Task weight 1: `GET /health`.
  - On test stop, print a count of responses per `X-Served-By` value, which proves
    round-robin across containers.
- [ ] 4.4 `loadtest/prepare_images.sh`: copy 20 held-out images from `data/test/` into
  `loadtest/images/` (gitignored). Use full-size images, so decoding them counts towards
  the measured latency.
- [ ] 4.5 `loadtest/run_matrix.sh`. For R in 1 2 3:
  1. `docker compose up -d --scale web=$R`, then `docker compose restart lb`.
  2. Wait for `/health`, then send 10 warm-up predictions.
  3. For U in 10 25 50:
     `locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u $U -r 5 -t 2m --csv loadtest/results/r${R}_u${U}`.
  4. Save `docker stats --no-stream` to `loadtest/results/r${R}_docker_stats.txt`.

  The whole matrix takes about 25 minutes. Close other heavy apps first, since about 5 GB
  of RAM is free.
- [ ] 4.6 `loadtest/summarize.py`:
  - Read every `*_stats.csv` and take the `/api/predict` and Aggregated rows.
  - Write `loadtest/results/summary.md` with: containers, users, requests, failure %,
    req/s, mean, median, p95 and p99 latency (ms).
  - Plot `loadtest/results/latency_vs_containers.png`: p95 latency and req/s against the
    number of containers, one line per user count.
- [ ] 4.7 Cloud run: warm up Render, then run Locust against it with `-u 10 -r 2 -t 1m`.
  Report it as its own row ("Render free tier, 1 instance, 0.1 CPU / 512 MB").
- [ ] 4.8 Commit `summary.md`, the chart and the `*_stats.csv` files. Gitignore
  `*_stats_history.csv`, `*.html` and `loadtest/images/`.
- [ ] 4.9 README section titled **"Flood Request Simulation (Locust) — Results"**, matching
  the brief's wording. Cover: the setup (4-core host, 1 CPU per container, nginx
  round-robin), the results table, the chart, 3–5 sentences on what the numbers show
  (throughput against containers, latency at a fixed number of users, where it saturates,
  failures), and how to reproduce it.
  - A rough estimate from the audit: about 145 ms of CPU per inference, plus decoding a
    multi-MB PNG, so roughly 4–6 req/s per container. **Report what you measure, not this.**
- [ ] 4.10 Document the caveat: retraining status is kept in each container's memory, so
  **scale to 1 container before retraining**. A stretch goal is to move it into
  `data/retrain_status.json` with a file lock.

**Done when:** all 9 local runs and the cloud run have CSVs, the summary table and chart
are in the README, and the `X-Served-By` counts show load spread over every container.
**Commit:** `feat: add Locust load tests with nginx-balanced replicas and results`

---

## Phase 5 — Dataset insights: at least 3 features, each with an interpretation  (P0, ~4 h) — G8

**Status:** ☐ not started

- [ ] 5.1 `scripts/compute_dataset_insights.py --aptos-dir data/raw --out app/static/data/dataset_insights.json`
  - For every image (3,662 of them, using `multiprocessing.Pool(4)`, about 5 min):
    width, height, megapixels, aspect ratio, whether it is square, and inside the fundus
    mask (grey > 10): brightness, contrast (standard deviation), and mean R, G and B.
    Also sharpness (Laplacian variance at 512 px).
  - Group by grade: count, median, Q1, Q3, P5, P95, the share of square images, and the
    Spearman correlation of each feature with grade.
  - **Commit the JSON.** It is a few KB and contains no images, so the Insights page also
    works on Render.
- [ ] 5.2 Insights gets a new **"Dataset features — what the data tells us"** section with
  4 cards, each holding a Chart.js chart and a short "What it tells us" paragraph:
  1. **Class balance.** Keep the existing chart and add the imbalance ratio (Grade 0 :
     Grade 3 = 1805 : 193 ≈ 9.4 : 1). The story: the majority class dominates, which is why
     class weights were used and why Grade 3/4 recall is the hard part.
  2. **Image resolution and shape by grade**: a bar of median megapixels plus a bar of the
     share of square images. In the probe (80 per grade), Grade 0 had a median of
     **1.1 MP and 46% square images**, against **4.2–5.1 MP and at most 6% square** for
     Grades 1–4. The story: an **acquisition bias** (different clinics or cameras). The
     model can learn this shortcut, which fits Grade 0 being predicted at about 100%
     confidence while Grades 1–4 are poorly separated.
  3. **Brightness and contrast by grade**: the median as a point plus an IQR floating bar
     (Chart.js `[q1, q3]` bar data). In the probe, Grade 0 images were brighter (≈95 vs
     83–87) and higher-contrast (≈18.6 vs 14–16). The story: also tied to how images were
     captured, which motivates normalisation such as the notebook's Ben Graham experiment.
  4. **Colour channels by grade**: grouped R, G and B means. In the probe the green channel
     was lower in DR grades (≈82 vs 67–73). The story: green carries the vessel and lesion
     contrast that graders rely on.

  The probe numbers above come from a sample. **Write the final text from the
  full-dataset JSON.** Put each interpretation in the template, not the JSON, so it is easy
  to edit.
- [ ] 5.3 Hide the "Seed buffer" and "Your uploads" series when they are empty. On Render
  they currently draw as rows of zeros.
- [ ] 5.4 `/api/insights` returns `dataset_features`. Test that it has at least 3 features
  and that the JSON file exists.

**Done when:** Insights shows 4 feature charts with interpretations, both locally and on
Render.
**Commit:** `feat: dataset feature visualisations with interpretations on Insights`

---

## Phase 6 — Bulk upload (multiple images, or a ZIP sorted by grade)  (P0, ~4 h) — G9

**Status:** ☐ not started

- [ ] 6.1 Add a helper `_save_image(data, filename, grade, source, db)` in
  `app/routes/upload.py`. The existing single-file `/api/upload` stays and calls it.
- [ ] 6.2 `POST /api/upload/bulk`:
  - Inputs: `files: list[UploadFile]`, `grade_label: int`, and
    `destination: "train" | "eval"` (default `"train"`, which maps to `source = "user"` or
    `"eval"`).
  - Limits: at most 100 files per request, 10 MB each, checked by magic bytes.
  - Returns `{saved: [...], rejected: [{filename, reason}], counts_by_grade}`.
- [ ] 6.3 `POST /api/upload/zip`:
  - Takes one `.zip` with folders `0/` to `4/` (also accept `grade_0/`); the folder name
    gives the label.
  - Safety limits: at most 500 entries and 200 MB uncompressed; skip folders, hidden files
    and `__MACOSX`.
  - Use only the base name of each entry, so `../x` cannot escape the uploads folder, and
    check magic bytes per file.
  - Returns the same response shape as 6.2.
- [ ] 6.4 Make `destination=eval` mean "Evaluation set" (`source="eval"`, never trained on).
  This lets a cloud deployment without APTOS images still evaluate models in production
  (Phase 7).
- [ ] 6.5 Rework `upload.html`:
  - Two tabs: **Images** (`<input multiple>`, drag and drop several files, a thumbnail
    grid, one grade selector) and **ZIP by grade folders**.
  - A destination choice: Training pool or Evaluation set.
  - A progress bar, a per-file result list showing saved or rejected and why, and a counts
    table with separate train, user and eval columns.
- [ ] 6.6 Tests, using synthetic PNGs generated in `conftest.py` so they run on a fresh
  clone:
  - Bulk upload of 5 images saves 5 rows.
  - A mix of valid and invalid files saves the valid ones and reports the rejects.
  - A ZIP with grade folders gets the labels right.
  - A ZIP entry named `../evil.png` is stored safely inside the uploads folder.
  - An oversized file is rejected.
  - `destination=eval` sets `source="eval"`.

**Done when:** you can upload 10 images in one go from the UI, and a ZIP of 15 images in
5 grade folders saves 15 correctly labelled rows.
**Commit:** `feat: bulk image and ZIP upload into the training or evaluation pool`

---

## Phase 7 — Evaluate retrained models on held-out data before promoting  (evaluation P0, gate P1, ~5 h) — G10, G11

**Status:** ☐ not started

- [ ] 7.1 Split the retraining pool into train and validation with **stratification**:
  `train_test_split(stratify=y)` when every class has at least 2 images, otherwise a random
  split. Correct the misleading "Stratified-ish" comment.
- [ ] 7.2 After training, score **both the new candidate and the current active model** on
  the evaluation set (all `source="eval"` rows; locally the 550 held-out images), using
  `src.model.evaluate_model` in batches. If there is no evaluation set, score on the
  internal validation split and label the result
  `eval_source = "validation split (no held-out set)"`.
- [ ] 7.3 Promotion rule: a pure function `should_promote(candidate, current, cfg) -> (bool, reason)`.
  - Defaults from D3, which can be overridden with `PROMOTION_QWK_TOLERANCE` and
    `PROMOTION_RECALL_TOLERANCE`.
  - A rejected candidate is **still saved** as a version with `promoted: false`, and can
    be activated by hand from Insights. Rename "Rollback" to "Activate" for versions that
    aren't active.
- [ ] 7.4 Add these fields to each registry entry: `eval_source`, `eval_n`,
  `eval_candidate {accuracy, qwk, weighted_f1, auc, grade34_recall}`, `eval_previous {…}`,
  `promoted`, `decision_reason`, `train_counts_by_grade`, and `train_accuracy` (renamed from
  `final_accuracy`; still read the old key for existing entries).
- [ ] 7.5 Add a `stage` field to the retraining status and show it on the Retrain page:
  `loading pool → preprocessing N images → training epoch k/8 → evaluating on N held-out
  images → decision`. The result card shows a **side-by-side table (current vs candidate,
  with coloured differences)** and the decision with its reason.
- [ ] 7.6 Insights history columns: Version | Type | Trained on | Held-out QWK |
  Held-out accuracy | Decision | Created | Action. Show training accuracy only in a tooltip.
- [ ] 7.7 (P2) Retrain "when the need arises":
  - Show a banner on Upload and Retrain: "N new labelled images since the last retrain —
    retraining recommended" once `RETRAIN_TRIGGER_THRESHOLD` is reached (default 20).
  - Optional `AUTO_RETRAIN=true`, off by default, starts retraining automatically when an
    upload crosses the threshold.
- [ ] 7.8 Tests:
  - Unit tests for `should_promote` covering promote, reject on QWK, and reject on recall.
  - A `@pytest.mark.slow` end-to-end retrain on a tiny synthetic pool for 1 epoch. Check
    that the registry entry has the evaluation fields and that **`exp5_base.keras` has the
    same SHA-256 before and after**. This extends the base-integrity claim from registry
    updates to real retrains.

**Done when:**
- A local retrain shows held-out metrics for both models plus the decision, and Insights
  lists the held-out QWK per version.
- Setting a negative tolerance shows the rejection path.

Retraining now takes about 1 extra minute for evaluation; the stage display covers it in
the UI.
**Commit:** `feat: evaluate retrained models on held-out data and gate promotion`

---

## Phase 8 — Run the full pipeline in the cloud  (guard P0, full deploy P1, ~3 h) — G12

**Status:** ☐ not started

- [ ] 8.1 **Guard (do this regardless).** Add a `RETRAIN_ENABLED` setting (default `true`).
  - In `render.yaml`, set it to `false`. The Retrain page then shows "Retraining is disabled
    on this free 512 MB instance — the full pipeline runs at <cloud URL> / locally via
    Docker", and the API returns 503 with the same message.
  - This stops a grader's click from crashing the instance by running out of memory.
  - Also fix the stale comment in `render.yaml` ("Primary deployment is Hugging Face Spaces").
- [ ] 8.2 **Hugging Face Spaces, Docker SDK.** The free CPU tier is about 2 vCPU and
  16 GB RAM; check the current limits before relying on them.
  - Create the Space and add a `scripts/deploy_hf.sh` that pushes to the Space remote with
    a README carrying the front matter `sdk: docker` and `app_port: 8000`. Keep that README
    in `deploy/hf/README.md` so the GitHub README stays clean.
  - `exp5_base.keras` is 15 MB, so it needs Git LFS or Xet on the Space (HF requires it
    for files over 10 MB).
  - HF runs containers as uid 1000: use `useradd -u 1000 appuser` in the Dockerfile.
  - The cloud training and evaluation pools start empty, because no APTOS images are
    shipped. The cloud demo is: upload a ZIP for training and a ZIP for evaluation
    (Phase 6), then retrain. `MIN_SAMPLES=5` still applies.
  - Storage there is ephemeral; document this.
- [ ] 8.3 If HF doesn't work out, keep the guard plus local Docker retraining, which is
  already documented. Optionally, pay for a Render instance with at least 2 GB.
- [ ] 8.4 Update the README links: the Render URL (prediction and insights), the HF URL
  (full pipeline), and which features each one supports.

**Done when:** on the cloud URL, uploading a ZIP, retraining, seeing the evaluation and
then predicting with the new version all work; Render shows the disabled message instead
of crashing; prediction on both URLs works.
**Commit:** `deploy: guard retraining on low-memory hosts; add Hugging Face Spaces deployment`

---

## Phase 9 — Notebook updates  (P1, ~3 h) — G13

**Status:** ☐ not started

- [ ] 9.1 Make the notebook runnable in this repo: cell 8's folder-name fix (from 1.6),
  `data/raw` via the symlink (0.4), and dependencies from `requirements-dev.txt`.
- [ ] 9.2 New `scripts/run_notebook_cells.py`. It uses `nbclient` to run a **chosen list of
  cells** in order and saves outputs **only** for those cells, leaving every other output
  untouched.
  - Setup cells to run: imports, config, the helper functions, data loading and the split
    cells. Confirm the exact indices when implementing.
  - Skip: cell 5 (pip), cell 11 (it raises if `kaggle.json` is missing even when the data
    is present), cell 14 (download), and **all training cells (about 54–64)**.
  - Target cells: **37** (sample images per class), **38** (Ben Graham comparison), **45**
    (pipeline stages), **48** (augmentation examples).
- [ ] 9.3 Add **Section 14 — "Deployed model: test and prediction functions"**:
  - Add the repo root to `sys.path` and import `src.prediction` and `src.model`, so the
    notebook uses **the same code as production**.
  - Load `../models/exp5_base.keras` and evaluate it on the same test split. Print
    accuracy, QWK, AUC, weighted F1 and Grade 3/4 recall (expected: 0.618 / 0.588 / 0.839 /
    0.625 / 0.390) and plot the confusion matrix.
  - Run `predict_image()` on held-out images `cd314653a4d8` (grade 0) and `e1fb532f55df`
    (grade 3), showing the image, the predicted grade, the label and the confidence.
  - A markdown note: these are the figures the app reports. They differ from the Exp 5 row
    printed in Section 10 because that row came from a different training session than the
    saved model file.
  - Run only this section's cells with the same runner.
- [ ] 9.4 A note at the top of the notebook linking the app and explaining where the
  deployed model comes from.

**Done when:**
- The 4 visual cells and Section 14 have outputs and nothing errors.
- The training-cell outputs are unchanged, and Section 14's numbers equal
  `baseline_metrics.json`.

**Commit:** `docs(notebook): run preprocessing visuals and add deployed-model evaluation section`

---

## Phase 10 — Docs, cleanup, video, submission  (P0, ~4 h + recording) — G14

**Status:** ☐ not started

- [ ] 10.1 Bring the README up to date:
  - **Links**: Render, Hugging Face, video, notebook.
  - **Features**: bulk upload, the evaluation gate, uptime and status, dataset insights,
    load testing.
  - **Architecture**: the diagram now includes `src/`, nginx and the replicas.
  - **Setup**: the venv; **downloading the data**
    (`kaggle competitions download -c aptos2019-blindness-detection` after accepting the
    rules); `scripts/prepare_data.py`; `docker compose up`; scaling.
  - **Model and evaluation**: the measured metrics and links.
  - **Retraining and promotion gate.**
  - **Dataset insights**: one paragraph of the story.
  - **Flood Request Simulation (Locust) — Results.**
  - **Deployment**, **Project structure** (laid out like the brief's tree), **Tests**.
  - **Limitations**: the resolution and acquisition bias, weak Grades 2 and 4, ephemeral
    storage, and replicas versus retraining.
- [ ] 10.2 Cleanup:
  - `.gitignore`: `data/raw`, `data/train/*` and `data/test/*` (except the README and
    `manifest.csv`), `loadtest/images/`, `backup/`, `VIDEO_GUIDE.md`.
  - `.dockerignore`: `loadtest/`, `notebook/`, `data/raw`, `data/train/*`, `data/test/*`,
    `backup/`.
  - Remove any remaining stale comments.
- [ ] 10.3 Update the untracked `VIDEO_GUIDE.md` for the new flow and held-out demo images.
- [ ] 10.4 Re-record the video (D5). Camera on throughout, 5–7 minutes:
  1. Intro, then the live URL with the uptime pill.
  2. About 30 s on the Insights feature story.
  3. Predict held-out Grade 0 and Grade 3 images and say the true labels out loud.
  4. Bulk ZIP upload.
  5. Retrain: show the live epochs, the held-out comparison and the decision.
  6. Predict again and show the new version.
  7. Activate or roll back.
  8. A quick look at the Locust results.

  Upload it to YouTube (unlisted is fine) and update the README link.
- [ ] 10.5 Fresh-clone check: `git clone` into a temporary folder, then
  `docker compose up --build`, `pytest`, and click through every page.
- [ ] 10.6 Merge `fixes` into `main` and push.
- [ ] 10.7 Submit. **Attempt 1:** a zip of the GitHub repo (GitHub "Download ZIP" of
  `main`, or `git archive --format=zip -o dr-detection-mlops.zip HEAD`). Check it contains
  `models/exp5_base.keras` and `notebook/`. **Attempt 2:** the GitHub URL. Each attempt
  has a fixed purpose, so **don't submit the zip until everything is final**. Aim to
  finish by Thursday afternoon.

**Commit:** `docs: final README with load-test results, setup and corrected metrics`

---

## Schedule

Revised on 2026-11-24, when Phase 0 started; the original plan assumed a Saturday start.

| Day | Phases | Notes |
|-----|--------|-------|
| Tue 24 Nov | 0 ✔, 1, 2, 3 | Run `evaluate_model.py` early; everything downstream relies on its numbers |
| Wed 25 Nov | 4, 5, 6 | Run the Locust matrix (about 25 min) while writing Phase 5 |
| Thu 26 Nov | 7 (minimal), 8.1, 9 (Section 14 only), 10, video, submit | Submit by about 18:00 to leave a buffer before 23:59 |

**If you fall behind, the minimum cut is:** Phases 0–6, **8.1**, a minimal Phase 7
(7.2 + 7.5: show the held-out evaluation for both models, no gate), Section 14 only from
Phase 9, and Phase 10. Drop 1.9, 7.3/7.7, 8.2 and 9.2 first.

---

## Final checklist (before attempt 1)

**Rubric**
- [ ] Video: camera on throughout; prediction **and** retraining both clearly shown.
- [ ] Retraining: bulk upload saved to the database, preprocessing of the uploaded data,
  and fine-tuning from `exp5` as the starting model, with the script and model file in the
  repo.
- [ ] Prediction: a data point goes in and the **correct** class comes out (held-out demo
  images), with the script (`src/prediction.py`) and the model file present.
- [ ] Evaluation: the notebook shows preprocessing, optimisation (EarlyStopping,
  ReduceLROnPlateau, Dropout, BatchNorm, class weights, grid search) and at least 4
  metrics, with no error cells.
- [ ] Deployment: web UI, public URL plus Docker, and dataset insights.

**Brief**
- [ ] Model uptime is visible in the UI.
- [ ] Data visualisations: at least 3 features, each with a written interpretation.
- [ ] Train and retrain are reachable from the UI; the retrain trigger button works.
- [ ] Bulk upload of multiple images.
- [ ] Locust results with different numbers of containers are in the README.
- [ ] Production evaluation of retrained models is demonstrated.
- [ ] The repo layout has `README.md`, `notebook/`, `src/{preprocessing,model,prediction}.py`,
  `data/{train,test}/` and `models/`.
- [ ] The README has the video link, URLs, description, setup steps and flood results.
- [ ] Cloud deployment works (and retraining there, or is clearly guarded).

---

## Appendix — reference measurements (audit, 2026-11-21)

Use these to confirm that the implementation reproduces reality.

- **Deployed model** `models/exp5_base.keras` (byte-identical to the original repo's
  `models/deep_learning/exp5_deep_cnn.keras`), scored on the notebook's test split
  (seed 42, 550 images) with the app's preprocessing:
  - accuracy **0.6182**, QWK **0.5879**, AUC **0.8386**, weighted F1 **0.6252**
  - class-3 recall 0.621, class-4 recall 0.159 → `grade34_recall` **0.390**
  - These match the original repo's `outputs/results_table.csv.run1.bak` exactly. The model
    is not the run-2 model: only 0.755 of its top predictions agree with run-2 `exp5.npz`.
- **Confusion matrix** (rows are true grades 0–4, columns predicted grades):
  `[242,2,0,27,0] [13,14,4,25,0] [26,13,59,40,12] [3,2,3,18,3] [5,5,16,11,7]`. The model
  over-predicts Grade 3.
- **Retrained v2** (local, now backed up) on the same split: accuracy 0.6255, QWK 0.5897.
  This result is contaminated, because 34 of the 200 seed images were from the test split.
- **Seed buffer composition:** 135 train / 31 val / 34 test.
- **Inference speed:** about 145 ms per image on CPU (`model.predict` and `model(x)` are
  about the same).
- **Live Render:** `/health` is ok, the cold start is about 57 s, and the seed and upload
  pools are empty.
- **This machine:** 4 CPUs, 15 GB RAM with about 5 GB free.
- **Feature probe** (80 images per grade, medians, Grades 0 → 4):
  - megapixels 1.10 / 4.19 / 5.07 / 5.07 / 5.07
  - share of square images 46% / 4% / 6% / 0% / 1%
  - brightness 94.6 / 82.7 / 84.5 / 84.4 / 87.4
  - contrast 18.6 / 14.3 / 15.2 / 16.4 / 14.9
  - green channel 82.1 / 66.9 / 69.4 / 70.7 / 72.8
  - Spearman correlation with grade: width +0.44, megapixels +0.44, aspect ratio +0.39,
    contrast −0.13. Sharpness and fundus area are flat.
