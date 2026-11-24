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

**Status:** ☑ done 2026-11-24

The brief requires this layout. The pure ML code now lives in `src/`. `app/` keeps only
the web layer, the database and the registry, and imports from `src/`. Nothing in `src/`
imports from `app/`.

- [x] 1.1 `src/__init__.py` and `src/config.py`:
  - `TARGET_SIZE`, `NUM_CLASSES`, `CLASS_LABELS`, `CLINICAL_NOTES`, the split constants
    (`SPLIT_SEED=42`, `VAL_SIZE=0.15`, `TEST_SIZE=0.15`), `FREEZE_CUTOFF=30`, `REPO_ROOT`
    and `DEFAULT_MODEL_PATH`.
  - `app/config.py` re-exports these and keeps only app settings. It also gains
    `TRAIN_DIR` and `TEST_DIR`.
- [x] 1.2 `src/preprocessing.py`:
  - `preprocess_image()` and `preprocess_batch()`, moved verbatim.
    `app/services/preprocess.py` is deleted.
  - `load_labels()`, `stratified_split()` (the notebook's two-stage split, giving
    2562 / 550 / 550, which a test checks) and `split_of_ids()`.
- [x] 1.3 `src/model.py`:
  - `load_model()` and `freeze_backbone()`. It raises if nothing is left to train, and a
    test checks that layer 30 is the global average pooling layer.
  - `build_finetune_model()` and `fine_tune()`, which adds early stopping with the best
    weights restored.
  - `compute_metrics(y_true, probs)` is pure and testable. It returns `n`, accuracy, QWK,
    AUC (`None` if a grade is missing), weighted and macro F1, `grade34_recall` (the mean
    of the class-3 and class-4 recall), per-class recall and the confusion matrix. NaN is
    turned into `None`, so results are always valid JSON.
  - `evaluate_model(model, paths_or_array, y)` is `predict_proba` plus `compute_metrics`,
    loading images in batches.
- [x] 1.4 `src/prediction.py`:
  - `predict_proba()` (paths loaded in batches, or an array), `format_prediction(probs)`
    and `predict_image()`.
  - Command line: `python -m src.prediction <images…> [--model …] [--json]`.
- [x] 1.5 App rewired:
  - `model_service` decodes the image **outside** the lock and runs only inference under
    it, then uses `format_prediction`.
  - `retrain_service` uses `build_finetune_model` and `fine_tune`.
  - The stale "Phase N", "Part A" and `IMPLEMENTATION_PLAN.md` references are removed
    from `app/`.
- [x] 1.6 `notebooks/` renamed to `notebook/` with `git mv`.
  - The cell-8 path check was patched at the byte level: the file uses CRLF line endings,
    so the git diff is a single line.
  - README links and project structure, `.dockerignore` and `.gitignore` updated.
- [x] 1.7 `data/train/` and `data/test/` created, each with a `README.md` and `.gitkeep`.
  Their images are ignored by git (except `data/test/manifest.csv`) and Docker.
- [x] 1.8 `tests/test_src.py` adds 14 tests: preprocessing (shape, BGR→RGB, bad file),
  the split, prediction (result keys, paths vs array, CLI), metrics (perfect, the
  `grade34_recall` definition, missing grades), and the freeze.
  - `conftest.py` gains the `synthetic_image` and `base_model` fixtures.
  - **Also fixed:** a bare `pytest` (the README's command) used to fail with
    `No module named 'app'`. The new `pytest.ini` sets `pythonpath = .` and registers the
    `slow` marker.
- [ ] 1.9 Skipped (D4 = no `.h5` export).

**Done when** (all checked 2026-11-24):
- `pytest` → 25 passed.
- `python -m src.prediction demo_images/grade0_1.png` → grade 0 (0.98). Held-out
  `cd314653a4d8` → grade 0 (1.00) and `e1fb532f55df` → grade 3 (0.84), the same as the audit.
- uvicorn: every page and `/docs` return 200, `/api/predict` works and rejects non-images
  with 400.
- One real retrain through the API completed (early stopping at epoch 5, created v4, the
  model was hot-swapped), and `exp5_base.keras` is unchanged.
- `docker compose build` succeeds.

**Commit:** `refactor: move ML core into src/ (preprocessing, model, prediction) to match required layout`

---

## Phase 2 — Correct metrics, data separation and held-out demo images  (P0, ~3 h) — G2, G3, G4

**Status:** ☑ done 2026-11-24

- [x] 2.1 `scripts/evaluate_model.py` writes `models/baseline_metrics.json` (committed).
  - Contents: the metrics, confusion matrix, split description, model SHA-256 and a
    timestamp.
  - Reproduced exactly: accuracy 0.6182, QWK 0.5879, AUC 0.8386, weighted F1 0.6252,
    `grade34_recall` 0.3899, n = 550. It takes about 2 min on CPU.
- [x] 2.2 `app/config.py`: `load_baseline_metrics()` reads the JSON (no more hard-coded
  numbers).
  - Insights shows 5 metric cards (QWK, accuracy, AUC, weighted F1, Grade 3+4 recall)
    with the note "Measured on 550 held-out test images … by
    `scripts/evaluate_model.py`", and a fallback message if the file is missing.
- [x] 2.3 README:
  - Old numbers replaced by a measured-performance table linking the JSON, with a
    sentence explaining the gap with the notebook's printed Exp 5 row.
  - The limitation now reads Grade 3+4 recall 0.390 (Proliferative 0.159).
  - New sections: "Training pool vs. held-out evaluation set" and "Getting the data" (the
    Kaggle steps and the three scripts).
  - The install step now uses `requirements-dev.txt`.
- [x] 2.4 `scripts/seed_data.py` renamed to `scripts/prepare_data.py` with `git mv`, and
  rewritten:
  - Replay buffer: 40 per grade from **training-split ids only**, copied to `data/train/`
    with rows `source="seed"` and `image_path="train/<id>.png"`.
  - Evaluation set: all 550 test-split images in `data/test/` with rows
    `source="eval"` and `image_path="test/<id>.png"`, plus `data/test/manifest.csv`
    (LF line endings, committed).
  - Deviation from the plan: the evaluation set is built **by default** (`--no-eval`
    skips it) instead of behind an `--eval` flag.
  - `--force` rebuilds either pool and removes the legacy `data/uploads/seed_*` files. It
    only removes *prepared* eval rows (`test/…`), so later user-uploaded eval images
    survive.
  - The script ends with an overlap check between the seed ids and the test split, and
    aborts if any id overlaps.
  - Deviation from the plan: copying the test samples moved to `pick_demo_images`, because
    choosing samples the model grades correctly needs the model.
- [x] 2.5 New `app/services/pool.py`:
  - `training_rows()` (seed + user, **never eval**), `eval_rows()`, `counts_by_grade()`,
    and `resolve_image_path()`.
  - Path rules: a path relative to `data/` resolves under `DATA_DIR`; a bare filename
    under `UPLOADS_DIR`; foreign absolute and Windows paths fall back to their file name
    in uploads.
  - `build_dataset()` in the retrain service uses it. Upload and Insights use
    `counts_by_grade`.
  - Insights adds a "Held-out eval set" stat, and its distribution chart hides empty pools.
- [x] 2.6 `tests/test_data_split.py`: the manifest has 550 rows with the right grade counts
  and equals the notebook's test split; seed picks come only from the training split;
  `training_rows` never returns eval rows; and the path-resolution rules hold.
- [x] 2.7 Local state reset:
  - The registry, `exp5_current`, the versions, `app.db` and `data/uploads/*` were deleted.
    Their pre-fixes copies are in `backup/pre-fixes/`; the v4 from the Phase 1 test
    retrain was discarded.
  - `prepare_data` then ran: 200 seed rows and 550 eval rows. The app re-initialises at
    `v1_base`.
- [x] 2.8 `scripts/pick_demo_images.py` scores the 550 held-out images and writes the
  top-3 correct per grade to `demo_images/heldout_grade{g}_{id}.png`, plus the top-1 per
  grade to `tests/sample_images/grade_{g}.png`. The old training-split images are in
  `backup/pre-fixes/`. Results (correct / total, best confidence):

  | Grade | Correct | Top picks (confidence) |
  |---|---|---|
  | 0 | 242/271 | `e3cd96cb094c`, `e03a74e7d74f`, `d4583e9525dc` (all 1.00) |
  | 1 | 14/56 | `05a5183c92d0` (0.69), `cc9270f06b65` (0.63), `531b39880c32` (0.57) |
  | 2 | 59/150 | `ba2624883599` (0.36) — weak, avoid in the video |
  | 3 | 18/29 | `e1fb532f55df` (0.84), `b019a49787c1` (0.84), `4360a112db10` (0.83) |
  | 4 | 7/44 | `d51b3fe0fa1b` (0.33) — weak, avoid in the video |

**Done when** (checked): `baseline_metrics.json` matches 2.1, `pytest` → 30 passed,
`build_dataset` sees 200 images (160 train / 40 val, 0 skipped, no eval rows), and the app
at `v1_base` serves the measured metrics and predicts the held-out demo images correctly.
**Commit:** `fix: report measured held-out metrics for the deployed model; separate train and eval data`

---

## Phase 3 — Stop inference blocking the server; add uptime and monitoring  (P0, ~3 h) — G5, G6

**Status:** ☑ done 2026-11-24

- [x] 3.1 `api_predict`, `api_upload` and `api_rollback` are now plain `def`, so they run
  in FastAPI's thread pool. The model lock stays: one inference at a time per process,
  which is what Phase 4 scales out with containers.
  - Extra: uploads are decoded **in memory** with the new
    `src.preprocessing.preprocess_image_bytes`, so there is no temporary file. A test
    checks the result is identical to reading the file.
- [x] 3.2 `app/utils/validation.py` provides `read_image_upload()`, shared by predict and
  upload. It checks the content type and magic bytes, and enforces a 10 MB limit by
  reading at most limit + 1 bytes.
- [x] 3.3 `model_service` tags the cached model with `_loaded_version`. Before each
  inference, if the registry's active version differs (another replica retrained or rolled
  back), it reloads under the lock. Predictions report the version that actually served
  them.
- [x] 3.4 `app/services/monitor.py`:
  - Start time (set in `lifespan`), counts of requests, 5xx errors and predictions, and the
    last 1000 prediction latencies (server-side, including any wait for the lock).
  - `snapshot()` adds mean, p50, p95 and max latency, and `format_duration()`.
  - The middleware in `main.py` records every request and adds `X-Served-By: <hostname>`
    and `X-Process-Time-ms`.
- [x] 3.5 New `app/routes/status.py`:
  - `/health` (moved from `main.py`) adds `uptime_seconds`, `started_at` and `hostname`.
  - New `/api/status` returns the full snapshot plus `loaded_version` and `last_retrained`.
- [x] 3.6 UI:
  - **A status pill in the header of every page** shows "● Online · up 3m 12s · v1_base".
    It polls `/health` every 15 s, ticks every second in between, and turns red "Offline"
    when unreachable.
  - **A "Model status" card at the top of Insights** shows status, up-time, serving
    version, predictions served, mean and p95 latency, requests served, server errors,
    last retrained, the container name and the start time. It refreshes every 10 s.
  - The inline JavaScript passes `node --check`.
- [x] 3.7 `tests/test_status.py` (8 tests): uptime, status fields, the `X-Served-By`
  header, the prediction counter and latency, the 10 MB limit, reloading on a version
  change, `format_duration`, and that every page has the pill.
  - Also fixed: the page routes used Starlette's deprecated
    `TemplateResponse(name, {"request": …})` signature, which produced 6 warnings once
    pages were rendered in tests. They now use `TemplateResponse(request, name, …)`.

**Done when** (measured with 20 parallel `/api/predict` calls on the grade-3 demo image):

| | `/health` latency during the burst | 20 predictions finished in |
|---|---|---|
| Before (`async def`) | 290–1,520 ms | 6.3 s |
| After (`def` + in-memory decode) | **1.5–5 ms** | **4.7 s** |

`pytest` → 40 passed. The pill and the status card render; `/api/status` reports the
counters.
**Commit:** `feat: add uptime and monitoring status; serve inference off the event loop`

---

## Phase 4 — Locust flood test across multiple containers  (P0, ~5 h incl. runs) — G7

**Status:** ☑ done 2026-11-24

- [x] 4.1 `docker-compose.yml` reworked:
  - `web` has no host port (`expose: 8000`, so `--scale` works), `cpus: ${WEB_CPUS:-1.0}`,
    `mem_limit: ${WEB_MEM:-2g}`, and the TF, OMP and `PREDICT_CONCURRENCY` settings. The
    healthcheck interval is now 10 s.
  - `lb` runs `nginx:1.27-alpine` on `${APP_PORT:-8000}:80` and waits for `web` to be healthy.
  - `model_service` applies the TF thread settings with `tf.config.threading` before the
    model loads.
  - Safety when several replicas start at once: base → current is copied to a temp file
    and renamed into place, the registry temp file is per-process, and `init_db` retries
    when two replicas create the table at the same moment.
- [x] 4.2 `loadtest/nginx.conf`: round-robin `upstream web:8000`,
  `client_max_body_size 20m`, 600 s timeouts. After scaling, run `docker compose restart lb`.
- [x] 4.3 `loadtest/locustfile.py`:
  - Weighted tasks: predict ×10 (random held-out image, fails on a non-200 status or a
    missing `predicted_grade`) and `/health` ×1. Waits 0.5–1.5 s between requests.
  - Counts responses per `X-Served-By` and saves them as `<prefix>_served_by.json`.
  - `LOADTEST_MAX_KB` filters images by size, for remote hosts.
- [x] 4.4 `loadtest/prepare_images.py` (Python rather than shell): 20 held-out images,
  4 per grade, mean 2.4 MB, gitignored.
- [x] 4.5 `loadtest/run_matrix.sh`: settings via env (`APP_PORT`, `REPLICAS`, `USERS`,
  `DURATION`, `OUT`, `STATS_AFTER`); writes `environment.txt`, warms up each replica, and
  takes a `docker stats` snapshot 60 s into each run.
- [x] 4.6 `loadtest/summarize.py [dir]` writes `summary.md` and
  `latency_vs_containers.png`.
  - The chart has two panels (throughput, p95 latency; one axis each) and a one-hue
    ordinal blue ramp for 10/25/50 users that passed the palette validator. Lines are
    direct-labelled only when clear of every neighbour, otherwise the legend carries them.
  - The headline is computed from the data.
- [x] 4.7 Cloud: Render (main-branch code, 0.1 CPU, 512 MB), 5 users, 1 min, images ≤ 1.1 MB.
  Result: 35 predictions at 0.6/s, median 5.9 s, p95 13 s, 1 failure (HTTP 502 from
  Render's proxy), 58 s cold start. Saved in `loadtest/results/cloud_render/`.
- [x] 4.8 Committed: the per-run stats, failures, `served_by`, `docker_stats`,
  `environment.txt`, `summary.md` and the chart. Gitignored with recursive `**`
  patterns: history, exceptions, logs, HTML and `loadtest/images/`.
- [x] 4.9 README: a **"Flood Request Simulation (Locust) — Results"** section (setup,
  chart, table, findings, the caveat, what the test caught, the cloud row, how to
  reproduce). The Docker setup section explains scaling and `APP_PORT`.
- [x] 4.10 The "retrain with a single replica" caveat is in the README and the
  docker-compose header.

**Found by the test and fixed (not in the original plan): admission control.**
- The first matrix (kept in `loadtest/results/run1_no_admission_control/`) showed one
  overloaded container getting *slower*: 1.31 → 0.98 → 0.74 predictions/s at 10/25/50
  users, with memory rising from 0.8 to 1.4 GB.
- Cause: about 40 request threads decoded 2.4 MB images at the same time and starved the
  inference.
- Fix: `model_service` now holds a `BoundedSemaphore(PREDICT_CONCURRENCY=2)` around
  decode + predict, and a test checks the cap. With it, one container holds 1.15–1.31/s
  under overload and the p95 at 50 users drops from 74 s to 43 s.

**Final results** (4-CPU laptop, 1 CPU per container, 2 min per run, 0 failures across
3,996 predictions, an even split across containers):

| Containers | 10 users | 25 users | 50 users | p95 at 50 users |
|---|---|---|---|---|
| 1 | 1.31/s | 1.15/s | 1.29/s | 43 s |
| 2 | 4.21/s | 4.38/s | 4.55/s | 14 s |
| 3 | 5.18/s | 5.71/s | 5.79/s | 10 s |

- Open caveat: a lone container runs at 1.15–1.5/s, while each of two runs at about 2.2/s.
  A repeat run ruled out run order, but the cause was not isolated. The README says so.
- Port 8000 on this machine is used by another process, so the runs used `APP_PORT=8080`.

**Commit:** `feat: add Locust load tests with nginx-balanced replicas and results`

---

## Phase 5 — Dataset insights: at least 3 features, each with an interpretation  (P0, ~4 h) — G8

**Status:** ☑ done 2026-11-24

- [x] 5.1 `scripts/compute_dataset_insights.py` measures all 3,662 images with
  `Pool(4)` in about 4 min.
  - Features: size (width, height, megapixels, aspect, square) and, inside the fundus
    mask (grey > 10): brightness, contrast, mean R, G and B, plus sharpness (Laplacian
    variance at 512 px) and fundus fraction.
  - Per grade: n, mean, median, Q1, Q3, P5 and P95, plus the share of square images and
    each feature's Spearman ρ with grade.
  - Output: `app/static/data/dataset_insights.json` (11 KB, committed, so it works on
    Render).
- [x] 5.2 Insights has a new **"Dataset features — what the data tells us"** section with a
  story paragraph and 4 cards, each with a "What it tells us" interpretation written from
  the full-dataset numbers:
  1. **Class balance**: the existing pool chart restyled. 49% No DR, 9.4 : 1 against
     Severe, grades 3 + 4 only 13%. Ties to class weights and the 0.39 Grade 3+4 recall.
  2. **Resolution and shape**: megapixels (middle-50% band + median) and the % square.
     Grade 0 is 1.1 MP and 50% square, grades 1–4 are 4.2–5.1 MP and 1–5% square,
     ρ = +0.61 (the strongest link). Interpreted as an acquisition difference, i.e. a
     possible shortcut.
  3. **Brightness and contrast**: two range charts. Grade 0 is 96 vs 83–87 for
     brightness and 18.0 vs 14.1–16.2 for contrast (ρ −0.31). The bands overlap heavily,
     so this is capture conditions, which motivates illumination normalisation.
  4. **Green channel**: a range chart plus a median R/G/B table. 82 vs 66–72, but it does
     not keep falling with severity, so it separates "some DR vs none", not severity.
     Blue shows no trend.
  - Charts:
    - Colours: the validated dark-mode palette slots, checked against the real card
      surface `#16283f` (all checks pass). Each pool keeps its own colour.
    - Marks: bars ≤ 24 px with 4 px rounded ends, median dots with a surface ring, and
      two-line grade labels that are never slanted.
    - Hover tooltips with image counts, and a legend on every multi-series chart.
  - A collapsible **data table** lists every feature's median by grade plus ρ (the
    text view of the charts).
  - Also: the metric cards are renamed "Deployed Model Performance" and laid out in one
    row of 5.
- [x] 5.3 Empty pools were already hidden in Phase 2; the chart colours now follow the
  pool, not its position.
- [x] 5.4 `/api/insights` returns `dataset_features`. New `tests/test_insights.py` (3
  tests) checks the JSON has ≥ 3 features with 5 grades each, that Q1 ≤ median ≤ Q3 and
  the counts sum to 3,662, that the API serves it, and that the page has the 6 canvases,
  4 interpretations and no unfilled placeholders.
- [x] Also: a README **"Dataset insights"** section (summary table plus the story),
  ahead of Phase 10.

**Done when** (checked): `pytest` → 44 passed; the page JavaScript passes `node --check`;
headless-Chrome screenshots show every chart rendering, plus the up-time pill and the
status card.
**Commit:** `feat: dataset feature visualisations with interpretations on Insights`

---

## Phase 6 — Bulk upload (multiple images, or a ZIP sorted by grade)  (P0, ~4 h) — G9

**Status:** ☑ done 2026-11-24

- [x] 6.1 Shared helpers in `app/routes/upload.py`:
  - `_save_image(db, data, filename, grade, source)` stores the file as
    `<random8>_<safe name>` and adds the row.
  - `_summary()` returns `saved`, `rejected`, `ignored`, the counts for all three pools,
    and the older `counts_by_grade` and `total_user_uploads` fields.
  - `_safe_name` now also strips Windows paths.
  - `app/utils/validation.py` gains a non-raising `image_problem(data)` that returns
    the reason a file is rejected.
- [x] 6.2 `POST /api/upload/bulk` takes `files[]`, `grade_label` and
  `destination=train|eval`, accepts up to 100 files at 10 MB each, and reports invalid
  files per file instead of failing the whole request.
- [x] 6.3 `POST /api/upload/zip`:
  - The grade comes from the first `0`–`4` or `grade_N` folder in each path, at any depth.
  - Limits: 500 entries and 200 MB unpacked, both checked before reading.
  - `__MACOSX/` and dot-files are counted as `ignored`; entries outside a grade folder,
    non-images, oversized, encrypted or corrupt files are rejected with a reason.
  - Nothing is extracted by its ZIP path, so `0/../../evil.png` is saved as
    `<random>_evil.png` inside uploads.
- [x] 6.4 `destination=eval` saves rows with `source="eval"`, which never enter training.
  The single-image `/api/upload` also accepts `destination`.
- [x] 6.5 `upload.html` rewritten:
  - **Images** and **ZIP sorted by grade** tabs, plus a Training pool / Evaluation set
    choice.
  - Drop zones that take several files, with up to 24 thumbnails and a "+N" counter.
  - Uploads go through XHR to show a **progress bar**, followed by "Checking and saving
    images…".
  - The result lists how many images were saved per grade and every rejected file with
    its reason.
  - A live **pool table**: replay buffer, your training uploads and the evaluation set,
    per grade.
  - nginx's `client_max_body_size` is raised to 250 MB for bulk and ZIP uploads.
- [x] 6.6 `tests/test_bulk_upload.py` (11 tests, synthetic PNGs and in-memory ZIPs):
  - Bulk upload of 5 files; a mixed batch keeps the good file and rejects the others with
    their reasons; `destination=eval`.
  - Bad grade or destination returns 400; more than 100 files returns 400.
  - ZIP grades come from folders at any depth (`grade_3/`, `batch/4/`), with rejects and
    ignored files counted.
  - A path-traversal entry stays inside uploads; a non-ZIP returns 400.
  - Single upload with a destination, and the page markup.
- [x] README feature list updated.

**Done when** (checked):
- `pytest` → 55 passed, and the page JavaScript passes `node --check`.
- **Browser end-to-end** (headless Chrome over the DevTools protocol, against a separate
  server with a throwaway database and uploads folder): selected 3 canvas-made PNGs plus
  one fake file, chose grade 3 and the evaluation set, and submitted.
  - The page showed "Saved 3 images to the evaluation set (3 × Severe DR)" and
    "notes.png — not a JPEG, PNG or BMP image".
  - The pool table updated live (grade-3 evaluation set 0 → 3), the form reset, and the
    ZIP tab toggled.
  - The real `data/app.db` was untouched.

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
