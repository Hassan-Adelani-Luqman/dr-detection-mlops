# data/test — held-out evaluation set

The 550 images of the **test split** of APTOS 2019 (the notebook's stratified
split, seed 42). The model never trains on them. They are used to:

- measure the deployed model's metrics (`scripts/evaluate_model.py`), and
- compare every retrained model with the current one before it is promoted.

Each image has a row in the database with `source="eval"`; retraining never
reads those rows. `manifest.csv` (committed) lists the `id_code` and true
`diagnosis` of every image. The images themselves are **not committed**
(APTOS licence); recreate them from your own copy of the dataset (see the main
README, "Local setup").
