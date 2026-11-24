# data/train — replay buffer

Labelled fundus images used **for retraining**, alongside user uploads
(`data/uploads/`). They form a balanced *replay buffer* (40 per grade by
default) so that fine-tuning on a few new uploads does not make the model
forget the grades it already knows.

- Drawn **only from the training split** of APTOS 2019 (the notebook's
  stratified split, seed 42), so none of them overlap the held-out test set in
  `data/test/`.
- Each image has a row in the database with `source="seed"`.
- Images are **not committed**: APTOS 2019 is Kaggle competition data and
  cannot be redistributed. Recreate them from your own copy of the dataset
  (see the main README, "Local setup").
