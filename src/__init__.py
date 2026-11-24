"""Core ML code for diabetic retinopathy grading.

- ``src.preprocessing``: image preprocessing and the dataset split.
- ``src.model``: loading, fine-tuning and evaluating the Keras model.
- ``src.prediction``: single-image and batch prediction (also a CLI).

The web app in ``app/`` imports from here. Nothing in ``src/`` imports from
``app/``, so these modules can be used on their own (scripts, the notebook).
"""
