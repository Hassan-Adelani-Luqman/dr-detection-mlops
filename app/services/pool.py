"""The image pool: which database rows are training data, which are held out.

Every labelled image has a row in ``uploaded_images`` with a ``source``:

- ``"seed"``: replay-buffer images from the APTOS training split (``data/train/``)
- ``"user"``: images uploaded through the app (``data/uploads/``)
- ``"eval"``: the held-out evaluation set (``data/test/``). **Never trained on.**

Retraining must read rows only through :func:`training_rows`, so evaluation
images can never leak into training.
"""
import re
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import CLASS_LABELS, DATA_DIR, UPLOADS_DIR
from app.database.models import UploadedImage

TRAINING_SOURCES = ("seed", "user")
EVAL_SOURCE = "eval"
SOURCES = (*TRAINING_SOURCES, EVAL_SOURCE)


def resolve_image_path(stored: str) -> Path:
    """Turn the ``image_path`` stored in the database into a file on disk.

    - Paths relative to ``data/`` (``train/x.png``, ``test/x.png``) resolve
      under ``DATA_DIR``; this is what ``scripts/prepare_data.py`` stores.
    - A bare filename (user uploads) resolves under ``UPLOADS_DIR``.
    - Older rows may hold an absolute path from another machine (for example a
      Windows path); those fall back to their file name under ``UPLOADS_DIR``.
    """
    normalised = stored.replace("\\", "/")
    path = Path(normalised)
    is_windows_abs = re.match(r"^[A-Za-z]:/", normalised) is not None
    if path.is_absolute() or is_windows_abs:
        return path if path.exists() else UPLOADS_DIR / path.name
    if "/" in normalised:
        return DATA_DIR / normalised
    return UPLOADS_DIR / normalised


def training_rows(db: Session) -> list:
    """All rows retraining may use: seed and user images, never eval."""
    return (db.query(UploadedImage)
              .filter(UploadedImage.source.in_(TRAINING_SOURCES))
              .all())


def eval_rows(db: Session) -> list:
    """The held-out evaluation set."""
    return db.query(UploadedImage).filter(UploadedImage.source == EVAL_SOURCE).all()


def counts_by_grade(db: Session, source: str) -> dict:
    """``{grade: count}`` for one source, with every grade present."""
    rows = (db.query(UploadedImage.grade_label, func.count(UploadedImage.id))
              .filter(UploadedImage.source == source)
              .group_by(UploadedImage.grade_label).all())
    counts = dict.fromkeys(CLASS_LABELS, 0)
    for grade, n in rows:
        counts[grade] = n
    return counts
