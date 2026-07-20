"""ORM models."""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from app.database.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UploadedImage(Base):
    """A labelled fundus image in the training pool.

    `source` distinguishes replay-buffer seed images ("seed") from images a
    user uploaded through the app ("user"). Retraining trains on both; the
    Insights page reports only user uploads.
    """

    __tablename__ = "uploaded_images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    grade_label = Column(Integer, nullable=False)          # 0–4
    source = Column(String, nullable=False, default="user")  # "user" | "seed"
    uploaded_at = Column(DateTime, default=_utcnow)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"<UploadedImage id={self.id} grade={self.grade_label} "
                f"source={self.source} file={self.filename!r}>")
