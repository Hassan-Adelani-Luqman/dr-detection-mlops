"""Upload routes: the /upload page and the /api/upload endpoint.

Uploaded images are saved to the uploads directory and recorded in the database
with source="user". These rows form the training pool used by retraining
(alongside the seed replay buffer).
"""
import re
import uuid
from pathlib import Path

from fastapi import (APIRouter, File, Form, HTTPException, Request, UploadFile,
                     Depends)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import APP_DIR, CLASS_LABELS, UPLOADS_DIR
from app.database.db import get_db
from app.database.models import UploadedImage

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"BM")


def _safe_name(filename: str) -> str:
    """Strip any path components and keep a conservative basename."""
    base = Path(filename or "image").name          # drop directory parts
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)    # allowlist chars
    return base[:80] or "image"


def _counts_by_grade(db: Session) -> dict:
    rows = (db.query(UploadedImage.grade_label, func.count(UploadedImage.id))
              .filter(UploadedImage.source == "user")
              .group_by(UploadedImage.grade_label).all())
    counts = {g: 0 for g in CLASS_LABELS}
    for grade, n in rows:
        counts[grade] = n
    return counts


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "labels": CLASS_LABELS,
         "counts": _counts_by_grade(db)},
    )


@router.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    grade_label: int = Form(...),
    db: Session = Depends(get_db),
):
    if grade_label not in CLASS_LABELS:
        raise HTTPException(status_code=400,
                            detail="grade_label must be an integer 0–4.")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 10 MB limit.")
    if not any(data.startswith(sig) for sig in _IMAGE_MAGIC):
        raise HTTPException(status_code=400,
                            detail="File is not a valid image (JPEG/PNG/BMP).")

    safe = _safe_name(file.filename)
    stored_name = f"{uuid.uuid4().hex[:8]}_{safe}"
    dest = UPLOADS_DIR / stored_name
    dest.write_bytes(data)

    row = UploadedImage(filename=safe, image_path=str(dest),
                        grade_label=grade_label, source="user")
    db.add(row)
    db.commit()
    db.refresh(row)

    counts = _counts_by_grade(db)
    total = sum(counts.values())
    return {
        "id": row.id,
        "filename": safe,
        "grade_label": grade_label,
        "grade_name": CLASS_LABELS[grade_label],
        "total_user_uploads": total,
        "counts_by_grade": {CLASS_LABELS[g]: counts[g] for g in CLASS_LABELS},
    }
