"""Upload routes: the /upload page and the upload API.

Labelled images are saved to the uploads directory and recorded in the
database. The destination decides their role:

- ``train`` -> ``source="user"``: joins the training pool used by retraining
  (alongside the seed replay buffer).
- ``eval``  -> ``source="eval"``: joins the held-out evaluation set, which
  retraining never trains on and uses to measure models.

Endpoints:
- ``POST /api/upload``       one image + its grade (kept for API clients)
- ``POST /api/upload/bulk``  many images that share one grade
- ``POST /api/upload/zip``   one ZIP whose folders give the grades (0/ ... 4/)
"""
import re
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import APP_DIR, CLASS_LABELS, UPLOADS_DIR
from app.database.db import get_db
from app.database.models import UploadedImage
from app.services.pool import counts_by_grade
from app.utils.validation import MAX_IMAGE_BYTES, image_problem, read_image_upload

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

DESTINATIONS = {"train": "user", "eval": "eval"}
MAX_BULK_FILES = 100
ZIP_MAX_ENTRIES = 500
ZIP_MAX_TOTAL_BYTES = 200 * 1024 * 1024        # uncompressed
_GRADE_FOLDER = re.compile(r"^(?:grade[ _-]?)?([0-4])$", re.IGNORECASE)


def _safe_name(filename: str) -> str:
    """Strip any path components and keep a conservative basename."""
    base = PurePosixPath((filename or "image").replace("\\", "/")).name   # drop directories
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)                          # allowlist chars
    return base[:80] or "image"


def _source_for(destination: str) -> str:
    if destination not in DESTINATIONS:
        raise HTTPException(status_code=400, detail="destination must be 'train' or 'eval'.")
    return DESTINATIONS[destination]


def _check_grade(grade_label: int) -> None:
    if grade_label not in CLASS_LABELS:
        raise HTTPException(status_code=400, detail="grade_label must be an integer 0–4.")


def _save_image(db: Session, data: bytes, filename: str, grade: int,
                source: str) -> UploadedImage:
    """Write one validated image under UPLOADS_DIR and add its row (not committed).

    The stored name gets a random prefix, so uploads never overwrite each other,
    and only the file name is stored, so the pool is portable between machines.
    """
    safe = _safe_name(filename)
    stored_name = f"{uuid.uuid4().hex[:8]}_{safe}"
    (UPLOADS_DIR / stored_name).write_bytes(data)
    row = UploadedImage(filename=safe, image_path=stored_name,
                        grade_label=grade, source=source)
    db.add(row)
    return row


def _pool_counts(db: Session) -> dict:
    """Counts by grade for every pool, keyed by pool then grade name."""
    return {source: {CLASS_LABELS[g]: n for g, n in counts_by_grade(db, source).items()}
            for source in ("seed", "user", "eval")}


def _summary(db: Session, destination: str, saved: list, rejected: list,
             ignored: int = 0) -> dict:
    db.commit()
    user = counts_by_grade(db, "user")
    return {
        "destination": destination,
        "saved": [{"id": r.id, "filename": r.filename, "grade_label": r.grade_label,
                   "grade_name": CLASS_LABELS[r.grade_label]} for r in saved],
        "rejected": rejected,
        "ignored": ignored,
        "pools": _pool_counts(db),
        "total_user_uploads": sum(user.values()),
        "counts_by_grade": {CLASS_LABELS[g]: user[g] for g in CLASS_LABELS},
    }


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        request,
        "upload.html",
        {"labels": CLASS_LABELS, "pools": {s: counts_by_grade(db, s)
                                           for s in ("seed", "user", "eval")},
         "max_files": MAX_BULK_FILES,
         "zip_max_mb": ZIP_MAX_TOTAL_BYTES // (1024 * 1024)},
    )


@router.post("/api/upload")
def api_upload(
    file: UploadFile = File(...),
    grade_label: int = Form(...),
    destination: str = Form("train"),
    db: Session = Depends(get_db),
):
    """One labelled image."""
    _check_grade(grade_label)
    source = _source_for(destination)
    data = read_image_upload(file)
    row = _save_image(db, data, file.filename, grade_label, source)
    db.commit()
    db.refresh(row)

    counts = counts_by_grade(db, "user")
    return {
        "id": row.id,
        "filename": row.filename,
        "grade_label": grade_label,
        "grade_name": CLASS_LABELS[grade_label],
        "destination": destination,
        "total_user_uploads": sum(counts.values()),
        "counts_by_grade": {CLASS_LABELS[g]: counts[g] for g in CLASS_LABELS},
    }


@router.post("/api/upload/bulk")
def api_upload_bulk(
    files: list[UploadFile] = File(...),
    grade_label: int = Form(...),
    destination: str = Form("train"),
    db: Session = Depends(get_db),
):
    """Many images that share one grade. Invalid files are reported, not fatal."""
    _check_grade(grade_label)
    source = _source_for(destination)
    if len(files) > MAX_BULK_FILES:
        raise HTTPException(status_code=400,
                            detail=f"At most {MAX_BULK_FILES} files per upload.")

    saved, rejected = [], []
    for f in files:
        data = f.file.read(MAX_IMAGE_BYTES + 1)
        problem = image_problem(data)
        if problem:
            rejected.append({"filename": f.filename, "reason": problem})
            continue
        saved.append(_save_image(db, data, f.filename, grade_label, source))
    return _summary(db, destination, saved, rejected)


def _grade_from_path(name: str):
    """The grade given by the first ``0``..``4`` (or ``grade_N``) folder in a ZIP path."""
    for part in PurePosixPath(name).parts[:-1]:
        m = _GRADE_FOLDER.match(part)
        if m:
            return int(m.group(1))
    return None


@router.post("/api/upload/zip")
def api_upload_zip(
    file: UploadFile = File(...),
    destination: str = Form("train"),
    db: Session = Depends(get_db),
):
    """One ZIP with a folder per grade, e.g. ``0/a.png``, ``3/b.jpg`` or ``grade_4/c.png``.

    Nothing is extracted to disk by its ZIP path: each image is read into memory,
    checked, and saved under a fresh name, so names like ``../x.png`` are harmless.
    """
    source = _source_for(destination)
    try:
        archive = zipfile.ZipFile(file.file)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File is not a valid ZIP archive.")

    entries = [i for i in archive.infolist() if not i.is_dir()]
    if len(entries) > ZIP_MAX_ENTRIES:
        raise HTTPException(status_code=400,
                            detail=f"ZIP has {len(entries)} files; the limit is {ZIP_MAX_ENTRIES}.")
    total = sum(i.file_size for i in entries)
    if total > ZIP_MAX_TOTAL_BYTES:
        raise HTTPException(status_code=400,
                            detail=f"ZIP unpacks to {total // (1024 * 1024)} MB; the limit is "
                                   f"{ZIP_MAX_TOTAL_BYTES // (1024 * 1024)} MB.")

    saved, rejected, ignored = [], [], 0
    for info in entries:
        name = info.filename.replace("\\", "/")
        base = PurePosixPath(name).name
        if name.startswith("__MACOSX/") or base.startswith(".") or not base:
            ignored += 1                      # OS metadata files, not images
            continue
        grade = _grade_from_path(name)
        if grade is None:
            rejected.append({"filename": name, "reason": "not inside a grade folder (0/ to 4/)"})
            continue
        if info.file_size > MAX_IMAGE_BYTES:
            rejected.append({"filename": name,
                             "reason": f"larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB"})
            continue
        try:
            with archive.open(info) as fh:
                data = fh.read(MAX_IMAGE_BYTES + 1)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
            rejected.append({"filename": name, "reason": f"could not be read ({exc})"})
            continue
        problem = image_problem(data)
        if problem:
            rejected.append({"filename": name, "reason": problem})
            continue
        saved.append(_save_image(db, data, base, grade, source))
    return _summary(db, destination, saved, rejected, ignored)
