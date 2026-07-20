"""Prediction routes: the /predict page and the /api/predict endpoint."""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_DIR, GRADE_COLORS
from app.services import model_service

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Magic-byte signatures for the image types we accept.
_IMAGE_MAGIC = (
    b"\xff\xd8\xff",              # JPEG
    b"\x89PNG\r\n\x1a\n",        # PNG
    b"BM",                        # BMP
)


def _looks_like_image(head: bytes) -> bool:
    return any(head.startswith(sig) for sig in _IMAGE_MAGIC)


@router.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@router.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail="Uploaded file is not an image.")

    tmp_path = None
    try:
        suffix = Path(file.filename or "upload").suffix or ".img"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)

        head = tmp_path.read_bytes()[:16]
        if not _looks_like_image(head):
            raise HTTPException(
                status_code=400,
                detail="File does not appear to be a valid image (JPEG/PNG/BMP).",
            )

        try:
            result = model_service.predict(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        result["grade_color"] = GRADE_COLORS[result["predicted_grade"]]
        return result
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
