"""Prediction routes: the /predict page and the /api/predict endpoint.

``api_predict`` is a plain ``def`` so FastAPI runs it in its thread pool: CPU
inference then never blocks the event loop, and /health and page loads stay
responsive while predictions are queued.
"""
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_DIR, GRADE_COLORS
from app.services import model_service
from app.utils.validation import read_image_upload

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@router.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(request, "predict.html")


@router.post("/api/predict")
def api_predict(file: UploadFile = File(...)):
    data = read_image_upload(file)
    try:
        result = model_service.predict_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result["grade_color"] = GRADE_COLORS[result["predicted_grade"]]
    return result
