"""Retraining routes: the /retrain page, the trigger, status and readiness.

model.fit() takes minutes, so retraining runs in a background thread and the
frontend polls /api/retrain/status. A module-level status dict holds live
progress; a lock prevents two concurrent retrains.

Retraining is triggered by the button on the Retrain page. The page also shows
when retraining is *recommended*: once RETRAIN_TRIGGER_THRESHOLD new labelled
images have been uploaded since the last retrain. With AUTO_RETRAIN=true, an
upload that crosses the threshold starts the retrain itself.
"""
import threading
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import (APP_DIR, AUTO_RETRAIN, PROMOTION_QWK_TOLERANCE,
                        PROMOTION_RECALL_TOLERANCE, RETRAIN_ENABLED,
                        RETRAIN_TRIGGER_THRESHOLD)
from app.database.db import SessionLocal
from app.database.models import UploadedImage
from app.services import retrain_service
from app.services.pool import eval_rows, training_rows
from app.services.registry import get_registry

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

RETRAIN_DISABLED_MESSAGE = (
    "Retraining is turned off on this deployment (RETRAIN_ENABLED=false), usually "
    "because the host does not have the memory TensorFlow training needs. Prediction, "
    "upload and insights still work; retrain on a larger host or locally with Docker.")

_lock = threading.Lock()
_status = {
    "status": "idle",        # idle | running | complete | error
    "stage": None,           # loading | preprocessing | training | evaluating | saving | done
    "stage_text": None,
    "trigger": None,         # manual | automatic
    "current_epoch": 0,
    "total_epochs": 0,
    "logs": [],
    "result": None,
    "error": None,
}


def _run():
    """Background worker. Owns its own DB session and always releases the lock."""
    db = SessionLocal()
    try:
        result = retrain_service.run_retraining(db, status=_status)
        _status["result"] = result
        _status["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        _status["status"] = "error"
        _status["error"] = str(exc)
    finally:
        db.close()
        if _lock.locked():
            _lock.release()


def start_retraining(trigger: str = "manual") -> bool:
    """Start a retrain in the background; False if disabled or one is already running."""
    if not RETRAIN_ENABLED or not _lock.acquire(blocking=False):
        return False
    _status.update({"status": "running", "stage": "starting", "stage_text": "Starting…",
                    "trigger": trigger, "current_epoch": 0, "total_epochs": 0,
                    "logs": [], "result": None, "error": None})
    threading.Thread(target=_run, daemon=True).start()
    return True


def readiness(db: Session) -> dict:
    """How big the pools are, and whether enough new data has arrived to retrain."""
    last = get_registry().get("last_retrained")
    new_uploads = db.query(UploadedImage).filter(UploadedImage.source == "user")
    if last:
        new_uploads = new_uploads.filter(UploadedImage.uploaded_at > datetime.fromisoformat(last))
    n_new = new_uploads.count()
    return {
        "training_pool": len(training_rows(db)),
        "eval_set": len(eval_rows(db)),
        "new_since_last_retrain": n_new,
        "last_retrained": last,
        "threshold": RETRAIN_TRIGGER_THRESHOLD,
        "recommended": n_new >= RETRAIN_TRIGGER_THRESHOLD,
        "auto_retrain": AUTO_RETRAIN and RETRAIN_ENABLED,
        "retrain_enabled": RETRAIN_ENABLED,
        "disabled_message": None if RETRAIN_ENABLED else RETRAIN_DISABLED_MESSAGE,
        "running": _status["status"] == "running",
        "promotion_rule": {"qwk_tolerance": PROMOTION_QWK_TOLERANCE,
                           "recall_tolerance": PROMOTION_RECALL_TOLERANCE},
    }


def maybe_auto_retrain(db: Session) -> bool:
    """With AUTO_RETRAIN on, start a retrain once the trigger threshold is reached."""
    if AUTO_RETRAIN and readiness(db)["recommended"]:
        return start_retraining(trigger="automatic")
    return False


@router.get("/retrain", response_class=HTMLResponse)
async def retrain_page(request: Request):
    return templates.TemplateResponse(request, "retrain.html")


@router.post("/api/retrain")
async def api_retrain():
    if not RETRAIN_ENABLED:
        raise HTTPException(status_code=503, detail=RETRAIN_DISABLED_MESSAGE)
    if not start_retraining():
        raise HTTPException(status_code=409, detail="A retrain is already running.")
    return {"status": "started"}


@router.get("/api/retrain/status")
async def api_retrain_status():
    return _status


@router.get("/api/retrain/readiness")
def api_retrain_readiness():
    db = SessionLocal()
    try:
        return readiness(db)
    finally:
        db.close()
