"""Retraining routes: the /retrain page, trigger, and status endpoint.

model.fit() takes minutes, so retraining runs in a background thread and the
frontend polls /api/retrain/status. A module-level status dict holds live
progress; a lock prevents two concurrent retrains.
"""
import threading

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import APP_DIR
from app.database.db import SessionLocal
from app.services import retrain_service

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

_lock = threading.Lock()
_status = {
    "status": "idle",        # idle | running | complete | error
    "current_epoch": 0,
    "total_epochs": 0,
    "logs": [],
    "result": None,
    "error": None,
}


def _reset_status():
    _status.update({"status": "running", "current_epoch": 0, "total_epochs": 0,
                    "logs": [], "result": None, "error": None})


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


@router.get("/retrain", response_class=HTMLResponse)
async def retrain_page(request: Request):
    return templates.TemplateResponse("retrain.html", {"request": request})


@router.post("/api/retrain")
async def api_retrain():
    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retrain is already running.")
    _reset_status()
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started"}


@router.get("/api/retrain/status")
async def api_retrain_status():
    return _status
