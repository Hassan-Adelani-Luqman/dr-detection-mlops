"""Health and status endpoints: model up-time and serving metrics.

- ``GET /health``: small and fast; used by Docker/Render health checks and the
  status pill in the page header.
- ``GET /api/status``: everything the Insights "Model status" card shows.
"""
from fastapi import APIRouter

from app.services import model_service, monitor
from app.services.registry import get_registry

router = APIRouter()


@router.get("/health")
async def health():
    reg = get_registry()
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "active_version": reg["active_version"],
        "retrain_count": reg["retrain_count"],
        "uptime_seconds": round(monitor.uptime_seconds(), 1),
        "started_at": monitor.started_at_iso(),
        "hostname": monitor.HOSTNAME,
    }


@router.get("/api/status")
async def status():
    reg = get_registry()
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "active_version": reg["active_version"],
        "loaded_version": model_service.loaded_version(),
        "retrain_count": reg["retrain_count"],
        "last_retrained": reg["last_retrained"],
        **monitor.snapshot(),
    }
