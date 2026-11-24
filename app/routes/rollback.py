"""Rollback route: promote a previous model version back to active.

Any versioned snapshot in models/versions/ can be restored, as can the original
base (v1_base), which lives at models/exp5_base.keras rather than in versions/.
The version string is validated against the registry and checked for path
traversal before any file operation.
"""
from fastapi import APIRouter, HTTPException

from app.config import BASE_MODEL_PATH, VERSIONS_DIR
from app.services import model_service
from app.services.registry import (BASE_VERSION, find_version, install_current_model,
                                   set_active_version)

router = APIRouter()


def _version_source_path(version: str):
    """Resolve the on-disk model file for a known version, safely."""
    if version == BASE_VERSION:
        return BASE_MODEL_PATH
    # Reject anything that could escape the versions directory.
    if "/" in version or "\\" in version or ".." in version:
        raise HTTPException(status_code=400, detail="Invalid version identifier.")
    path = (VERSIONS_DIR / f"{version}.keras").resolve()
    if VERSIONS_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid version path.")
    return path


@router.post("/api/rollback/{version}")
def api_rollback(version: str):
    # Must be a version the registry knows about.
    if find_version(version) is None:
        raise HTTPException(status_code=404, detail=f"Unknown version: {version}")

    source = _version_source_path(version)
    if not source.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file for {version} is not available on disk.",
        )

    install_current_model(source)          # atomic: other replicas never see a partial file
    set_active_version(version)
    model_service.reload_model()

    return {"status": "ok", "active_version": version,
            "message": f"Rolled back to {version}."}
