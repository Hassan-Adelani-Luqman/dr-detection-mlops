"""Model version registry.

The registry is a JSON file tracking every model version, which one is active,
and retrain history. The original weights (`exp5_base.keras`) are NEVER
overwritten; on first run we copy them to `exp5_current.keras` and record the
base as version ``v1_base``.

Writes go through a temp file + ``os.replace`` so a crash mid-write cannot leave
a half-written (corrupt) registry on disk.
"""
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Optional

from app.config import (
    BASE_MODEL_PATH,
    BASE_TRAINING_SAMPLES,
    CURRENT_MODEL_PATH,
    REGISTRY_PATH,
)

BASE_VERSION = "v1_base"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_registry() -> dict:
    """Ensure the current model and registry exist. Idempotent.

    - Copies base -> current if current is missing.
    - Creates the registry JSON with the base version active if missing.
    Returns the registry dict.
    """
    if not CURRENT_MODEL_PATH.exists():
        if not BASE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Base model not found at {BASE_MODEL_PATH}. Cannot initialise."
            )
        # copyfile (not copy2): avoid EPERM from metadata copy on bind mounts.
        # Copy to a temp name then rename, so another replica starting at the
        # same time can never load a half-written model file.
        tmp = CURRENT_MODEL_PATH.with_suffix(f".{os.getpid()}.tmp")
        shutil.copyfile(BASE_MODEL_PATH, tmp)
        os.replace(tmp, CURRENT_MODEL_PATH)

    if not REGISTRY_PATH.exists():
        registry = {
            "active_version": BASE_VERSION,
            "retrain_count": 0,
            "last_retrained": None,
            "history": [
                {
                    "version": BASE_VERSION,
                    "type": "base",
                    "samples_trained_on": BASE_TRAINING_SAMPLES,
                    "created_at": _now_iso(),
                }
            ],
        }
        _write_registry(registry)
        return registry

    return get_registry()


def _write_registry(registry: dict) -> None:
    """Atomically write the registry JSON (temp file + os.replace)."""
    tmp = REGISTRY_PATH.with_suffix(f".{os.getpid()}.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def get_registry() -> dict:
    """Read and return the registry, initialising it if absent."""
    if not REGISTRY_PATH.exists():
        return init_registry()
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def install_current_model(source) -> None:
    """Make ``source`` the served model file, atomically (temp copy + rename).

    Other replicas may load the current model at any moment, so it must never
    be visible half-written.
    """
    tmp = CURRENT_MODEL_PATH.with_suffix(f".{os.getpid()}.tmp")
    shutil.copyfile(source, tmp)          # copyfile: no metadata (bind-mount safe)
    os.replace(tmp, CURRENT_MODEL_PATH)


def register_version(entry: dict, promote: bool) -> dict:
    """Append a retrained version to the history; make it active only if ``promote``.

    Every retrain counts (rejected candidates are kept and can be activated later).
    """
    registry = get_registry()
    registry["history"].append(entry)
    if entry.get("type") == "retrained":
        registry["retrain_count"] = registry.get("retrain_count", 0) + 1
        registry["last_retrained"] = entry.get("created_at", _now_iso())
    if promote:
        registry["active_version"] = entry["version"]
    _write_registry(registry)
    return registry


def update_registry(new_entry: dict, active_version: str) -> dict:
    """Append a history entry, promote ``active_version``, bump retrain count.

    ``new_entry`` should already contain at least ``version`` and ``type``.
    Only entries of type "retrained" increment the retrain count / timestamp.
    """
    registry = get_registry()
    registry["history"].append(new_entry)
    registry["active_version"] = active_version
    if new_entry.get("type") == "retrained":
        registry["retrain_count"] = registry.get("retrain_count", 0) + 1
        registry["last_retrained"] = new_entry.get("created_at", _now_iso())
    _write_registry(registry)
    return registry


def set_active_version(version: str) -> dict:
    """Promote an existing version to active (used by rollback)."""
    registry = get_registry()
    known = {h["version"] for h in registry["history"]}
    if version not in known:
        raise ValueError(f"Unknown version: {version}")
    registry["active_version"] = version
    _write_registry(registry)
    return registry


def get_active_version() -> str:
    return get_registry()["active_version"]


def get_history() -> list:
    return get_registry()["history"]


def find_version(version: str) -> Optional[dict]:
    """Return the history entry for ``version``, or None."""
    for entry in get_registry()["history"]:
        if entry["version"] == version:
            return entry
    return None
