"""Insights routes: dataset distribution, model metrics, and version history.

Satisfies the "data insights based on the dataset" part of the deployment
criterion. The distribution chart contrasts the training pool with the original
APTOS class balance, exposing the imbalance the model was trained on.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import (APP_DIR, APTOS_DISTRIBUTION, BASELINE_METRICS,
                        CLASS_LABELS)
from app.database.db import get_db
from app.database.models import UploadedImage
from app.services.registry import get_registry

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _distribution(db: Session, source: str) -> dict:
    rows = (db.query(UploadedImage.grade_label, func.count(UploadedImage.id))
              .filter(UploadedImage.source == source)
              .group_by(UploadedImage.grade_label).all())
    counts = {g: 0 for g in CLASS_LABELS}
    for grade, n in rows:
        counts[grade] = n
    return counts


@router.get("/insights", response_class=HTMLResponse)
async def insights_page(request: Request):
    return templates.TemplateResponse("insights.html", {"request": request})


@router.get("/api/insights")
async def api_insights(db: Session = Depends(get_db)):
    registry = get_registry()
    user_counts = _distribution(db, "user")
    seed_counts = _distribution(db, "seed")

    labels = [CLASS_LABELS[g] for g in sorted(CLASS_LABELS)]
    return {
        "class_labels": labels,
        "distributions": {
            "user_uploads": [user_counts[g] for g in sorted(CLASS_LABELS)],
            "seed_buffer": [seed_counts[g] for g in sorted(CLASS_LABELS)],
            "aptos_original": [APTOS_DISTRIBUTION[g] for g in sorted(CLASS_LABELS)],
        },
        "totals": {
            "user_uploads": sum(user_counts.values()),
            "seed_buffer": sum(seed_counts.values()),
            "aptos_original": sum(APTOS_DISTRIBUTION.values()),
        },
        "baseline_metrics": BASELINE_METRICS,
        "model": {
            "active_version": registry["active_version"],
            "retrain_count": registry["retrain_count"],
            "last_retrained": registry["last_retrained"],
        },
        "history": registry["history"],
    }
