"""Insights routes: dataset distribution, model metrics, and version history.

Satisfies the "data insights based on the dataset" part of the deployment
criterion. The distribution chart contrasts the training pool with the original
APTOS class balance, exposing the imbalance the model was trained on.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import APP_DIR, APTOS_DISTRIBUTION, CLASS_LABELS, load_baseline_metrics
from app.database.db import get_db
from app.services.pool import counts_by_grade
from app.services.registry import get_registry

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@router.get("/insights", response_class=HTMLResponse)
async def insights_page(request: Request):
    return templates.TemplateResponse(request, "insights.html")


@router.get("/api/insights")
async def api_insights(db: Session = Depends(get_db)):
    registry = get_registry()
    grades = sorted(CLASS_LABELS)
    user_counts = counts_by_grade(db, "user")
    seed_counts = counts_by_grade(db, "seed")
    eval_counts = counts_by_grade(db, "eval")

    return {
        "class_labels": [CLASS_LABELS[g] for g in grades],
        "distributions": {
            "user_uploads": [user_counts[g] for g in grades],
            "seed_buffer": [seed_counts[g] for g in grades],
            "eval_set": [eval_counts[g] for g in grades],
            "aptos_original": [APTOS_DISTRIBUTION[g] for g in grades],
        },
        "totals": {
            "user_uploads": sum(user_counts.values()),
            "seed_buffer": sum(seed_counts.values()),
            "eval_set": sum(eval_counts.values()),
            "aptos_original": sum(APTOS_DISTRIBUTION.values()),
        },
        "baseline_metrics": load_baseline_metrics(),
        "model": {
            "active_version": registry["active_version"],
            "retrain_count": registry["retrain_count"],
            "last_retrained": registry["last_retrained"],
        },
        "history": registry["history"],
    }
