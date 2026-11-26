"""Insights routes: dataset features, model metrics, and version history.

Satisfies the "data insights based on the dataset" part of the deployment
criterion and the brief's "interpret at least 3 features" requirement: the
page charts class balance, image resolution and shape, brightness and contrast,
and the green channel by grade (precomputed by
scripts/compute_dataset_insights.py), each with a written interpretation.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import (APP_DIR, APTOS_DISTRIBUTION, CLASS_LABELS,
                        load_baseline_metrics, load_dataset_insights)
from app.database.db import get_db
from app.services.pool import counts_by_grade
from app.services.registry import get_registry

router = APIRouter()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _history(registry: dict, baseline: dict | None) -> list:
    """Version history with each version's held-out metrics and decision.

    Retrained versions carry the metrics measured when they were trained; the
    base version uses the committed baseline measurement on the same 550 images.
    """
    rows = []
    for entry in registry["history"]:
        h = dict(entry)
        h.setdefault("train_accuracy", h.get("final_accuracy"))   # older entries
        if h.get("eval_candidate"):
            c = h["eval_candidate"]
            h["heldout"] = {"qwk": c["qwk"], "accuracy": c["accuracy"],
                            "n": h.get("eval_n"), "source": h.get("eval_source")}
        elif h.get("type") == "base" and baseline:
            h["heldout"] = {"qwk": baseline["qwk"], "accuracy": baseline["accuracy"],
                            "n": baseline["n"], "source": "scripts/evaluate_model.py"}
        else:
            h["heldout"] = None
        rows.append(h)
    return rows


@router.get("/insights", response_class=HTMLResponse)
async def insights_page(request: Request):
    return templates.TemplateResponse(request, "insights.html")


@router.get("/api/insights")
async def api_insights(db: Session = Depends(get_db)):
    registry = get_registry()
    baseline = load_baseline_metrics()
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
        "baseline_metrics": baseline,
        "dataset_features": load_dataset_insights(),
        "model": {
            "active_version": registry["active_version"],
            "retrain_count": registry["retrain_count"],
            "last_retrained": registry["last_retrained"],
        },
        "history": _history(registry, baseline),
    }
