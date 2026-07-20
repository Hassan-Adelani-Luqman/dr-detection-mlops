"""FastAPI application entry point.

Startup (via lifespan) initialises the database, the model registry, and loads
the current model once. Routers are mounted for each feature; templates and
static assets are wired up here.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import APP_DIR
from app.database.db import init_db
from app.services import model_service
from app.services.registry import get_active_version, get_registry, init_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dr-mlops")

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: initialising database, registry, and model...")
    init_db()
    init_registry()
    model_service.load_current_model()
    logger.info("Startup complete. Active version: %s", get_active_version())
    yield
    logger.info("Shutdown.")


app = FastAPI(title="DR Detection MLOps", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# Routers
from app.routes import predict as predict_routes  # noqa: E402

app.include_router(predict_routes.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "active_version": get_active_version()},
    )


@app.get("/health")
async def health():
    reg = get_registry()
    return {
        "status": "ok",
        "model_loaded": model_service.is_loaded(),
        "active_version": reg["active_version"],
        "retrain_count": reg["retrain_count"],
    }
