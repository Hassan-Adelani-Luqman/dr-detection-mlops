"""FastAPI application entry point.

Startup (via lifespan) initialises the database, the model registry, and loads
the current model once. Routers are mounted for each feature; templates and
static assets are wired up here. A middleware times every request for the
monitoring counters and tags responses with the serving container.
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import APP_DIR
from app.database.db import init_db
from app.services import model_service, monitor
from app.services.registry import get_active_version, init_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dr-mlops")

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: initialising database, registry, and model...")
    init_db()
    init_registry()
    model_service.load_current_model()
    monitor.mark_started()
    logger.info("Startup complete on %s. Active version: %s",
                monitor.HOSTNAME, get_active_version())
    yield
    logger.info("Shutdown.")


app = FastAPI(title="DR Detection MLOps", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Time each request, count it, and say which container served it."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        monitor.record_request(request.url.path, status, elapsed_ms)
    response.headers["X-Served-By"] = monitor.HOSTNAME
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
    return response


# Routers
from app.routes import predict as predict_routes  # noqa: E402
from app.routes import upload as upload_routes  # noqa: E402
from app.routes import retrain as retrain_routes  # noqa: E402
from app.routes import rollback as rollback_routes  # noqa: E402
from app.routes import insights as insights_routes  # noqa: E402
from app.routes import status as status_routes  # noqa: E402

app.include_router(predict_routes.router)
app.include_router(upload_routes.router)
app.include_router(retrain_routes.router)
app.include_router(rollback_routes.router)
app.include_router(insights_routes.router)
app.include_router(status_routes.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"active_version": get_active_version()},
    )

