"""Shared pytest fixtures.

The database and uploads directory are redirected to temporary locations
(before any app module is imported) so tests never touch real data. The model
and registry use the real models directory: exp5_base.keras is read-only, and
the integrity test relies on hashing the genuine base file.
"""
import os
import tempfile
from pathlib import Path

import pytest

# --- Redirect DB + uploads to temp locations BEFORE importing app modules ---
_TMP = Path(tempfile.mkdtemp(prefix="drmlops_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["UPLOADS_DIR"] = str(_TMP / "uploads")
(_TMP / "uploads").mkdir(parents=True, exist_ok=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "tests" / "sample_images"


def _sample_images() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p for p in SAMPLES_DIR.glob("grade_*.*")
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


requires_samples = pytest.mark.skipif(
    len(_sample_images()) == 0,
    reason="No sample images in tests/sample_images/ "
           "(run scripts.seed_data to populate).",
)


@pytest.fixture(scope="session")
def client():
    """A TestClient with the app lifespan run (db, registry, model ready)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_images() -> list[Path]:
    return _sample_images()


@pytest.fixture()
def grade0_image() -> Path:
    imgs = [p for p in _sample_images() if p.stem == "grade_0"]
    if not imgs:
        pytest.skip("grade_0 sample image not available.")
    return imgs[0]
