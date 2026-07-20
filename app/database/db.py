"""SQLAlchemy engine, session factory, and FastAPI dependency.

`check_same_thread=False` is required because the retraining thread (Phase 5)
touches the database from outside the request thread. SQLite is otherwise
single-writer, which is fine for this app's load.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def init_db() -> None:
    """Create all tables. Idempotent — safe to call on every startup."""
    # Import models so they register on Base.metadata before create_all.
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
