# Python 3.12 matches the local venv (TensorFlow 2.18 supports 3.9–3.12).
FROM python:3.12-slim

# System deps for opencv-python-headless (libGL/glib are needed even headless).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Keep TF quiet and Python unbuffered for clean container logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application (the .dockerignore excludes runtime data, tests, venv).
COPY . .

# Run as a non-root user; give it ownership so it can write models/ and data/
# (registry, current model, uploaded images, retrained versions).
RUN useradd --create-home appuser \
    && mkdir -p /app/data/uploads /app/models/versions \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Shell form so ${PORT} is expanded (Render/HF may inject a different port).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
