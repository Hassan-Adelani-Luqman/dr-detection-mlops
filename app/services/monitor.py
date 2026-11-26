"""In-process monitoring: uptime, request counters, prediction latency.

Every container (replica) keeps its own numbers, and reports its hostname (the
container id under Docker), so with several replicas behind a load balancer
you can see which one answered and how the load was spread.

All counters are in memory and reset when the process restarts.
"""
import socket
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

HOSTNAME = socket.gethostname()
PREDICT_PATH = "/api/predict"

_lock = threading.Lock()
_started_at = time.time()
_requests = 0
_errors = 0                     # responses with status >= 500
_predictions = 0                # successful /api/predict responses
_latencies_ms = deque(maxlen=1000)   # server-side time of recent predictions


def mark_started() -> None:
    """Record the start time. Called once from the app's lifespan."""
    global _started_at
    _started_at = time.time()


def uptime_seconds() -> float:
    return time.time() - _started_at


def started_at_iso() -> str:
    return datetime.fromtimestamp(_started_at, timezone.utc).isoformat(timespec="seconds")


def record_request(path: str, status: int, elapsed_ms: float) -> None:
    """Count one finished request (called by the HTTP middleware)."""
    global _requests, _errors, _predictions
    with _lock:
        _requests += 1
        if status >= 500:
            _errors += 1
        if path == PREDICT_PATH and status == 200:
            _predictions += 1
            _latencies_ms.append(elapsed_ms)


def format_duration(seconds: float) -> str:
    """``3725`` -> ``"1h 2m"``; ``65`` -> ``"1m 5s"``."""
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _latency_summary() -> dict:
    with _lock:
        values = np.array(_latencies_ms, dtype=float)
    if values.size == 0:
        return {"window": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "window": int(values.size),
        "mean": round(float(values.mean()), 1),
        "p50": round(float(np.percentile(values, 50)), 1),
        "p95": round(float(np.percentile(values, 95)), 1),
        "max": round(float(values.max()), 1),
    }


def snapshot() -> dict:
    """Counters and uptime for this process."""
    up = uptime_seconds()
    with _lock:
        counts = {"requests_served": _requests, "server_errors": _errors,
                  "predictions_served": _predictions}
    return {
        "hostname": HOSTNAME,
        "started_at": started_at_iso(),
        "uptime_seconds": round(up, 1),
        "uptime": format_duration(up),
        **counts,
        "prediction_latency_ms": _latency_summary(),
    }
