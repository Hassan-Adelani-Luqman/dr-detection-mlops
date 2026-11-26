"""Locust flood test for the prediction API.

Each simulated user uploads a held-out fundus image to /api/predict (weight
10) and occasionally checks /health (weight 1), waiting 0.5-1.5 s between
requests. Images come from loadtest/images/ (run `python loadtest/prepare_images.py`).

The app tags every response with the container that served it
(X-Served-By); at the end of the run the split across containers is printed
and saved next to Locust's CSV output, which shows the load balancer at work.

    locust -f loadtest/locustfile.py --host http://localhost:8000 \\
           --headless -u 25 -r 5 -t 2m --csv loadtest/results/r1_u25

Set LOADTEST_MAX_KB to upload only images up to that size (useful against a
remote host on a slow uplink).
"""
import json
import os
import random
from collections import Counter
from pathlib import Path

from locust import HttpUser, between, events, task

IMAGE_DIR = Path(__file__).resolve().parent / "images"
MAX_KB = int(os.getenv("LOADTEST_MAX_KB", "0"))

# Loaded once and shared by all simulated users.
IMAGES = [(p.name, p.read_bytes()) for p in sorted(IMAGE_DIR.glob("*.png"))
          if not MAX_KB or p.stat().st_size <= MAX_KB * 1024]
if not IMAGES:
    raise SystemExit(f"No images in {IMAGE_DIR} (run `python loadtest/prepare_images.py`).")

served_by = Counter()


class PredictUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        with self.client.post("/api/predict", files={"file": (name, data, "image/png")},
                              name="/api/predict", catch_response=True) as resp:
            served_by[resp.headers.get("X-Served-By", "unknown")] += 1
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif "predicted_grade" not in resp.json():
                resp.failure("response has no predicted_grade")
            else:
                resp.success()

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


@events.test_stop.add_listener
def report_served_by(environment, **_kwargs):
    total = sum(served_by.values()) or 1
    print("\nPredictions per container (X-Served-By):")
    for host, n in served_by.most_common():
        print(f"  {host:<20} {n:>6}  ({100 * n / total:.0f}%)")
    prefix = getattr(environment.parsed_options, "csv_prefix", None)
    if prefix:
        Path(f"{prefix}_served_by.json").write_text(json.dumps(dict(served_by), indent=2) + "\n")
