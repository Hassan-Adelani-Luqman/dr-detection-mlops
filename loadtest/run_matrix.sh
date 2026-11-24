#!/usr/bin/env bash
# Run the Locust flood test against 1, 2 and 3 app containers behind nginx.
#
#   ./loadtest/run_matrix.sh                       # defaults below
#   APP_PORT=8080 ./loadtest/run_matrix.sh         # if port 8000 is taken
#   REPLICAS="1 2" USERS="10" DURATION=30s ./loadtest/run_matrix.sh   # quick check
#
# Writes Locust CSVs, per-container request counts and docker stats snapshots
# to loadtest/results/, then run `python loadtest/summarize.py`.
set -uo pipefail
cd "$(dirname "$0")/.."

export APP_PORT=${APP_PORT:-8000}
REPLICAS=${REPLICAS:-"1 2 3"}
USERS=${USERS:-"10 25 50"}
DURATION=${DURATION:-2m}
SPAWN_RATE=${SPAWN_RATE:-5}
STATS_AFTER=${STATS_AFTER:-60}   # seconds into each run for the docker stats snapshot
HOST="http://localhost:${APP_PORT}"
OUT=${OUT:-loadtest/results}
LOCUST=${LOCUST:-.venv/bin/locust}
mkdir -p "$OUT"

[ -n "$(ls loadtest/images/*.png 2>/dev/null)" ] || .venv/bin/python loadtest/prepare_images.py

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(nproc) CPUs, $(free -g | awk '/Mem:/ {print $2}') GB RAM, $(uname -sr)"
  echo "docker: $(docker --version)"
  echo "per container: ${WEB_CPUS:-1.0} CPU, ${WEB_MEM:-2g} memory, TF threads ${TF_THREADS:-1}"
  echo "locust: $($LOCUST --version 2>&1 | head -1)"
  echo "users: $USERS | spawn rate: $SPAWN_RATE/s | duration: $DURATION | wait 0.5-1.5 s"
  echo "images: $(ls loadtest/images/*.png | wc -l) held-out fundus photos"
} > "$OUT/environment.txt"
cat "$OUT/environment.txt"

docker compose build web

for R in $REPLICAS; do
  echo "=== $R container(s) ==="
  docker compose up -d --scale web="$R" --wait
  docker compose restart lb >/dev/null
  for _ in $(seq 60); do curl -sf "$HOST/health" >/dev/null && break; sleep 1; done

  # Warm-up: the first prediction in each container is slow (graph tracing).
  IMG=$(ls loadtest/images/*.png | head -1)
  for _ in $(seq $((R * 4))); do curl -s -o /dev/null -F "file=@$IMG" "$HOST/api/predict"; done

  for U in $USERS; do
    PREFIX="$OUT/r${R}_u${U}"
    echo "--- $R container(s), $U users, $DURATION ---"
    "$LOCUST" -f loadtest/locustfile.py --host "$HOST" --headless \
      -u "$U" -r "$SPAWN_RATE" -t "$DURATION" --csv "$PREFIX" --only-summary \
      > "$PREFIX.log" 2>&1 &
    LPID=$!
    # Resource snapshot in the middle of the run (after users have ramped up).
    sleep "$STATS_AFTER"
    docker stats --no-stream --format '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' \
      | grep -E "web|lb" > "$PREFIX.docker_stats.txt"
    wait "$LPID"
    tail -n 12 "$PREFIX.log"
    sleep 5
  done
done

docker compose up -d --scale web=1 --wait >/dev/null && docker compose restart lb >/dev/null
echo "Done. Summarise with: .venv/bin/python loadtest/summarize.py"
