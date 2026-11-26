"""Summarise the Locust runs in loadtest/results/ into a table and a chart.

Reads every r{containers}_u{users}_stats.csv written by run_matrix.sh (plus the
per-container request counts and docker stats snapshots) and writes:

- loadtest/results/summary.md                  markdown table (also pasted into the README)
- loadtest/results/latency_vs_containers.png   throughput and p95 latency vs containers

    python loadtest/summarize.py [results_dir]       # default: loadtest/results
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

# Chart tokens (reference palette, light mode). Users is an ordered magnitude,
# so the series use one blue ramp light -> dark (validated as an ordinal ramp).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
RAMP = ["#6da7ec", "#2a78d6", "#104281"]


def _num(value: str) -> float:
    return float(value) if value not in ("", "N/A") else float("nan")


def load_runs() -> list:
    runs = []
    for path in sorted(RESULTS.glob("r*_u*_stats.csv")):
        m = re.match(r"r(\d+)_u(\d+)_stats\.csv", path.name)
        if not m:
            continue
        containers, users = int(m.group(1)), int(m.group(2))
        with open(path, newline="") as fh:
            rows = {r["Name"]: r for r in csv.DictReader(fh)}
        p = rows.get("/api/predict")
        if p is None:
            continue
        served_path = RESULTS / f"r{containers}_u{users}_served_by.json"
        served = json.loads(served_path.read_text()) if served_path.exists() else {}
        served.pop("unknown", None)
        stats_path = RESULTS / f"r{containers}_u{users}.docker_stats.txt"
        cpu = []
        if stats_path.exists():
            for line in stats_path.read_text().splitlines():
                name, cpu_pct, *_ = line.split("\t")
                if "-web-" in name:
                    cpu.append(float(cpu_pct.rstrip("%")))
        requests = int(p["Request Count"])
        runs.append({
            "containers": containers,
            "users": users,
            "requests": requests,
            "failures": int(p["Failure Count"]),
            "rps": _num(p["Requests/s"]),
            "median_ms": _num(p["Median Response Time"]),
            "avg_ms": _num(p["Average Response Time"]),
            "p95_ms": _num(p["95%"]),
            "p99_ms": _num(p["99%"]),
            "served": served,
            "cpu": cpu,
        })
    return sorted(runs, key=lambda r: (r["containers"], r["users"]))


def _spread(served: dict) -> str:
    total = sum(served.values())
    if not total:
        return "—"
    return " / ".join(f"{100 * n / total:.0f}%" for _, n in sorted(served.items()))


def write_table(runs: list) -> str:
    lines = [
        "| Containers | Users | Predictions | Failures | Throughput (pred/s) "
        "| Median (ms) | p95 (ms) | p99 (ms) | Split across containers | CPU per container |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|",
    ]
    for r in runs:
        fail_pct = 100 * r["failures"] / r["requests"] if r["requests"] else 0
        cpu = " / ".join(f"{c:.0f}%" for c in r["cpu"]) or "—"
        lines.append(
            f"| {r['containers']} | {r['users']} | {r['requests']:,} | "
            f"{r['failures']} ({fail_pct:.1f}%) | {r['rps']:.2f} | {r['median_ms']:,.0f} | "
            f"{r['p95_ms']:,.0f} | {r['p99_ms']:,.0f} | {_spread(r['served'])} | {cpu} |")
    table = "\n".join(lines)
    (RESULTS / "summary.md").write_text(
        "# Locust flood test — results\n\nPOST /api/predict with held-out fundus images; "
        "see environment.txt for the setup.\n\n" + table + "\n", encoding="utf-8")
    return table


def _style(ax, title: str, ylabel: str, containers: list) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("Containers behind the load balancer", color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xticks(containers)
    ax.set_xlim(min(containers) - 0.25, max(containers) + 0.75)   # room for end labels
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _end_labels(ax, series: list) -> None:
    """Label a line at its right end only if no other line ends close to it.

    Lines that end near each other are left to the legend: labelling one of
    a converging pair would put its name beside the other line's marker.
    """
    lo, hi = ax.get_ylim()
    for x, y, text in series:
        others = [oy for _, oy, ot in series if ot != text]
        if all(abs(y - oy) / (hi - lo) >= 0.08 for oy in others):
            ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2)


def plot(runs: list) -> Path:
    users = sorted({r["users"] for r in runs})
    containers = sorted({r["containers"] for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    panels = [("rps", "Throughput", "Predictions per second", 1.0),
              ("p95_ms", "95th-percentile latency", "Seconds", 1000.0)]
    for ax, (key, title, ylabel, scale) in zip(axes, panels):
        _style(ax, title, ylabel, containers)
        ends, top = [], 0.0
        for colour, u in zip(RAMP[-len(users):], users):
            pts = sorted((r["containers"], r[key] / scale) for r in runs if r["users"] == u)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=colour, linewidth=2, solid_capstyle="round",
                    solid_joinstyle="round", marker="o", markersize=8,
                    markerfacecolor=colour, markeredgecolor=SURFACE, markeredgewidth=2,
                    label=f"{u} users", zorder=3)
            ends.append((xs[-1], ys[-1], f"{u} users"))
            top = max(top, *ys)
        ax.set_ylim(0, top * 1.12)       # zero baseline, headroom set from the data
        _end_labels(ax, ends)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(users), frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.98, 1.0))

    best = {u: max((r for r in runs if r["users"] == u), key=lambda r: r["containers"])
            for u in users}
    one = {u: next((r for r in runs if r["users"] == u and r["containers"] == 1), None)
           for u in users}
    u_max = max(users)
    if one[u_max]:
        fig.suptitle(
            f"At {u_max} users, {best[u_max]['containers']} containers serve "
            f"{best[u_max]['rps']:.1f} predictions/s vs {one[u_max]['rps']:.1f} with 1",
            x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = RESULTS / "latency_vs_containers.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


def main() -> None:
    global RESULTS
    ap = argparse.ArgumentParser(description="Summarise Locust results.")
    ap.add_argument("results_dir", nargs="?", type=Path, default=RESULTS)
    RESULTS = ap.parse_args().results_dir.resolve()
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No Locust results in {RESULTS} (run loadtest/run_matrix.sh).")
    print(write_table(runs))
    print(f"\nChart: {plot(runs)}")


if __name__ == "__main__":
    main()
