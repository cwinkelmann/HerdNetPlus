#!/usr/bin/env python3
"""Pull Phase-13 data-scaling results from wandb and compare across N.

Wandb project: ``hn_phase13_data_scaling``. Run names follow
``[<date>_]phase13_N<N>_s<seed>[_docker][_<wandb_id>]`` (parsed below).

Usage:
    python tools/pull_wandb_phase13.py
    python tools/pull_wandb_phase13.py --project hn_phase13_data_scaling
    python tools/pull_wandb_phase13.py --csv phase13_results.csv
    python tools/pull_wandb_phase13.py --plot phase13_scaling.png
    python tools/pull_wandb_phase13.py --include-failed
"""

from __future__ import annotations

import argparse
import csv as csv_module
import re
import sys
from pathlib import Path

import wandb

RUN_NAME_RE = re.compile(
    r"phase13_N(?P<n>\w+)_s(?P<seed>\d+)(?P<docker>_docker)?",
    re.IGNORECASE,
)

# Summary keys we care about. (wandb key, output column, type)
METRIC_KEYS = [
    ("best_f1_score", "f1", float),
    ("best_f2_score", "f2", float),
    ("best_recall", "recall", float),
    ("best_precision", "precision", float),
    ("best_mae", "mae", float),
    ("best_rmse", "rmse", float),
    ("best_total_loss", "loss", float),
    ("best_epoch", "best_epoch", int),
]


def parse_n(name: str) -> str | None:
    m = RUN_NAME_RE.search(name)
    return m.group("n") if m else None


def n_sort_key(n: str) -> int:
    """Sort order: 19 < 38 < ... < 2432 < full."""
    if n.lower() == "full":
        return 10**9
    try:
        return int(n)
    except ValueError:
        return 10**8


def get(summary: dict, key: str, cast) -> object:
    v = summary.get(key)
    if v is None:
        return None
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def collect_runs(project: str, include_failed: bool) -> list[dict]:
    api = wandb.Api()
    runs = api.runs(project)
    rows = []
    for r in runs:
        n = parse_n(r.name)
        if n is None:
            print(f"  (skip, can't parse N) {r.name}", file=sys.stderr)
            continue
        if r.state != "finished" and not include_failed:
            print(f"  (skip, state={r.state}) {r.name}", file=sys.stderr)
            continue
        row = {
            "n_label": n,
            "n_sort": n_sort_key(n),
            "seed": r.config.get("seed", "?"),
            "state": r.state,
            "docker": "_docker" in r.name,
            "run_name": r.name,
            "run_id": r.id,
            "url": r.url,
        }
        for wandb_key, col, cast in METRIC_KEYS:
            row[col] = get(r.summary, wandb_key, cast)
        rows.append(row)
    rows.sort(key=lambda d: (d["n_sort"], d["seed"]))
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no runs)")
        return

    cols = ["n_label", "seed", "state", "docker",
            "f1", "f2", "recall", "precision", "mae", "rmse", "best_epoch"]
    widths = {c: max(len(c), 5) for c in cols}
    fmt_rows: list[dict] = []
    for r in rows:
        fr = {}
        for c in cols:
            v = r.get(c)
            if v is None:
                s = "—"
            elif isinstance(v, float):
                s = f"{v:.4f}" if c not in ("mae", "rmse") else f"{v:.2f}"
            elif isinstance(v, bool):
                s = "yes" if v else "no"
            else:
                s = str(v)
            widths[c] = max(widths[c], len(s))
            fr[c] = s
        fmt_rows.append(fr)

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for fr in fmt_rows:
        print("  ".join(fr[c].ljust(widths[c]) for c in cols))


def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    cols = ["n_label", "seed", "state", "docker",
            "f1", "f2", "recall", "precision", "mae", "rmse",
            "best_epoch", "run_name", "run_id", "url"]
    with open(path, "w", newline="") as f:
        w = csv_module.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    print(f"\nwrote {path}")


def plot_scaling(rows: list[dict], path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot", file=sys.stderr)
        return
    finished = [r for r in rows if r["state"] == "finished" and r["f1"] is not None]
    if not finished:
        print("no finished runs with F1 — skipping plot", file=sys.stderr)
        return

    xs = [r["n_sort"] for r in finished]
    f1 = [r["f1"] for r in finished]
    mae = [r["mae"] for r in finished]
    labels = [r["n_label"] for r in finished]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ys, ylabel in zip(axes, [f1, mae], ["best F1", "best MAE"]):
        ax.plot(xs, ys, "o-", color="#1f77b4")
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("training pool size N (log)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase-13 data scaling — B4 warm-started from Phase-8")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="hn_phase13_data_scaling",
                    help="wandb project (default: hn_phase13_data_scaling)")
    ap.add_argument("--csv", help="write results to CSV")
    ap.add_argument("--plot", help="write scaling-curve PNG (F1 + MAE vs N)")
    ap.add_argument("--include-failed", action="store_true",
                    help="include runs with state != finished")
    args = ap.parse_args()

    rows = collect_runs(args.project, args.include_failed)
    print(f"\n{len(rows)} runs from project '{args.project}':\n")
    print_table(rows)
    if args.csv:
        save_csv(rows, args.csv)
    if args.plot:
        plot_scaling(rows, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
