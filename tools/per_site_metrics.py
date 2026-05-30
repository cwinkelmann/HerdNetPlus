"""
Per-site precision/recall/F1 breakdown for HerdNet stitched detections.

Site identifier = the part of the image filename before "___".

Usage:
    python tools/per_site_metrics.py \\
        --gt /path/to/val/herdnet_format.csv \\
        --det output/.../detections.csv \\
        --model "phase13_single_b4" \\
        --threshold 0.30 \\
        --out output/.../test_per_site.csv \\
        [--radius 100]

Appends one row per (model, threshold, site) plus one row with site=OVERALL.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_gt(path: str) -> dict:
    df = pd.read_csv(path)
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _, r in df.iterrows():
        out[r["images"]].append((float(r["x"]), float(r["y"])))
    return out


def load_det(path: str) -> dict:
    df = pd.read_csv(path)
    out: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for _, r in df.iterrows():
        out[r["images"]].append((float(r["x"]), float(r["y"]), float(r["scores"])))
    return out


def site_of(image: str) -> str:
    return image.split("___", 1)[0] if "___" in image else "UNKNOWN"


def greedy_match(gt_pts, det_pts, radius: float) -> tuple[int, int, int]:
    order = sorted(range(len(det_pts)), key=lambda i: -det_pts[i][2])
    used: set[int] = set()
    tp = 0
    fp = 0
    for di in order:
        dx, dy, _ = det_pts[di]
        best, best_d = -1, float("inf")
        for gi, (gx, gy) in enumerate(gt_pts):
            if gi in used:
                continue
            d = math.hypot(dx - gx, dy - gy)
            if d < best_d and d <= radius:
                best_d = d
                best = gi
        if best >= 0:
            used.add(best)
            tp += 1
        else:
            fp += 1
    fn = len(gt_pts) - tp
    return tp, fp, fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--det", required=True)
    p.add_argument("--model", required=True, help="Label for this model variant")
    p.add_argument("--threshold", required=True, type=float)
    p.add_argument("--dataset", default="test", help="val or test (label only)")
    p.add_argument("--out", required=True, help="Aggregate CSV (appended)")
    p.add_argument("--radius", type=float, default=100.0)
    args = p.parse_args()

    gt = load_gt(args.gt)
    det = load_det(args.det)
    images = sorted(set(gt.keys()) | set(det.keys()))

    site_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    overall = [0, 0, 0]

    for img in images:
        gt_pts = gt.get(img, [])
        det_pts = det.get(img, [])
        tp, fp, fn = greedy_match(gt_pts, det_pts, args.radius)
        s = site_of(img)
        site_totals[s][0] += tp
        site_totals[s][1] += fp
        site_totals[s][2] += fn
        overall[0] += tp
        overall[1] += fp
        overall[2] += fn

    rows = []
    for site in sorted(site_totals):
        tp, fp, fn = site_totals[site]
        n_gt = tp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "model": args.model, "dataset": args.dataset, "threshold": args.threshold,
            "site": site, "n_gt": n_gt, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        })

    tp, fp, fn = overall
    n_gt = tp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rows.append({
        "model": args.model, "dataset": args.dataset, "threshold": args.threshold,
        "site": "OVERALL", "n_gt": n_gt, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
    })

    df = pd.DataFrame(rows)
    header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    df.to_csv(args.out, mode="a", header=header, index=False)

    overall_row = rows[-1]
    print(f"  [{args.model} ts={args.threshold} {args.dataset}] OVERALL: "
          f"F1={overall_row['f1']:.4f} P={overall_row['precision']:.4f} "
          f"R={overall_row['recall']:.4f} tp={tp} fp={fp} fn={fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
