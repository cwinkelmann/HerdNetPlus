"""
Error analysis for HerdNet stitched detections.

Loads ground-truth point annotations + a detections.csv and:
  1. Does greedy 1:1 GT-vs-detection matching within a radius (default 100 px,
     matches HerdNetEvaluator's threshold).
  2. Per-frame TP / FP / FN with stratification by GT density.
  3. Score distributions for TPs vs FPs, stratified by frame density.
  4. Saves the top-K highest-confidence FPs and all FNs as 256x256 crops with
     keypoint annotations.
  5. Writes a markdown summary to stdout.

Specifically designed to test whether the LMDS adapt_ts mechanism creates
spurious detections on sparse frames (few-iguana frames where the per-tile
heatmap max is itself a false detection).

Usage:
    python tools/error_analysis.py \\
        --gt /path/to/val/herdnet_format.csv \\
        --det output/phase13_eval_20260508_142316/Nfull/ts_0.20/detections.csv \\
        --images /path/to/val/Default \\
        --out docs/benchmarks/assets/error_analysis_phase13 \\
        --radius 100 \\
        --top-fp 30
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def load_gt(path: str) -> dict:
    """Return {image: [(x, y), ...]} from a HerdNet-format GT CSV."""
    df = pd.read_csv(path)
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _, r in df.iterrows():
        out[r["images"]].append((float(r["x"]), float(r["y"])))
    return out


def load_det(path: str) -> dict:
    """Return {image: [(x, y, score), ...]} from a HerdNet detections CSV."""
    df = pd.read_csv(path)
    out: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for _, r in df.iterrows():
        out[r["images"]].append((float(r["x"]), float(r["y"]), float(r["scores"])))
    return out


def greedy_match(
    gt_points: list[tuple[float, float]],
    det_points: list[tuple[float, float, float]],
    radius: float,
) -> tuple[list[int], list[int], list[int]]:
    """Greedy 1:1 matching of dets (sorted by score desc) to GT within radius.

    Returns (matched_det_idx_list, matched_gt_idx_list, unmatched_det_idx_list).
    Unmatched GTs = set(range(len(gt_points))) - set(matched_gt_idx_list).
    """
    order = sorted(range(len(det_points)), key=lambda i: -det_points[i][2])
    used_gt: set[int] = set()
    tp_det: list[int] = []
    tp_gt: list[int] = []
    fp_det: list[int] = []

    for di in order:
        dx, dy, _ = det_points[di]
        best_gi, best_d = -1, float("inf")
        for gi, (gx, gy) in enumerate(gt_points):
            if gi in used_gt:
                continue
            d = math.hypot(dx - gx, dy - gy)
            if d < best_d and d <= radius:
                best_d = d
                best_gi = gi
        if best_gi >= 0:
            used_gt.add(best_gi)
            tp_det.append(di)
            tp_gt.append(best_gi)
        else:
            fp_det.append(di)
    return tp_det, tp_gt, fp_det


def density_bucket(n_gt: int) -> str:
    """Coarse density binning for stratified analysis."""
    if n_gt <= 2:
        return "sparse_1-2"
    if n_gt <= 10:
        return "medium_3-10"
    return "dense_11+"


def crop_with_marker(
    img_path: Path,
    cx: float,
    cy: float,
    out_path: Path,
    color: str,
    size: int = 256,
) -> None:
    """Save a size x size crop centred on (cx, cy) with a circle marker."""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    half = size // 2
    x0 = max(0, int(cx) - half)
    y0 = max(0, int(cy) - half)
    x1 = min(W, x0 + size)
    y1 = min(H, y0 + size)
    x0 = max(0, x1 - size)
    y0 = max(0, y1 - size)
    crop = img.crop((x0, y0, x1, y1))
    draw = ImageDraw.Draw(crop)
    mx, my = cx - x0, cy - y0
    r = 18
    draw.ellipse([mx - r, my - r, mx + r, my + r], outline=color, width=4)
    crop.save(out_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--det", required=True)
    p.add_argument("--images", required=True, help="Directory containing full-size frames")
    p.add_argument("--out", required=True, help="Output directory for crops + report")
    p.add_argument("--radius", type=float, default=100.0)
    p.add_argument("--top-fp", type=int, default=30)
    p.add_argument("--top-fn", type=int, default=30)
    p.add_argument("--max-crops", type=int, default=100, help="Max FP crops dumped")
    args = p.parse_args()

    out_dir = Path(args.out)
    (out_dir / "fp").mkdir(parents=True, exist_ok=True)
    (out_dir / "fn").mkdir(parents=True, exist_ok=True)

    gt = load_gt(args.gt)
    det = load_det(args.det)
    all_images = sorted(set(gt.keys()) | set(det.keys()))

    # ---- per-frame matching ----
    per_frame = []
    all_fp = []  # (image, x, y, score)
    all_fn = []  # (image, x, y)
    for img in all_images:
        gt_pts = gt.get(img, [])
        det_pts = det.get(img, [])
        tp_det, tp_gt, fp_det_idx = greedy_match(gt_pts, det_pts, args.radius)
        fn_gt_idx = sorted(set(range(len(gt_pts))) - set(tp_gt))

        # Score stats
        tp_scores = [det_pts[i][2] for i in tp_det]
        fp_scores = [det_pts[i][2] for i in fp_det_idx]
        n_gt = len(gt_pts)
        n_tp = len(tp_det)
        n_fp = len(fp_det_idx)
        n_fn = len(fn_gt_idx)
        per_frame.append({
            "image": img,
            "bucket": density_bucket(n_gt),
            "n_gt": n_gt,
            "tp": n_tp,
            "fp": n_fp,
            "fn": n_fn,
            "tp_score_mean": float(np.mean(tp_scores)) if tp_scores else None,
            "fp_score_mean": float(np.mean(fp_scores)) if fp_scores else None,
            "fp_score_max": float(max(fp_scores)) if fp_scores else None,
            "fp_score_min": float(min(fp_scores)) if fp_scores else None,
        })

        for i in fp_det_idx:
            x, y, s = det_pts[i]
            all_fp.append({"image": img, "x": x, "y": y, "score": s, "n_gt": n_gt})
        for i in fn_gt_idx:
            x, y = gt_pts[i]
            all_fn.append({"image": img, "x": x, "y": y, "n_gt": n_gt})

    df_frame = pd.DataFrame(per_frame)
    df_frame.to_csv(out_dir / "per_frame.csv", index=False)
    df_fp = pd.DataFrame(all_fp).sort_values("score", ascending=False)
    df_fp.to_csv(out_dir / "fp.csv", index=False)
    df_fn = pd.DataFrame(all_fn)
    df_fn.to_csv(out_dir / "fn.csv", index=False)

    # ---- stratified summary ----
    print("# Error analysis — Phase 13 N=full single B4, ts=0.20")
    print()
    print(f"- GT CSV: `{args.gt}`")
    print(f"- Detections: `{args.det}`")
    print(f"- Match radius: {args.radius:.0f} px")
    print()

    total_gt = int(df_frame["n_gt"].sum())
    total_tp = int(df_frame["tp"].sum())
    total_fp = int(df_frame["fp"].sum())
    total_fn = int(df_frame["fn"].sum())
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print("## Aggregate")
    print()
    print(f"- Frames: {len(df_frame)}")
    print(f"- GT iguanas: {total_gt}")
    print(f"- TP / FP / FN: **{total_tp} / {total_fp} / {total_fn}**")
    print(f"- Precision: **{precision:.4f}**")
    print(f"- Recall: **{recall:.4f}**")
    print(f"- F1: **{f1:.4f}**")
    print()

    # ---- stratified analysis ----
    print("## Stratified by frame density (THE KEY adapt_ts TEST)")
    print()
    print("If the LMDS `adapt_ts` mechanism is artificially boosting FPs on sparse")
    print("frames (where per-tile max is itself a spurious detection), we should see")
    print("FP-per-GT rise sharply at low GT density.")
    print()
    print("| Bucket | Frames | GT | TP | FP | FN | Recall | Precision | FP/GT | FP/frame | mean FP score |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for bucket in ["sparse_1-2", "medium_3-10", "dense_11+"]:
        sub = df_frame[df_frame["bucket"] == bucket]
        if sub.empty:
            continue
        n = len(sub)
        sub_gt = int(sub["n_gt"].sum())
        sub_tp = int(sub["tp"].sum())
        sub_fp = int(sub["fp"].sum())
        sub_fn = int(sub["fn"].sum())
        sub_recall = sub_tp / (sub_tp + sub_fn) if sub_tp + sub_fn else 0
        sub_precision = sub_tp / (sub_tp + sub_fp) if sub_tp + sub_fp else 0
        fp_per_gt = sub_fp / sub_gt if sub_gt else float("nan")
        fp_per_frame = sub_fp / n if n else 0
        mean_fp_score = float(sub["fp_score_mean"].mean()) if sub["fp_score_mean"].notna().any() else float("nan")
        print(f"| {bucket} | {n} | {sub_gt} | {sub_tp} | {sub_fp} | {sub_fn} | {sub_recall:.4f} | {sub_precision:.4f} | {fp_per_gt:.3f} | {fp_per_frame:.2f} | {mean_fp_score:.3f} |")
    print()

    print("## FP score distribution by density bucket")
    print()
    print("| Bucket | <0.30 | 0.30-0.50 | 0.50-0.70 | 0.70-0.90 | ≥0.90 |")
    print("|---|---|---|---|---|---|")
    for bucket in ["sparse_1-2", "medium_3-10", "dense_11+"]:
        bucket_imgs = set(df_frame[df_frame["bucket"] == bucket]["image"])
        sub_fp = df_fp[df_fp["image"].isin(bucket_imgs)]
        bands = [
            (sub_fp["score"] < 0.30).sum(),
            ((sub_fp["score"] >= 0.30) & (sub_fp["score"] < 0.50)).sum(),
            ((sub_fp["score"] >= 0.50) & (sub_fp["score"] < 0.70)).sum(),
            ((sub_fp["score"] >= 0.70) & (sub_fp["score"] < 0.90)).sum(),
            (sub_fp["score"] >= 0.90).sum(),
        ]
        print(f"| {bucket} | {bands[0]} | {bands[1]} | {bands[2]} | {bands[3]} | {bands[4]} |")
    print()

    # ---- top FPs (highest confidence) and FNs ----
    print("## Top false positives by confidence")
    print()
    top_fp = df_fp.head(args.top_fp)
    print("| rank | image | (x, y) | score | frame_GT |")
    print("|---|---|---|---|---|")
    for i, (_, r) in enumerate(top_fp.iterrows(), 1):
        print(f"| {i} | {r['image']} | ({int(r['x'])}, {int(r['y'])}) | {r['score']:.4f} | {r['n_gt']} |")
    print()

    # ---- crop dumps ----
    img_dir = Path(args.images)
    n_fp_to_dump = min(args.max_crops, len(df_fp))
    print(f"## Saving {n_fp_to_dump} highest-confidence FP crops to `{out_dir}/fp/`")
    for i, (_, r) in enumerate(df_fp.head(n_fp_to_dump).iterrows()):
        out_name = out_dir / "fp" / f"FP_{i:03d}_score{r['score']:.3f}_{Path(r['image']).stem}_{int(r['x'])}_{int(r['y'])}.jpg"
        crop_with_marker(img_dir / r["image"], r["x"], r["y"], out_name, color="orange")
    print(f"Saving all {len(df_fn)} FN crops to `{out_dir}/fn/`")
    for i, (_, r) in enumerate(df_fn.iterrows()):
        out_name = out_dir / "fn" / f"FN_{i:03d}_{Path(r['image']).stem}_{int(r['x'])}_{int(r['y'])}.jpg"
        crop_with_marker(img_dir / r["image"], r["x"], r["y"], out_name, color="red")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
