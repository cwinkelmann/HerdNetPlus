"""
Two-model ensemble for phase-15 candidate generation.

Idea: each model's vegetation hallucinations are largely uncorrelated.
Real iguanas tend to be predicted by BOTH models. So if we only keep
predictions where both models agree (one of model-A's points lies
within `radius` px of one of model-B's points on the same image), we
drop most of the FPs while losing relatively few real iguanas.

The output is a `detections.csv` with the same schema as a single-model
run, so the existing
``tools/human_in_the_loop/100_HIT_phase15_upload.py`` ingests it
unchanged. Each consensus point uses the AVERAGE xy of the two matched
points and the MAX of the two confidence scores.

Usage:
    python ensemble_detections.py \\
        --det-a /path/to/al_v7_inference/detections.csv \\
        --det-b /path/to/al_v8_inference/detections.csv \\
        --out  /path/to/consensus_detections.csv \\
        [--radius 50] [--min-score 0.3]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import linear_sum_assignment


def _consensus_for_image(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    radius: float,
) -> pd.DataFrame:
    """For one image, return the consensus detections.

    Hungarian-match A's predictions to B's predictions by xy distance,
    constrained by `radius`. Each kept pair emits one consensus row
    (averaged xy, max score). Unpaired predictions are dropped.
    """
    a_xy = df_a[["x", "y"]].to_numpy(dtype=float)
    b_xy = df_b[["x", "y"]].to_numpy(dtype=float)
    if len(a_xy) == 0 or len(b_xy) == 0:
        return df_a.iloc[0:0]

    # Pairwise distance matrix.
    diff = a_xy[:, None, :] - b_xy[None, :, :]
    cost = np.sqrt((diff * diff).sum(-1))
    # Hungarian on the (possibly rectangular) cost matrix.
    row_ind, col_ind = linear_sum_assignment(cost)

    out_rows = []
    template = df_a.iloc[0]
    for ai, bj in zip(row_ind, col_ind):
        if cost[ai, bj] > radius:
            continue
        ax, ay = a_xy[ai]
        bx, by = b_xy[bj]
        x = 0.5 * (ax + bx)
        y = 0.5 * (ay + by)
        s_a = float(df_a.iloc[ai]["scores"])
        s_b = float(df_b.iloc[bj]["scores"])
        d_a = float(df_a.iloc[ai].get("dscores", s_a))
        d_b = float(df_b.iloc[bj].get("dscores", s_b))
        out_rows.append({
            **template.to_dict(),
            "x": x,
            "y": y,
            "scores": max(s_a, s_b),    # tightest possible vote
            "dscores": max(d_a, d_b),
        })
    if not out_rows:
        return df_a.iloc[0:0]
    return pd.DataFrame(out_rows, columns=df_a.columns)


def ensemble(
    det_a: Path,
    det_b: Path,
    out: Path,
    radius: float = 50.0,
    min_score: float = 0.0,
) -> dict:
    logger.info(f"Loading model A detections: {det_a}")
    df_a = pd.read_csv(det_a)
    logger.info(f"Loading model B detections: {det_b}")
    df_b = pd.read_csv(det_b)

    if min_score > 0:
        df_a = df_a[df_a["scores"] >= min_score].reset_index(drop=True)
        df_b = df_b[df_b["scores"] >= min_score].reset_index(drop=True)

    logger.info(
        f"Model A: {len(df_a)} dets, {df_a['images'].nunique()} images. "
        f"Model B: {len(df_b)} dets, {df_b['images'].nunique()} images."
    )

    images = sorted(set(df_a["images"]) | set(df_b["images"]))
    chunks = []
    consensus_per_image = Counter()
    a_only_per_image = Counter()
    b_only_per_image = Counter()
    for img in images:
        sub_a = df_a[df_a["images"] == img]
        sub_b = df_b[df_b["images"] == img]
        if len(sub_a) == 0 or len(sub_b) == 0:
            a_only_per_image[img] = len(sub_a)
            b_only_per_image[img] = len(sub_b)
            continue
        cons = _consensus_for_image(sub_a, sub_b, radius)
        if len(cons) > 0:
            chunks.append(cons)
            consensus_per_image[img] = len(cons)
        a_only_per_image[img] = max(0, len(sub_a) - len(cons))
        b_only_per_image[img] = max(0, len(sub_b) - len(cons))

    if chunks:
        out_df = pd.concat(chunks, ignore_index=True)
    else:
        out_df = df_a.iloc[0:0]

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    logger.info(f"Wrote consensus detections: {out}  rows={len(out_df)}")

    summary = {
        "a_total": int(len(df_a)),
        "b_total": int(len(df_b)),
        "consensus_total": int(len(out_df)),
        "a_drop_rate": round(1.0 - len(out_df) / max(len(df_a), 1), 3),
        "b_drop_rate": round(1.0 - len(out_df) / max(len(df_b), 1), 3),
        "a_only_unmatched_total": sum(a_only_per_image.values()),
        "b_only_unmatched_total": sum(b_only_per_image.values()),
        "images_with_consensus": len([1 for c in consensus_per_image.values() if c > 0]),
    }
    logger.info(
        f"Consensus rows: {summary['consensus_total']}  "
        f"(A drop {summary['a_drop_rate']*100:.1f}%, "
        f"B drop {summary['b_drop_rate']*100:.1f}%)"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--det-a", required=True, type=Path,
                   help="detections.csv from model A (e.g. al_v7)")
    p.add_argument("--det-b", required=True, type=Path,
                   help="detections.csv from model B (e.g. al_v8)")
    p.add_argument("--out", required=True, type=Path,
                   help="output consensus_detections.csv")
    p.add_argument("--radius", type=float, default=50.0,
                   help="Hungarian-match radius in px (default 50)")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="drop both A and B predictions below this score "
                        "before intersecting (default 0.0 = no filter)")
    args = p.parse_args(argv)
    ensemble(args.det_a, args.det_b, args.out,
             radius=args.radius, min_score=args.min_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
