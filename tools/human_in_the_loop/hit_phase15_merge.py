"""
Hungarian-matched GT+prediction merge for the Phase-15 annotation cleanup loop.

Produces a single point set per image where every point carries provenance
metadata (matched_pair / gt_only / pred_only / borderline) so the CVAT
reviewer can prioritise their attention and the download step can route
edits into the right edit-log category (A/B/C/D/E per Phase-15 doc).

Designed to plug into the existing CVAT upload infrastructure used by
``scripts/human_in_the_loop/helper.py`` in the ``active-learning`` repo —
the output is a HastyAnnotationV2 whose label class names encode the
provenance (e.g. ``iguana_pred``, ``iguana_gt``, ``iguana_matched``), which
survive a round trip through CVAT.

Two callers wire this up:
  - ``100_HIT_phase15_upload.py``  — runs the merge, uploads to CVAT
  - ``101_HIT_phase15_download.py`` — pulls corrections, applies them
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import linear_sum_assignment

# These come from the active-learning project. Imported lazily where needed
# so this module can be imported even without the full env wired up.
try:
    from com.biospheredata.types.HastyAnnotationV2 import (
        AnnotatedImage,
        HastyAnnotationV2,
        ImageLabel,
        Keypoint,
    )
except ImportError:  # pragma: no cover
    HastyAnnotationV2 = AnnotatedImage = ImageLabel = Keypoint = None


# Provenance suffixes appended to the class name. Round-trip safe through
# CVAT — class names survive label edits unless a reviewer manually re-types.
SUFFIX_MATCHED = "_matched"        # GT had a predicted neighbour within radius
SUFFIX_GT_ONLY = "_gt"             # GT with no predicted neighbour (review for B = false GT)
SUFFIX_PRED_ONLY = "_pred"         # Prediction with no GT neighbour (review for A = missed GT)
SUFFIX_BORDERLINE = "_borderline"  # Hungarian assigned but distance > radius


@dataclass
class MergeResult:
    """Per-image merge outcome (what gets uploaded for review)."""

    image: str
    matched: list[tuple[float, float, float]]      # (x, y, score) at GT position; reviewer may delete or relocate
    gt_only: list[tuple[float, float, float | None]]   # (x, y, None) GT keypoints with no nearby prediction
    pred_only: list[tuple[float, float, float]]    # (x, y, score) predictions with no nearby GT
    borderline: list[tuple[float, float, float]]   # Hungarian assigned but distance > radius


def _radius_constrained_hungarian(
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
    radius: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Match GT to predictions with linear_sum_assignment, then filter by radius.

    Returns (matched, unmatched_gt_idx, unmatched_pred_idx) where matched is
    [(gt_idx, pred_idx, distance), ...] for pairs whose distance <= radius.
    Pairs assigned by Hungarian but with distance > radius are also returned
    in the matched list but the caller decides how to label them — we mark
    them ``borderline`` upstream.
    """
    n_gt = len(gt_pts)
    n_pred = len(pred_pts)
    if n_gt == 0:
        return [], [], list(range(n_pred))
    if n_pred == 0:
        return [], list(range(n_gt)), []

    cost = np.linalg.norm(
        gt_pts[:, np.newaxis, :] - pred_pts[np.newaxis, :, :], axis=2
    )
    gt_idx, pred_idx = linear_sum_assignment(cost)

    matched_in_radius: list[tuple[int, int, float]] = []
    matched_out_of_radius: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for gi, pi in zip(gt_idx, pred_idx):
        d = float(cost[gi, pi])
        used_gt.add(int(gi))
        used_pred.add(int(pi))
        if d <= radius:
            matched_in_radius.append((int(gi), int(pi), d))
        else:
            matched_out_of_radius.append((int(gi), int(pi), d))

    unmatched_gt = [i for i in range(n_gt) if i not in used_gt]
    unmatched_pred = [i for i in range(n_pred) if i not in used_pred]
    return matched_in_radius + matched_out_of_radius, unmatched_gt, unmatched_pred


def merge_one_image(
    image: str,
    df_gt: pd.DataFrame,
    df_pred: pd.DataFrame,
    radius: float,
    pred_score_threshold: float,
) -> MergeResult:
    """Hungarian-merge GT and predictions for one image.

    df_gt: columns ``images, x, y, ...`` filtered to this image.
    df_pred: columns ``images, x, y, scores, ...`` filtered to this image
    and pre-filtered by score >= pred_score_threshold.
    """
    gt_pts = df_gt[["x", "y"]].to_numpy(dtype=float) if len(df_gt) else np.zeros((0, 2))
    pred_pts = df_pred[["x", "y"]].to_numpy(dtype=float) if len(df_pred) else np.zeros((0, 2))
    pred_scores = df_pred["scores"].to_numpy(dtype=float) if len(df_pred) else np.zeros((0,))

    matches, unmatched_gt_idx, unmatched_pred_idx = _radius_constrained_hungarian(
        gt_pts, pred_pts, radius
    )

    matched: list[tuple[float, float, float]] = []
    borderline: list[tuple[float, float, float]] = []
    for gi, pi, d in matches:
        x, y = float(gt_pts[gi, 0]), float(gt_pts[gi, 1])  # anchor on GT position
        score = float(pred_scores[pi])
        if d <= radius:
            matched.append((x, y, score))
        else:
            borderline.append((x, y, score))

    gt_only = [
        (float(gt_pts[i, 0]), float(gt_pts[i, 1]), None) for i in unmatched_gt_idx
    ]
    pred_only = [
        (float(pred_pts[i, 0]), float(pred_pts[i, 1]), float(pred_scores[i]))
        for i in unmatched_pred_idx
    ]
    return MergeResult(
        image=image, matched=matched, gt_only=gt_only,
        pred_only=pred_only, borderline=borderline,
    )


def merge_dataset(
    df_gt: pd.DataFrame,
    df_pred: pd.DataFrame,
    radius: float,
    pred_score_threshold: float,
) -> list[MergeResult]:
    """Run the Hungarian merge over every image present in either set."""
    df_pred = df_pred[df_pred["scores"] >= pred_score_threshold].copy()
    images: Iterable[str] = sorted(set(df_gt["images"]) | set(df_pred["images"]))
    results = []
    for image in images:
        results.append(
            merge_one_image(
                image=image,
                df_gt=df_gt[df_gt["images"] == image],
                df_pred=df_pred[df_pred["images"] == image],
                radius=radius,
                pred_score_threshold=pred_score_threshold,
            )
        )
    return results


def merge_summary(results: list[MergeResult]) -> dict:
    """Counts useful for the upload log message."""
    total = {"matched": 0, "gt_only": 0, "pred_only": 0, "borderline": 0}
    for r in results:
        total["matched"] += len(r.matched)
        total["gt_only"] += len(r.gt_only)
        total["pred_only"] += len(r.pred_only)
        total["borderline"] += len(r.borderline)
    total["images"] = len(results)
    return total


def merge_results_to_hasty(
    results: list[MergeResult],
    hA_reference: "HastyAnnotationV2",
    dataset_name: str,
    base_class_name: str = "iguana",
    include_matched: bool = False,
) -> "HastyAnnotationV2":
    """Convert per-image MergeResult list into a HastyAnnotationV2.

    Each point becomes a Keypoint label whose ``class_name`` encodes the
    provenance (e.g. ``iguana_pred``, ``iguana_gt``, ``iguana_matched``,
    ``iguana_borderline``). ``include_matched=False`` (default) skips the
    points that are already in agreement between GT and predictions — these
    don't need review and clutter the CVAT task. Set ``True`` for a full
    review pass.
    """
    if HastyAnnotationV2 is None:
        raise ImportError(
            "com.biospheredata.types.HastyAnnotationV2 unavailable — install "
            "the active-learning project or run from that env."
        )

    images: list[AnnotatedImage] = []
    ref_by_name: dict[str, AnnotatedImage] = {img.image_name: img for img in hA_reference.images}

    for r in results:
        labels: list[ImageLabel] = []

        def _make_label(x: float, y: float, suffix: str, score: float | None) -> ImageLabel:
            kp = Keypoint(
                id=str(uuid.uuid4()),
                x=int(round(x)),
                y=int(round(y)),
                keypoint_class_id="body",
            )
            return ImageLabel(
                id=str(uuid.uuid4()),
                class_name=f"{base_class_name}{suffix}",
                keypoints=[kp],
                attributes={"score": score} if score is not None else {},
            )

        if include_matched:
            for x, y, s in r.matched:
                labels.append(_make_label(x, y, SUFFIX_MATCHED, s))
        for x, y, _ in r.gt_only:
            labels.append(_make_label(x, y, SUFFIX_GT_ONLY, None))
        for x, y, s in r.pred_only:
            labels.append(_make_label(x, y, SUFFIX_PRED_ONLY, s))
        for x, y, s in r.borderline:
            labels.append(_make_label(x, y, SUFFIX_BORDERLINE, s))

        ref_img = ref_by_name.get(r.image)
        if ref_img is None:
            logger.warning(
                f"{r.image} not in reference Hasty — skipping. "
                f"Make sure hA_reference contains every image you want to review."
            )
            continue

        new_img = AnnotatedImage(
            **{
                **ref_img.model_dump(exclude={"labels"}),
                "labels": labels,
                "dataset_name": dataset_name,
            }
        )
        images.append(new_img)

    return HastyAnnotationV2(
        project_name=dataset_name,
        images=images,
        export_format_version="1.1",
        label_classes=hA_reference.label_classes,
    )


def classify_corrected_label(class_name: str, was_moved: bool, was_deleted: bool) -> str:
    """Map a corrected CVAT label back to a Phase-15 edit category (A..E).

    Categories from docs/phase15_annotation_cleanup_loop.md:
      A: missed iguana    — pred_only kept by reviewer  -> add to GT
      B: false GT          — gt_only deleted by reviewer -> remove from GT
      C: relocation       — any provenance moved >0 px  -> update (x, y)
      D: borderline       — borderline kept              -> flag for second opinion
      E: confirmed FP     — pred_only deleted            -> log failure mode
    """
    name = class_name.lower()
    if was_deleted:
        if name.endswith(SUFFIX_GT_ONLY):
            return "B"
        if name.endswith(SUFFIX_PRED_ONLY):
            return "E"
        if name.endswith(SUFFIX_MATCHED):
            return "B"  # matched but deleted = reviewer says neither GT nor pred is right
        if name.endswith(SUFFIX_BORDERLINE):
            return "E"
        return "E"
    # kept (not deleted)
    if was_moved:
        return "C"
    if name.endswith(SUFFIX_PRED_ONLY):
        return "A"
    if name.endswith(SUFFIX_BORDERLINE):
        return "D"
    if name.endswith(SUFFIX_GT_ONLY) or name.endswith(SUFFIX_MATCHED):
        return "C" if was_moved else "kept_unchanged"
    return "kept_unchanged"
