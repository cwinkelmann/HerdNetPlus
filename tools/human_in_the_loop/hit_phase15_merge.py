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

# Distinct CVAT marker colours per provenance — picked to be visually
# distinct and reasonable for colour-vision deficiencies.
PROVENANCE_COLORS = {
    SUFFIX_MATCHED:    "#2ca02c",   # green   : model + GT agree
    SUFFIX_GT_ONLY:    "#2c7fb8",   # blue    : GT only — possible false GT (B)
    SUFFIX_PRED_ONLY:  "#d62728",   # red     : prediction only — possible missed iguana (A)
    SUFFIX_BORDERLINE: "#ffcc00",   # yellow  : paired by Hungarian but too far apart (D)
}

# The canonical class name the Hasty master uses for iguana keypoints.
# The download step collapses every provenance variant back to this.
CANONICAL_CLASS_NAME = "iguana_point"

# Non-iguana classes in the Hasty schema. If a CVAT reviewer changes a
# pred_only/borderline label's class to one of these (or any non-iguana
# class), the download step treats it as a hard-negative promotion: the
# label is added to the master Hasty under that class so future training
# sees it as an explicit negative example at that position.
NON_IGUANA_CLASSES = {
    "not_iguana_but_similar_look",
    "ugly_stone",
    "fresh_lava",
    "trash",
    "hard_negative",
    "crab",
    "turtle",
    "seal",
    "bird",
}


def is_iguana_class(class_name: str) -> bool:
    """True if the label class represents an iguana (any iguana_point* variant)."""
    if not class_name:
        return True
    cn = class_name.lower()
    if cn in NON_IGUANA_CLASSES:
        return False
    return "iguana" in cn


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


def suppress_near_hard_negatives(
    results: list[MergeResult],
    hA_master,
    radius: float,
) -> tuple[int, int]:
    """Drop pred_only / borderline candidates that sit on top of an existing
    non-iguana label in the master Hasty.

    Why this matters: the iter-N download promotes deleted pred_only markers
    to `not_iguana_but_similar_look` keypoints in master. Without this
    filter, the model — which hasn't been retrained between iterations —
    keeps emitting predictions at those same positions, and the next
    upload's Hungarian merge surfaces them again as red `iguana_point_pred`
    markers. The reviewer would have to keep deleting the same FPs.

    Non-iguana labels considered (per NON_IGUANA_CLASSES): keypoint position
    if present, else bbox centroid (read-only — we don't mutate boxes).

    Returns (n_pred_only_dropped, n_borderline_dropped).
    """
    from collections import defaultdict

    by_key: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for img in hA_master.images:
        if not img.dataset_name or not img.image_name:
            continue
        for label in img.labels:
            if label.class_name not in NON_IGUANA_CLASSES:
                continue
            pos: tuple[float, float] | None = None
            if label.keypoints:
                kp = label.keypoints[0]
                if kp.x is not None and kp.y is not None:
                    pos = (float(kp.x), float(kp.y))
            elif getattr(label, "incenter_centroid", None) is not None:
                ic = label.incenter_centroid
                if getattr(ic, "x", None) is not None and getattr(ic, "y", None) is not None:
                    pos = (float(ic.x), float(ic.y))
            if pos is not None:
                by_key[(img.dataset_name, img.image_name)].append(pos)

    r2 = radius * radius
    n_pred = 0
    n_border = 0
    for r in results:
        if "___" not in r.image:
            continue
        ds, hname = r.image.split("___", 1)
        positions = by_key.get((ds, hname))
        if not positions:
            continue

        def too_close(x, y):
            for hx, hy in positions:
                if (x - hx) ** 2 + (y - hy) ** 2 <= r2:
                    return True
            return False

        before_p = len(r.pred_only)
        r.pred_only = [t for t in r.pred_only if not too_close(t[0], t[1])]
        n_pred += before_p - len(r.pred_only)
        before_b = len(r.borderline)
        r.borderline = [t for t in r.borderline if not too_close(t[0], t[1])]
        n_border += before_b - len(r.borderline)
    return n_pred, n_border


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
    base_class_name: str = "iguana_point",
    include_matched: bool = False,
    images_path: "Path | None" = None,
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
            # Reference Hasty doesn't have this image (common for the
            # Phase-13 use case: tiles use the `<dataset>___<name>` convention
            # while the Hasty master stores raw `<name>` with `dataset_name`
            # carried separately). Build a minimal AnnotatedImage; read
            # width/height from the tile on disk if images_path is given.
            width, height = 0, 0
            if images_path is not None:
                tile_path = images_path / r.image
                if tile_path.exists():
                    import PIL.Image as _PIL
                    with _PIL.open(tile_path) as im:
                        width, height = im.size
            new_img = AnnotatedImage(
                image_name=r.image,
                dataset_name=dataset_name,
                labels=labels,
                width=width,
                height=height,
            )
        else:
            new_img = AnnotatedImage(
                **{
                    **ref_img.model_dump(exclude={"labels"}),
                    "labels": labels,
                    "dataset_name": dataset_name,
                }
            )
        images.append(new_img)

    # Inject the 4 provenance label-classes (with colors) into the existing
    # reference label_classes list so CVAT shows distinct markers per
    # provenance. Existing classes are preserved — reviewers can still
    # re-type to e.g. "not_iguana_but_similar_look" if they want.
    label_classes = _label_classes_with_provenance(
        hA_reference.label_classes,
        base_class_name=base_class_name,
    )

    return HastyAnnotationV2(
        project_name=dataset_name,
        images=images,
        export_format_version="1.1",
        label_classes=label_classes,
    )


def _label_classes_with_provenance(existing, base_class_name: str):
    """Return existing label_classes + 4 provenance-tagged variants.

    Each provenance variant inherits the existing class's attributes list if
    a base class is found in ``existing``; otherwise it uses an empty list.
    Colors are pulled from PROVENANCE_COLORS.
    """
    import uuid

    # Try to mimic the base class's attribute/description so CVAT shows
    # the same edit affordances.
    base = None
    for lc in existing:
        # ``existing`` is a list of LabelClass models (or dicts depending on
        # how Hasty was loaded). Handle both shapes.
        name = lc.class_name if hasattr(lc, "class_name") else lc.get("class_name")
        if name == base_class_name or name == CANONICAL_CLASS_NAME:
            base = lc
            break

    def _attr_of(lc):
        return lc.attributes if hasattr(lc, "attributes") else lc.get("attributes", [])

    base_attrs = _attr_of(base) if base is not None else []

    # Detect whether existing entries are pydantic models (HastyAnnotationV2's
    # LabelClass) or plain dicts. Construct in the matching shape.
    use_models = bool(existing) and hasattr(existing[0], "model_dump")
    if use_models:
        LabelClass = type(existing[0])
    else:
        LabelClass = None

    new_entries = []
    max_norder = 0
    for lc in existing:
        n = lc.norder if hasattr(lc, "norder") else lc.get("norder", 0) or 0
        try:
            max_norder = max(max_norder, float(n))
        except (TypeError, ValueError):
            pass

    for i, (suffix, color) in enumerate(PROVENANCE_COLORS.items(), start=1):
        payload = {
            "class_id": str(uuid.uuid4()),
            "parent_class_id": None,
            "class_name": f"{base_class_name}{suffix}",
            "class_type": "object",
            "color": color,
            "norder": float(max_norder + i),
            "icon_url": None,
            "attributes": list(base_attrs) if base_attrs else [],
            "description": f"Phase-15 provenance: {suffix.lstrip('_')}",
            "use_description_as_prompt": False,
        }
        if use_models:
            new_entries.append(LabelClass(**payload))
        else:
            new_entries.append(payload)

    return list(existing) + new_entries


def classify_corrected_label(
    pre_class_name: str,
    post_class_name: str | None,
    was_moved: bool,
    was_deleted: bool,
) -> str:
    """Map a (pre, post) label pair back to a Phase-15 edit category.

    Implicit-action rule (no class re-typing required in CVAT):
      A: kept as iguana   — pred_only/borderline kept by reviewer
                            -> add to master as iguana_point
      B: false GT removed — gt_only/matched DELETED by reviewer
                            -> remove from master
      C: relocated        — pre <-> post position differs >1 px (still iguana)
      H: hard_negative    — pred_only/borderline DELETED by reviewer
                            OR pred_only/borderline KEPT with reviewer-set
                            non-iguana class (e.g. not_iguana_but_similar_look)
                            -> add to master at that position as a non-iguana
                            label (default: not_iguana_but_similar_look)

    Categories D (borderline) and E (confirmed FP) from the previous version
    collapse into A and H respectively under the simpler rule.
    """
    pre = pre_class_name.lower() if pre_class_name else ""

    if was_deleted:
        if pre.endswith(SUFFIX_GT_ONLY) or pre.endswith(SUFFIX_MATCHED):
            return "B"  # GT was wrong, remove
        if pre.endswith(SUFFIX_PRED_ONLY) or pre.endswith(SUFFIX_BORDERLINE):
            return "H"  # deleted model-suggested point = hard negative
        return "H"  # unknown provenance, treat conservatively as hard-neg

    # Kept (not deleted).
    # Explicit class re-typing to a non-iguana class also routes to H
    # (the optional path the reviewer can use to specify the negative type).
    if post_class_name and not is_iguana_class(post_class_name):
        return "H"

    if was_moved:
        return "C"
    if pre.endswith(SUFFIX_PRED_ONLY) or pre.endswith(SUFFIX_BORDERLINE):
        return "A"  # kept = iguana
    if pre.endswith(SUFFIX_GT_ONLY) or pre.endswith(SUFFIX_MATCHED):
        return "kept_unchanged"
    return "kept_unchanged"
    return "kept_unchanged"
