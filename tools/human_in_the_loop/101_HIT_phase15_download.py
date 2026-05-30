"""
Phase 15 — download CVAT corrections and apply them to the Hasty master.

What this does:
1. Loads the report.json written by ``100_HIT_phase15_upload.py``.
2. Pulls the corrected annotations from CVAT (delegates to the existing
   ``hit_cvat_download`` helper in the active-learning repo).
3. Compares each post-correction label against the pre-correction
   intermediate Hasty (by id) to determine: deleted? moved? kept?
4. Routes each correction into a Phase-15 edit category (A/B/C/D/E),
   updates the master Hasty annotation, and appends rows to the
   edit log CSV (audit trail).
5. Writes:
     - corrected Hasty annotation (the new master)
     - edit log CSV (append-only)
     - per-iteration summary report

Companion script: ``100_HIT_phase15_upload.py``.
Helper module: ``hit_phase15_merge.py``.
Plan: ``docs/phase15_annotation_cleanup_loop.md`` (this repo).

Categories from the plan doc:
  A: missed iguana    — pred_only kept by reviewer  -> add to GT
  B: false GT          — gt_only deleted by reviewer -> remove from GT
  C: relocation       — any provenance moved >0 px  -> update (x, y)
  D: borderline       — borderline kept              -> flag for second opinion
  E: confirmed FP     — pred_only deleted            -> log failure mode
"""

from __future__ import annotations

import copy
import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from active_learning.config.dataset_filter import DatasetCorrectionReportConfig
from com.biospheredata.types.HastyAnnotationV2 import (
    AnnotatedImage,
    HastyAnnotationV2,
    ImageLabel,
    Keypoint,
)

# Re-use the existing CVAT download path. This wraps cvat2hasty + foDataset2Hasty.
from scripts.human_in_the_loop.helper import hit_cvat_download  # type: ignore

# Local helper (in this directory).
sys.path.append(str(Path(__file__).parent))
from hit_phase15_merge import (
    classify_corrected_label,
    SUFFIX_GT_ONLY,
    SUFFIX_PRED_ONLY,
    SUFFIX_MATCHED,
    SUFFIX_BORDERLINE,
)


EDIT_LOG_COLUMNS = [
    "timestamp",
    "iteration_id",
    "image",
    "site",
    "label_id",
    "class_name",
    "category",  # A / B / C / D / E / kept_unchanged
    "x_old", "y_old",
    "x_new", "y_new",
    "score",
    "notes",
]


def _site_of(image: str) -> str:
    return image.split("___", 1)[0] if "___" in image else "UNKNOWN"


def _keypoint_xy(label: ImageLabel) -> tuple[float | None, float | None]:
    if label.keypoints:
        kp = label.keypoints[0]
        return float(kp.x), float(kp.y)
    if label.incenter_centroid is not None:
        return float(label.incenter_centroid.x), float(label.incenter_centroid.y)
    return None, None


def _index_by_id(hA: HastyAnnotationV2) -> dict[str, ImageLabel]:
    out: dict[str, ImageLabel] = {}
    for img in hA.images:
        for label in img.labels:
            if label.id:
                out[label.id] = label
    return out


def _moved(p_pre: tuple[float | None, float | None],
           p_post: tuple[float | None, float | None],
           tolerance_px: float = 1.0) -> bool:
    if None in p_pre or None in p_post:
        return False
    return math.hypot(p_pre[0] - p_post[0], p_pre[1] - p_post[1]) > tolerance_px


@dataclass
class CategoryCounts:
    A: int = 0
    B: int = 0
    C: int = 0
    D: int = 0
    E: int = 0
    kept_unchanged: int = 0


def apply_corrections_to_master(
    hA_pre: HastyAnnotationV2,
    hA_post: HastyAnnotationV2,
    hA_master: HastyAnnotationV2,
    iteration_id: str,
    edit_log_path: Path,
) -> tuple[HastyAnnotationV2, CategoryCounts]:
    """Compute deltas between pre and post and apply to the master Hasty.

    hA_pre: the intermediate Hasty we uploaded (with provenance class names).
    hA_post: what came back from CVAT after correction.
    hA_master: the production GT to update. We modify a deep copy.
    """
    pre_index = _index_by_id(hA_pre)
    post_index = _index_by_id(hA_post)

    pre_ids = set(pre_index.keys())
    post_ids = set(post_index.keys())
    deleted_ids = pre_ids - post_ids
    surviving_ids = pre_ids & post_ids
    new_ids = post_ids - pre_ids  # reviewer drew a brand-new label not in pre

    hA_master = copy.deepcopy(hA_master)
    master_image_by_name = {img.image_name: img for img in hA_master.images}

    counts = CategoryCounts()
    edit_log_rows: list[dict] = []

    # ---- Pre-existing labels: deleted or kept (possibly moved) ----
    for label_id in pre_ids:
        pre_label = pre_index[label_id]
        pre_xy = _keypoint_xy(pre_label)
        # Find which image this label belongs to (we have to scan hA_pre).
        image_name = next(
            (img.image_name for img in hA_pre.images if any(l.id == label_id for l in img.labels)),
            None,
        )
        site = _site_of(image_name) if image_name else "UNKNOWN"

        if label_id in deleted_ids:
            category = classify_corrected_label(pre_label.class_name, was_moved=False, was_deleted=True)
            x_new, y_new = None, None
        else:
            post_label = post_index[label_id]
            post_xy = _keypoint_xy(post_label)
            was_moved = _moved(pre_xy, post_xy)
            category = classify_corrected_label(post_label.class_name, was_moved=was_moved, was_deleted=False)
            x_new, y_new = post_xy

        setattr(counts, category if category in ("A","B","C","D","E") else "kept_unchanged",
                getattr(counts, category if category in ("A","B","C","D","E") else "kept_unchanged") + 1)

        score = (pre_label.attributes or {}).get("score") if pre_label.attributes else None
        edit_log_rows.append({
            "timestamp": datetime.utcnow().isoformat(),
            "iteration_id": iteration_id,
            "image": image_name,
            "site": site,
            "label_id": label_id,
            "class_name": pre_label.class_name,
            "category": category,
            "x_old": pre_xy[0], "y_old": pre_xy[1],
            "x_new": x_new, "y_new": y_new,
            "score": score,
            "notes": "",
        })

        # Apply to master.
        if image_name is None or image_name not in master_image_by_name:
            continue
        master_img = master_image_by_name[image_name]

        if category == "B":  # remove from GT
            _remove_keypoint_near(master_img, pre_xy, radius=10)
        elif category == "C":  # relocate
            _relocate_keypoint_near(master_img, pre_xy, (x_new, y_new), radius=10)
        elif category == "A":  # missed iguana — add to master (pre_label was pred_only)
            _add_keypoint(master_img, (x_new, y_new), class_name="iguana")

    # ---- Brand-new labels the reviewer drew that weren't in the upload ----
    for label_id in new_ids:
        post_label = post_index[label_id]
        x, y = _keypoint_xy(post_label)
        image_name = next(
            (img.image_name for img in hA_post.images if any(l.id == label_id for l in img.labels)),
            None,
        )
        site = _site_of(image_name) if image_name else "UNKNOWN"
        edit_log_rows.append({
            "timestamp": datetime.utcnow().isoformat(),
            "iteration_id": iteration_id,
            "image": image_name,
            "site": site,
            "label_id": label_id,
            "class_name": post_label.class_name,
            "category": "A",  # newly drawn iguana == missed annotation
            "x_old": None, "y_old": None,
            "x_new": x, "y_new": y,
            "score": None,
            "notes": "newly_drawn",
        })
        counts.A += 1
        if image_name and image_name in master_image_by_name:
            _add_keypoint(master_image_by_name[image_name], (x, y), class_name="iguana")

    # ---- Append edit log ----
    edit_log_path.parent.mkdir(parents=True, exist_ok=True)
    file_existed = edit_log_path.exists() and edit_log_path.stat().st_size > 0
    with edit_log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EDIT_LOG_COLUMNS)
        if not file_existed:
            writer.writeheader()
        for row in edit_log_rows:
            writer.writerow(row)

    return hA_master, counts


def _remove_keypoint_near(image: AnnotatedImage, xy: tuple[float, float], radius: float) -> None:
    if xy[0] is None:
        return
    new_labels = []
    removed = 0
    for l in image.labels:
        lx, ly = _keypoint_xy(l)
        if lx is None or removed >= 1:
            new_labels.append(l); continue
        if math.hypot(lx - xy[0], ly - xy[1]) <= radius:
            removed += 1
            continue
        new_labels.append(l)
    image.labels = new_labels


def _relocate_keypoint_near(image: AnnotatedImage, xy_old: tuple[float, float],
                            xy_new: tuple[float, float], radius: float) -> None:
    if xy_old[0] is None or xy_new[0] is None:
        return
    for l in image.labels:
        lx, ly = _keypoint_xy(l)
        if lx is None:
            continue
        if math.hypot(lx - xy_old[0], ly - xy_old[1]) <= radius:
            if l.keypoints:
                l.keypoints[0].x = int(round(xy_new[0]))
                l.keypoints[0].y = int(round(xy_new[1]))
            return


def _add_keypoint(image: AnnotatedImage, xy: tuple[float, float], class_name: str) -> None:
    if xy[0] is None:
        return
    import uuid
    kp = Keypoint(
        id=str(uuid.uuid4()),
        x=int(round(xy[0])),
        y=int(round(xy[1])),
        keypoint_class_id="body",
    )
    image.labels.append(ImageLabel(
        id=str(uuid.uuid4()),
        class_name=class_name,
        keypoints=[kp],
        attributes={"source": "phase15_review"},
    ))


def phase15_cvat_download_and_update(
    report_path: Path,
    master_hasty_path: Path,
    output_master_hasty_path: Path,
    edit_log_path: Path,
    iteration_id: str | None = None,
) -> CategoryCounts:
    """Download corrections from CVAT and apply to the master Hasty.

    report_path: report.json produced by 100_HIT_phase15_upload.py.
    master_hasty_path: current production Hasty GT to update.
    output_master_hasty_path: where the updated Hasty is written.
    edit_log_path: append-only CSV audit trail (one row per edit).
    iteration_id: string used in the edit log; defaults to the report's
        analysis_date.
    """
    report_config = DatasetCorrectionReportConfig.load(report_path)
    iteration_id = iteration_id or report_config.analysis_date

    logger.info(f"Pulling reviewed annotations from CVAT for dataset {report_config.dataset_name}")
    hA_post = hit_cvat_download(report_path)

    # Pre-correction intermediate Hasty (the file we uploaded).
    logger.info(f"Loading pre-correction Hasty: {report_config.hA_prediction_path}")
    hA_pre = HastyAnnotationV2.from_file(report_config.hA_prediction_path)

    logger.info(f"Loading master Hasty to update: {master_hasty_path}")
    hA_master = HastyAnnotationV2.from_file(master_hasty_path)

    hA_master_updated, counts = apply_corrections_to_master(
        hA_pre=hA_pre,
        hA_post=hA_post,
        hA_master=hA_master,
        iteration_id=iteration_id,
        edit_log_path=edit_log_path,
    )

    output_master_hasty_path.parent.mkdir(parents=True, exist_ok=True)
    hA_master_updated.save(output_master_hasty_path)
    logger.info(f"Wrote updated master Hasty: {output_master_hasty_path}")
    logger.info(
        f"Edits for iteration {iteration_id}: "
        f"A={counts.A} (missed iguanas added), "
        f"B={counts.B} (false GT removed), "
        f"C={counts.C} (relocated), "
        f"D={counts.D} (borderline — second opinion needed), "
        f"E={counts.E} (confirmed FPs), "
        f"kept_unchanged={counts.kept_unchanged}"
    )
    logger.info(f"Edit log appended at: {edit_log_path}")
    return counts


if __name__ == "__main__":
    # Example for Phase-15 iteration 0 (the seed pass on val).
    base = Path("/home/christian/data/training_data/2026_05_08_data_scaling/val/phase15_iter0")
    report_path = base / "report"

    # The current production Hasty for the val tiles.
    master_hasty_path = Path(
        "/home/christian/data/training_data/2026_05_08_data_scaling/val/hasty_format_full_size.json"
    )
    output_master_hasty_path = (
        master_hasty_path.parent / "hasty_format_full_size_phase15_iter0_corrected.json"
    )
    edit_log_path = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")

    phase15_cvat_download_and_update(
        report_path=report_path,
        master_hasty_path=master_hasty_path,
        output_master_hasty_path=output_master_hasty_path,
        edit_log_path=edit_log_path,
        iteration_id="phase15_iter0_val",
    )
