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


# Load FIFTYONE_CVAT_* before fiftyone is imported (helper.py imports it).
def _load_cvat_env() -> None:
    candidate = Path("/home/christian/hnee/active-learning/.env")
    if not candidate.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not available — CVAT credentials must be exported manually")
        return
    load_dotenv(candidate, override=False)


_load_cvat_env()

from active_learning.config.dataset_filter import DatasetCorrectionReportConfig  # noqa: E402
from com.biospheredata.types.HastyAnnotationV2 import (  # noqa: E402
    AnnotatedImage,
    HastyAnnotationV2,
    ImageLabel,
    Keypoint,
)

# scripts.human_in_the_loop.helper lives in the active-learning repo, which
# isn't pip-installed as a top-level package — add the repo root to
# sys.path so `from scripts...` resolves.
_AL_REPO = Path("/home/christian/hnee/active-learning")
if _AL_REPO.is_dir() and str(_AL_REPO) not in sys.path:
    sys.path.insert(0, str(_AL_REPO))

# The repo's hit_cvat_download() expects a cropped-tile workflow we don't
# use (it requires config.hA_prediction_tiled_path + per-crop metadata).
# We do the equivalent pull manually using the lower-level primitives.
from active_learning.reconstruct_hasty_annotation_cvat import (  # noqa: E402
    download_cvat_annotations,
)
import os as _os
import uuid as _uuid


_NON_IGUANA_CLASSES = {
    "not_iguana_but_similar_look",
    "ugly_stone",
    "fresh_lava",
    "trash",
    "hard_negative",
}


def _is_iguana_class(class_name: str) -> bool:
    """True if the label class represents an iguana (any of our point classes).

    Anything else (not_iguana_but_similar_look, ugly_stone, …) is treated as
    a hard-negative promotion.
    """
    if not class_name:
        return True  # default: treat empty/unknown as iguana
    cn = class_name.lower()
    if cn in _NON_IGUANA_CLASSES:
        return False
    return "iguana" in cn


def _completed_image_names(anno_results, dataset_name: str) -> set[str] | None:
    """Return the set of image filenames whose CVAT task state is 'completed'.

    Uses fiftyone's annotation backend get_status() to query CVAT for each
    task in the run, then maps task -> samples via frame_id_map. Returns
    None if status can't be fetched (then we fall back to processing
    everything — keeps backwards compatibility).
    """
    try:
        import fiftyone as fo
        ds = fo.load_dataset(dataset_name)
    except Exception as e:
        logger.warning(f"can't load fo dataset {dataset_name}: {e}")
        return None
    try:
        status = anno_results.get_status()
    except Exception as e:
        logger.warning(f"can't get CVAT task status: {e}")
        return None

    completed_task_ids: set[int] = set()
    for label_field, per_task in status.items():
        for task_id, info in per_task.items():
            tstatus = info.get("status", "") if isinstance(info, dict) else ""
            if str(tstatus).lower() in ("completed", "acceptance"):
                completed_task_ids.add(int(task_id))
    if not completed_task_ids:
        logger.warning("No CVAT tasks reported 'completed' — nothing to download.")
        return set()

    # frame_id_map: {task_id: {frame_id: {"sample_id": ..., "frame_id": ...}}}
    frame_id_map = getattr(anno_results, "frame_id_map", {}) or {}
    sample_ids: set[str] = set()
    for task_id in completed_task_ids:
        for fd in (frame_id_map.get(task_id, {}) or {}).values():
            sid = fd.get("sample_id")
            if sid:
                sample_ids.add(sid)

    # Look up filenames for those sample ids.
    completed_filenames: set[str] = set()
    if sample_ids:
        view = ds.select(list(sample_ids))
        for s in view:
            import os as _os
            completed_filenames.add(_os.path.basename(s.filepath))
    logger.info(
        f"CVAT completion filter: {len(completed_task_ids)} task(s) completed, "
        f"{len(completed_filenames)} image(s) available for download."
    )
    return completed_filenames


def _hit_cvat_download_whole_tile(
    report_config, only_completed: bool = True
) -> HastyAnnotationV2:
    """Pull CVAT corrections without the cropped-tile machinery.

    The active-learning foDataset2Hasty() has an UnboundLocalError on the
    whole-tile keypoint path (uses ``orgininal_image`` before assignment
    when samples lack hasty_image_id). We do the conversion directly: one
    fiftyone sample -> one AnnotatedImage, one fo.Keypoint -> one Hasty
    ImageLabel with a Keypoint at the denormalised pixel position.
    Downstream uses position-based matching to diff pre vs post, so we
    don't need to preserve label IDs across the round trip.
    """
    hA_pre = HastyAnnotationV2.from_file(report_config.hA_prediction_path)
    _view, dataset = download_cvat_annotations(dataset_name=report_config.dataset_name)

    # Optional filter: only include samples whose CVAT task is in state
    # "completed" — lets partial reviews not pollute the master.
    completed_filenames: set[str] | None = None
    if only_completed:
        try:
            import fiftyone as fo
            ds = fo.load_dataset(report_config.dataset_name)
            anno_results = ds.load_annotation_results(report_config.dataset_name)
            completed_filenames = _completed_image_names(
                anno_results, report_config.dataset_name
            )
        except Exception as e:
            logger.warning(
                f"can't determine CVAT task completion (will process all): {e}"
            )

    pre_by_name = {img.image_name: img for img in hA_pre.images}
    keypoint_class_id = "body"
    for schema in (hA_pre.keypoint_schemas or []):
        for kp_class in (schema.keypoint_classes or []):
            if "body" in kp_class.keypoint_class_name.lower():
                keypoint_class_id = kp_class.keypoint_class_id
                break

    post_images: list[AnnotatedImage] = []
    for sample in dataset:
        image_filename = _os.path.basename(sample.filepath)
        if completed_filenames is not None and image_filename not in completed_filenames:
            continue  # skip samples whose CVAT task isn't completed
        width = height = 0
        if hasattr(sample, "metadata") and sample.metadata is not None:
            width = sample.metadata.width or 0
            height = sample.metadata.height or 0
        if (width <= 0 or height <= 0) and image_filename in pre_by_name:
            width = pre_by_name[image_filename].width or width
            height = pre_by_name[image_filename].height or height
        if width <= 0 or height <= 0:
            logger.warning(f"no width/height for {image_filename}, skipping")
            continue

        labels: list[ImageLabel] = []
        keypoints_field = getattr(sample, "detection", None)
        if keypoints_field is not None:
            for kp in (keypoints_field.keypoints or []):
                if not kp.points:
                    continue
                nx, ny = kp.points[0]
                x = int(round(nx * width))
                y = int(round(ny * height))
                hasty_kp = Keypoint(
                    x=x, y=y, norder=0, keypoint_class_id=keypoint_class_id
                )
                # Recover the original Hasty label.id round-tripped via
                # the kp["hasty_id"] custom attribute set at upload time
                # (see 100_HIT_phase15_upload.py). If absent — e.g. a
                # reviewer drew a brand-new label — fall back to a fresh
                # uuid; downstream matching will detect it as new via the
                # position-fallback pass.
                hasty_id = (
                    getattr(kp, "hasty_id", None)
                    or (kp.attributes.get("hasty_id") if hasattr(kp, "attributes") and kp.attributes else None)
                    or str(_uuid.uuid4())
                )
                labels.append(ImageLabel(
                    id=str(hasty_id),
                    class_name=kp.label,
                    keypoints=[hasty_kp],
                    attributes={"cvat": "downloaded"},
                ))

        image_id = (pre_by_name[image_filename].image_id
                    if image_filename in pre_by_name else str(_uuid.uuid4()))
        dataset_name_for_image = (pre_by_name[image_filename].dataset_name
                                  if image_filename in pre_by_name else None)
        post_images.append(AnnotatedImage(
            image_id=image_id,
            image_name=image_filename,
            dataset_name=dataset_name_for_image,
            ds_image_name=None,
            width=width,
            height=height,
            image_status="DONE",
            tags=[],
            image_mode=None,
            labels=labels,
        ))

    hA_post = hA_pre.copy(deep=True)
    hA_post.images = post_images
    # Restrict pre to the same image set so apply_corrections_to_master
    # doesn't see "pre had this image, post doesn't" for un-reviewed tiles.
    filtered_pre: HastyAnnotationV2 | None = None
    if completed_filenames is not None:
        filtered_pre = hA_pre.copy(deep=True)
        filtered_pre.images = [
            img for img in filtered_pre.images if img.image_name in completed_filenames
        ]
    return hA_post, filtered_pre

# Local helper (in this directory).
sys.path.append(str(Path(__file__).parent))
from hit_phase15_merge import (
    classify_corrected_label,
    is_iguana_class,
    SUFFIX_GT_ONLY,
    SUFFIX_PRED_ONLY,
    SUFFIX_MATCHED,
    SUFFIX_BORDERLINE,
    CANONICAL_CLASS_NAME,
)


EDIT_LOG_COLUMNS = [
    "timestamp",
    "iteration_id",
    "image",
    "site",
    "label_id",
    "pre_class_name",     # what the reviewer saw on the label at upload
    "post_class_name",    # what came back from CVAT (may be different if reviewer re-classed)
    "category",           # A / B / C / D / E / H / kept_unchanged
    "fate",               # human-readable: kept_as_iguana / false_positive / hard_negative / ...
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
    H: int = 0  # hard-negative promotion
    kept_unchanged: int = 0


def _bump(counts: "CategoryCounts", category: str) -> None:
    """Increment the counter for category — falls back to kept_unchanged."""
    attr = category if category in ("A", "B", "C", "D", "E", "H") else "kept_unchanged"
    setattr(counts, attr, getattr(counts, attr) + 1)


# Map kept-pre-label categories to a per-pre-label "fate" recorded in the
# edit log so an analyst can later filter on these directly.
_CATEGORY_TO_FATE = {
    "A": "kept_as_iguana",       # pred_only kept as iguana — promoted to GT
    "B": "false_gt_removed",      # gt/matched deleted — GT was wrong
    "C": "relocated",              # same label moved
    "D": "kept_borderline",       # borderline kept — needs second opinion
    "E": "false_positive",        # pred_only deleted — model was wrong
    "H": "hard_negative",         # pred_only kept as non-iguana — promoted to negatives
    "kept_unchanged": "kept_unchanged",
}


def _greedy_position_match(
    pre_labels: list[ImageLabel],
    post_labels: list[ImageLabel],
    same_label_radius: float = 50.0,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Hybrid pre <-> post matching: id-first, position-fallback.

    Pass 1: match labels where ``pre.id == post.id``. From iteration 1
    onwards the upload stores the Hasty label.id as a CVAT attribute
    ``hasty_id`` that round-trips through fiftyone, so kept labels keep
    their identity even if relocated arbitrarily far.

    Pass 2: greedy nearest-neighbour position match on the remaining
    labels within ``same_label_radius`` px. Catches labels that lost
    their hasty_id (legacy iteration-0 data, CVAT quirks, etc.) and
    keeps the diff working without the id round trip.

    Returns (matched, unmatched_pre, unmatched_post). matched is
    [(pre_idx, post_idx, distance), ...].
    """
    used_pre: set[int] = set()
    used_post: set[int] = set()
    matched: list[tuple[int, int, float]] = []

    pre_xy = [_keypoint_xy(l) for l in pre_labels]
    post_xy = [_keypoint_xy(l) for l in post_labels]

    # --- Pass 1: id-based ---
    post_by_id = {l.id: j for j, l in enumerate(post_labels) if l.id}
    for i, pre in enumerate(pre_labels):
        if not pre.id or pre.id not in post_by_id:
            continue
        j = post_by_id[pre.id]
        if j in used_post:
            continue
        px, py = pre_xy[i]
        qx, qy = post_xy[j]
        d = (math.hypot(px - qx, py - qy)
             if None not in (px, py, qx, qy) else 0.0)
        used_pre.add(i)
        used_post.add(j)
        matched.append((i, j, d))

    # --- Pass 2: position fallback on remaining ---
    candidates: list[tuple[float, int, int]] = []
    for i, (px, py) in enumerate(pre_xy):
        if i in used_pre or px is None:
            continue
        for j, (qx, qy) in enumerate(post_xy):
            if j in used_post or qx is None:
                continue
            d = math.hypot(px - qx, py - qy)
            if d <= same_label_radius:
                candidates.append((d, i, j))
    candidates.sort()
    for d, i, j in candidates:
        if i in used_pre or j in used_post:
            continue
        used_pre.add(i)
        used_post.add(j)
        matched.append((i, j, d))

    unmatched_pre = [i for i in range(len(pre_labels)) if i not in used_pre]
    unmatched_post = [j for j in range(len(post_labels)) if j not in used_post]
    return matched, unmatched_pre, unmatched_post


def apply_corrections_to_master(
    hA_pre: HastyAnnotationV2,
    hA_post: HastyAnnotationV2,
    hA_master: HastyAnnotationV2,
    iteration_id: str,
    edit_log_path: Path,
    tile_to_master_image: "dict[str, AnnotatedImage] | None" = None,
) -> tuple[HastyAnnotationV2, CategoryCounts]:
    """Compute deltas between pre and post and apply to the master Hasty.

    Uses POSITION-BASED matching, not label-id matching: CVAT/foDataset2Hasty
    assigns fresh UUIDs to every label on download, so the pre-correction
    label.id doesn't survive the round trip. Instead we Hungarian-match
    pre to post per image by keypoint position within a 50 px radius.
    Pairs in radius → kept (possibly moved). Unpaired pre → deleted.
    Unpaired post → newly drawn.

    hA_pre: the intermediate Hasty we uploaded (with provenance class names).
    hA_post: what came back from CVAT after correction.
    hA_master: the production GT to update. We modify a deep copy.
    """
    pre_by_image: dict[str, list[ImageLabel]] = {
        img.image_name: list(img.labels) for img in hA_pre.images
    }
    post_by_image: dict[str, list[ImageLabel]] = {
        img.image_name: list(img.labels) for img in hA_post.images
    }

    hA_master = copy.deepcopy(hA_master)
    if tile_to_master_image is None:
        # Fallback: lookup by image_name only (works if tile_name == master image_name).
        master_image_by_name = {img.image_name: img for img in hA_master.images}
    else:
        # Use the upload's tile -> master mapping so edits route to the right
        # Hasty record even when names collide across datasets.
        master_image_by_name = dict(tile_to_master_image)

    counts = CategoryCounts()
    edit_log_rows: list[dict] = []

    all_image_names = set(pre_by_image) | set(post_by_image)
    for image_name in sorted(all_image_names):
        pre_labels = pre_by_image.get(image_name, [])
        post_labels = post_by_image.get(image_name, [])
        site = _site_of(image_name)

        matched, unmatched_pre, unmatched_post = _greedy_position_match(
            pre_labels, post_labels, same_label_radius=50.0
        )

        # KEPT (matched): apply category based on pre's provenance suffix
        # and whether the position moved >1 px and whether the class changed.
        for pre_idx, post_idx, distance in matched:
            pre_label = pre_labels[pre_idx]
            post_label = post_labels[post_idx]
            pre_xy = _keypoint_xy(pre_label)
            post_xy = _keypoint_xy(post_label)
            was_moved = distance > 1.0
            category = classify_corrected_label(
                pre_class_name=pre_label.class_name,
                post_class_name=post_label.class_name,
                was_moved=was_moved,
                was_deleted=False,
            )

            score = (pre_label.attributes or {}).get("score") if pre_label.attributes else None
            edit_log_rows.append({
                "timestamp": datetime.utcnow().isoformat(),
                "iteration_id": iteration_id,
                "image": image_name,
                "site": site,
                "label_id": pre_label.id,
                "pre_class_name": pre_label.class_name,
                "post_class_name": post_label.class_name,
                "category": category,
                "fate": _CATEGORY_TO_FATE.get(category, category),
                "x_old": pre_xy[0], "y_old": pre_xy[1],
                "x_new": post_xy[0], "y_new": post_xy[1],
                "score": score,
                "notes": "",
            })
            _bump(counts, category)
            if image_name not in master_image_by_name:
                continue
            master_img = master_image_by_name[image_name]
            if category == "C":
                _relocate_keypoint_near(master_img, pre_xy, post_xy, radius=10)
            elif category == "A":
                _add_keypoint(master_img, post_xy, class_name=CANONICAL_CLASS_NAME)
            elif category == "H":
                # Hard-negative promotion: add label to master under the
                # non-iguana class the reviewer chose. Future training sees
                # an explicit negative at this position.
                _add_keypoint(master_img, post_xy, class_name=post_label.class_name)

        # DELETED (pre with no post match): category B / E / D based on suffix.
        for i in unmatched_pre:
            pre_label = pre_labels[i]
            pre_xy = _keypoint_xy(pre_label)
            category = classify_corrected_label(
                pre_class_name=pre_label.class_name,
                post_class_name=None,
                was_moved=False,
                was_deleted=True,
            )
            score = (pre_label.attributes or {}).get("score") if pre_label.attributes else None
            edit_log_rows.append({
                "timestamp": datetime.utcnow().isoformat(),
                "iteration_id": iteration_id,
                "image": image_name,
                "site": site,
                "label_id": pre_label.id,
                "pre_class_name": pre_label.class_name,
                "post_class_name": None,
                "category": category,
                "fate": _CATEGORY_TO_FATE.get(category, category),
                "x_old": pre_xy[0], "y_old": pre_xy[1],
                "x_new": None, "y_new": None,
                "score": score,
                "notes": "deleted_in_cvat",
            })
            _bump(counts, category)
            if image_name in master_image_by_name:
                if category == "B":
                    _remove_keypoint_near(master_image_by_name[image_name], pre_xy, radius=10)
                elif category == "H":
                    # Deleted pred/borderline -> hard negative example. Add to
                    # master at the deleted position as not_iguana_but_similar_look.
                    _add_keypoint(
                        master_image_by_name[image_name],
                        pre_xy,
                        class_name="not_iguana_but_similar_look",
                    )

        # NEWLY DRAWN (post with no pre match). A or H based on class.
        for j in unmatched_post:
            post_label = post_labels[j]
            post_xy = _keypoint_xy(post_label)
            is_iguana = is_iguana_class(post_label.class_name)
            category = "A" if is_iguana else "H"
            edit_log_rows.append({
                "timestamp": datetime.utcnow().isoformat(),
                "iteration_id": iteration_id,
                "image": image_name,
                "site": site,
                "label_id": post_label.id,
                "pre_class_name": None,
                "post_class_name": post_label.class_name,
                "category": category,
                "fate": _CATEGORY_TO_FATE.get(category, category),
                "x_old": None, "y_old": None,
                "x_new": post_xy[0], "y_new": post_xy[1],
                "score": None,
                "notes": "newly_drawn",
            })
            _bump(counts, category)
            if image_name in master_image_by_name:
                target_class = (
                    CANONICAL_CLASS_NAME if is_iguana else post_label.class_name
                )
                _add_keypoint(
                    master_image_by_name[image_name], post_xy, class_name=target_class
                )

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
    hA_post, hA_pre_filtered = _hit_cvat_download_whole_tile(report_config, only_completed=True)

    # Pre-correction intermediate Hasty (the file we uploaded). When CVAT
    # task-completion filtering was applied, the filter returned a Hasty
    # restricted to only the completed images; use it instead of the file.
    if hA_pre_filtered is not None:
        hA_pre = hA_pre_filtered
        logger.info(
            f"Using completion-filtered pre-Hasty: {len(hA_pre.images)} images "
            f"(unreviewed images are not in the diff)."
        )
    else:
        logger.info(f"Loading pre-correction Hasty: {report_config.hA_prediction_path}")
        hA_pre = HastyAnnotationV2.from_file(report_config.hA_prediction_path)

    logger.info(f"Loading master Hasty to update: {master_hasty_path}")
    hA_master = HastyAnnotationV2.from_file(master_hasty_path)

    # Load the tile -> (hasty_dataset, hasty_image_name) mapping written
    # by the upload step. This is the only reliable way to route a tile's
    # corrections to the right master Hasty image (the tile uses a
    # <dataset>___<filename> prefixed name while Hasty stores bare names).
    import json as _json
    report_dir = Path(report_path).parent if Path(report_path).is_file() else Path(report_path)
    mapping_paths = list(report_dir.glob("*_image_to_dataset.json"))
    if not mapping_paths:
        raise RuntimeError(
            f"No *_image_to_dataset.json mapping file in {report_dir}. The "
            f"upload step writes this; without it we can't route edits to "
            f"the master."
        )
    mapping = _json.loads(mapping_paths[0].read_text())
    logger.info(f"Loaded tile->Hasty mapping from {mapping_paths[0]} "
                f"({len(mapping)} entries)")

    # Resolve tile_name -> master AnnotatedImage via (dataset, hasty_name).
    master_by_dataset_and_name: dict[tuple[str, str], AnnotatedImage] = {}
    for img in hA_master.images:
        if img.image_name and img.dataset_name:
            master_by_dataset_and_name.setdefault(
                (img.dataset_name, img.image_name), img
            )
    tile_to_master_image: dict[str, AnnotatedImage] = {}
    unmatched_tiles: list[str] = []
    for tile_name, route in mapping.items():
        key = (route["dataset_name"], route["hasty_image_name"])
        master_img = master_by_dataset_and_name.get(key)
        if master_img is None:
            unmatched_tiles.append(tile_name)
        else:
            tile_to_master_image[tile_name] = master_img

    if unmatched_tiles:
        logger.error(
            f"{len(unmatched_tiles)} mapping entries could not be resolved "
            f"to a master Hasty image — edits for these would be silently "
            f"lost. Has the master Hasty changed since upload?"
        )
        for t in unmatched_tiles[:10]:
            logger.error(f"  unresolved: {t}")
        raise RuntimeError(
            "Master Hasty has lost images referenced in the upload. Inspect "
            f"{master_hasty_path} and confirm the right file is being updated."
        )
    logger.info(
        f"Resolved {len(tile_to_master_image)} of {len(mapping)} tile names "
        f"to master Hasty records."
    )

    hA_master_updated, counts = apply_corrections_to_master(
        hA_pre=hA_pre,
        hA_post=hA_post,
        hA_master=hA_master,
        iteration_id=iteration_id,
        edit_log_path=edit_log_path,
        tile_to_master_image=tile_to_master_image,
    )

    output_master_hasty_path.parent.mkdir(parents=True, exist_ok=True)
    hA_master_updated.save(output_master_hasty_path)
    logger.info(f"Wrote updated master Hasty: {output_master_hasty_path}")
    logger.info(
        f"Edits for iteration {iteration_id}: "
        f"A={counts.A} (kept as iguana / promoted to GT), "
        f"B={counts.B} (false GT removed), "
        f"C={counts.C} (relocated), "
        f"D={counts.D} (borderline — second opinion), "
        f"E={counts.E} (confirmed FPs — no master change), "
        f"H={counts.H} (hard-negative promoted), "
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
