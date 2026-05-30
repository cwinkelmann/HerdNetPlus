"""
Phase 15 — upload merged GT + predictions to CVAT for human correction.

What this does:
1. Loads herdnet-format GT CSV + detections.csv produced by stitched inference.
2. For each image, Hungarian-matches GT vs predictions (radius-constrained).
3. Tags each point with its provenance:
     - matched (GT had a prediction nearby — usually skipped unless full review)
     - gt_only (GT with no near prediction — possible false GT [B])
     - pred_only (prediction with no near GT — possible missed iguana [A])
     - borderline (Hungarian assigned but distance > radius)
4. Writes an intermediate HastyAnnotationV2 with provenance encoded in
   class names, plus a DatasetCorrectionReportConfig pointing at it.
5. Uploads to CVAT via the same fiftyone + submit_for_cvat_evaluation
   path the existing 060/061 HIT scripts use.

Then a human reviews in CVAT; ``101_HIT_phase15_download.py`` pulls the
result and routes edits into the Phase-15 edit log.

Companion script: ``101_HIT_phase15_download.py``.
Helper module: ``hit_phase15_merge.py``.
Plan: ``docs/phase15_annotation_cleanup_loop.md`` (this repo).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pandas as pd
import PIL.Image
from loguru import logger


# Load FIFTYONE_CVAT_URL / USERNAME / PASSWORD from the active-learning
# repo's .env *before* fiftyone imports so the CVAT integration sees them.
# Falls back silently if the file isn't present (the user can also export
# the vars manually).
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
    for k in ("FIFTYONE_CVAT_URL", "FIFTYONE_CVAT_USERNAME"):
        if k in os.environ:
            logger.info(f"loaded {k} from {candidate}")


_load_cvat_env()

from active_learning.config.dataset_filter import (  # noqa: E402
    DatasetCorrectionConfig,
    DatasetCorrectionReportConfig,
)
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2  # noqa: E402

import fiftyone as fo  # noqa: E402

# CVAT project name — every Phase-15 task lands in the same project so
# the reviewer can find them all in one place.
CVAT_PROJECT_NAME = "Hasty_Corr"
CVAT_ORGANIZATION = "IguanasFromAbove"

# Local helper (in this directory). When running directly add this dir to PYTHONPATH
# or set `python -m scripts.human_in_the_loop.100_HIT_phase15_upload` semantics.
import sys
sys.path.append(str(Path(__file__).parent))
from hit_phase15_merge import (
    merge_dataset,
    merge_summary,
    merge_results_to_hasty,
    PROVENANCE_COLORS,
    SUFFIX_GT_ONLY,
    SUFFIX_PRED_ONLY,
    SUFFIX_MATCHED,
    SUFFIX_BORDERLINE,
    CANONICAL_CLASS_NAME,
)


def _preflight_match_to_hasty_datasets(
    image_names: list[str], hA_master: HastyAnnotationV2
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Match Phase-13 tile filenames to Hasty (dataset_name, image_name).

    Phase-13 val/test tiles use a prefixed naming convention:
        ``<hasty_dataset_name>___<hasty_image_name>``
    while the Hasty master stores just ``<hasty_image_name>`` with
    ``dataset_name`` carried separately. We split on the ``___`` separator
    and verify (dataset_name, image_name) is in the master.

    Returns ({tile_name -> {"dataset_name": ..., "hasty_image_name": ...}},
             missing). The download step uses this mapping to find the
    right master Hasty image to update — including when two datasets share
    the same image_name (e.g. multiple sites with "DJI_0141.JPG").
    """
    by_dataset_and_name: set[tuple[str, str]] = set()
    by_name_only: dict[str, str] = {}
    for img in hA_master.images:
        if not img.image_name or not img.dataset_name:
            continue
        by_dataset_and_name.add((img.dataset_name, img.image_name))
        # Fallback for tile names without a ___ prefix: first hit wins.
        by_name_only.setdefault(img.image_name, img.dataset_name)

    mapping: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for tile_name in image_names:
        # 1. Try the prefixed convention <dataset>___<image>.
        if "___" in tile_name:
            ds, hname = tile_name.split("___", 1)
            if (ds, hname) in by_dataset_and_name:
                mapping[tile_name] = {"dataset_name": ds, "hasty_image_name": hname}
                continue
        # 2. Fallback: lookup by image_name only.
        ds = by_name_only.get(tile_name)
        if ds is not None:
            mapping[tile_name] = {"dataset_name": ds, "hasty_image_name": tile_name}
            continue
        missing.append(tile_name)
    return mapping, missing


def _build_fiftyone_keypoints(
    hA_image, base_class_name: str
) -> "fo.Keypoints":
    """Convert one AnnotatedImage's labels to a fiftyone Keypoints field.

    Coordinates are normalised to [0, 1] using the image width/height
    stored on the AnnotatedImage. If width/height aren't set we fall back
    to reading the file (slower).
    """
    width = float(getattr(hA_image, "width", 0) or 0)
    height = float(getattr(hA_image, "height", 0) or 0)
    keypoints: list[fo.Keypoint] = []
    for label in hA_image.labels:
        for kp in label.keypoints or []:
            if width <= 0 or height <= 0:
                continue  # caller should compute_metadata() after add
            nx = max(0.0, min(1.0, kp.x / width))
            ny = max(0.0, min(1.0, kp.y / height))
            # Score is preserved in the intermediate Hasty file's
            # ImageLabel.attributes — we don't push it into fiftyone here
            # (fo.Keypoint's `confidence` field expects a list-per-point,
            # not a scalar, and the score isn't load-bearing for the
            # human review).
            kp = fo.Keypoint(
                label=label.class_name,
                points=[(nx, ny)],
            )
            # Store the Hasty label id as a custom field on the fo.Keypoint.
            # fiftyone allows arbitrary attributes on embedded docs and
            # preserves them across the CVAT round trip — the download step
            # reads `getattr(kp, "hasty_id", None)` to recover the original
            # label.id. This makes id-based pre <-> post matching reliable.
            if label.id:
                kp["hasty_id"] = str(label.id)
            keypoints.append(kp)
    return fo.Keypoints(keypoints=keypoints)


def phase15_cvat_upload(
    config: DatasetCorrectionConfig,
    report_path: Path,
    pred_score_threshold: float = 0.5,
    include_matched: bool = False,
    only_fp_images: bool = True,
    base_class_name: str = "iguana_point",
) -> DatasetCorrectionReportConfig:
    """Hungarian-merge GT+predictions and push to CVAT for review.

    config: standard DatasetCorrectionConfig pointing at
            - herdnet_annotation_name (GT csv)
            - hasty_ground_truth_annotation_name (Hasty GT json)
            - detections_path (model output csv)
            - hasty_reference_annotation_name (for label_classes)
            - radius (match radius in pixels)
            - images_path, output_path, corrected_path
    report_path: where the report.json is written for the download step.
    pred_score_threshold: drop predictions with score < this. 0.5 keeps the
            tail; for the Phase-15 loop default 0.15 (more candidate FPs to
            review) is typical. Tune per iteration.
    include_matched: if True also push agreed-upon GT points (full review).
            If False (default) only push points that disagree — much shorter
            review queue per iteration.

    Returns the DatasetCorrectionReportConfig pointing at the intermediate
    Hasty file, which ``101_HIT_phase15_download.py`` consumes.
    """
    config.output_path.mkdir(parents=True, exist_ok=True)
    config.corrected_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)
    config_path = report_path / f"{config.dataset_name}_config.json"
    config.save(config_path)

    report_config = DatasetCorrectionReportConfig(**config.model_dump())
    report_config.report_path = report_path

    logger.info(f"Loading GT csv: {config.subset_base_path / config.herdnet_annotation_name}")
    df_gt = pd.read_csv(config.subset_base_path / config.herdnet_annotation_name)
    logger.info(f"Loading detections: {config.detections_path}")
    df_pred = pd.read_csv(config.detections_path)

    logger.info(
        f"Hungarian-matching: radius={config.radius}px, "
        f"pred_score_threshold={pred_score_threshold}, "
        f"include_matched={include_matched}"
    )
    results = merge_dataset(df_gt, df_pred, radius=config.radius, pred_score_threshold=pred_score_threshold)
    stats = merge_summary(results)
    logger.info(
        f"Merge complete on {stats['images']} images: "
        f"matched={stats['matched']}, gt_only={stats['gt_only']} (review B), "
        f"pred_only={stats['pred_only']} (review A), borderline={stats['borderline']} (review D)"
    )
    if stats["gt_only"] + stats["pred_only"] + stats["borderline"] == 0:
        raise ValueError(
            "No disagreement found between GT and predictions — nothing to review."
        )

    if only_fp_images:
        # Drop images whose only candidates are gt_only / matched / borderline —
        # keep only images that have at least one pred_only point (= apparent
        # false positive, which Phase-13 error analysis showed are the
        # highest-value review candidates).
        kept_results = [r for r in results if len(r.pred_only) > 0]
        n_dropped = len(results) - len(kept_results)
        kept_stats = merge_summary(kept_results)
        logger.info(
            f"only_fp_images=True — keeping {len(kept_results)} of "
            f"{len(results)} images (dropped {n_dropped} images with no FPs)."
        )
        logger.info(
            f"After filter: matched={kept_stats['matched']}, "
            f"gt_only={kept_stats['gt_only']}, pred_only={kept_stats['pred_only']}, "
            f"borderline={kept_stats['borderline']}"
        )
        results = kept_results
        if not results:
            raise ValueError(
                "After only_fp_images filter, no images remain — nothing to upload."
            )

    logger.info(f"Loading Hasty GT: {config.subset_base_path / config.hasty_ground_truth_annotation_name}")
    hA_ground_truth = HastyAnnotationV2.from_file(
        config.subset_base_path / config.hasty_ground_truth_annotation_name
    )
    logger.info(f"Loading reference Hasty (for label_classes): "
                f"{config.reference_base_path / config.hasty_reference_annotation_name}")
    hA_reference = HastyAnnotationV2.from_file(
        config.reference_base_path / config.hasty_reference_annotation_name
    )

    # --- PRE-FLIGHT: match every input image to a Hasty dataset_name -----
    # Without this, the download step can't route corrections back into the
    # right Hasty dataset and the master would silently end up wrong.
    input_image_names = [r.image for r in results]
    image_to_dataset, missing = _preflight_match_to_hasty_datasets(
        input_image_names, hA_reference
    )
    if missing:
        logger.error(
            f"{len(missing)} / {len(input_image_names)} input images NOT FOUND "
            f"in the Hasty master at {config.reference_base_path / config.hasty_reference_annotation_name}."
        )
        for m in missing[:15]:
            logger.error(f"  missing: {m}")
        if len(missing) > 15:
            logger.error(f"  (+ {len(missing) - 15} more)")
        raise RuntimeError(
            "Pre-flight failed: cannot route these images to a Hasty dataset on download. "
            "Run /data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/unzipped_images_rsync "
            "and the master Hasty manifest to be sure the images you're trying to review are present."
        )
    mapping_path = report_path / f"{config.dataset_name}_image_to_dataset.json"
    mapping_path.write_text(json.dumps(image_to_dataset, indent=2))
    n_distinct_datasets = len({m["dataset_name"] for m in image_to_dataset.values()})
    logger.info(
        f"PRE-FLIGHT OK: all {len(image_to_dataset)} input images map to a Hasty record "
        f"({n_distinct_datasets} distinct datasets). Mapping at {mapping_path}"
    )

    # Build the intermediate Hasty (provenance class names + colors).
    hA_merged = merge_results_to_hasty(
        results=results,
        hA_reference=hA_ground_truth,
        dataset_name=config.dataset_name,
        base_class_name=base_class_name,
        include_matched=include_matched,
        images_path=config.images_path,
    )

    # Persist the intermediate Hasty — used by the download step.
    hA_intermediate_path = config.corrected_path / f"{config.dataset_name}_phase15_intermediate_hasty.json"
    hA_merged.save(hA_intermediate_path)
    report_config.hA_prediction_path = hA_intermediate_path
    logger.info(f"Wrote intermediate Hasty: {hA_intermediate_path}")

    # ---- Build the fiftyone dataset directly from the merged Hasty ------
    try:
        fo.delete_dataset(config.dataset_name)
    except Exception:
        pass
    dataset = fo.Dataset(name=config.dataset_name)
    provenance_classes = [
        f"{base_class_name}{s}" for s in
        (SUFFIX_GT_ONLY, SUFFIX_PRED_ONLY, SUFFIX_MATCHED, SUFFIX_BORDERLINE)
    ]
    dataset.default_classes = provenance_classes
    dataset.persistent = True

    n_samples = 0
    for hA_image in hA_merged.images:
        if not hA_image.labels:
            continue  # skip images with no flagged candidates
        image_path = config.images_path / hA_image.image_name
        if not image_path.exists():
            logger.warning(f"image file missing on disk, skipping: {image_path}")
            continue
        # Backfill width/height if Hasty doesn't have them (some exports don't).
        if not getattr(hA_image, "width", None) or not getattr(hA_image, "height", None):
            with PIL.Image.open(image_path) as im:
                hA_image.width, hA_image.height = im.size
        sample = fo.Sample(filepath=str(image_path))
        sample["detection"] = _build_fiftyone_keypoints(hA_image, base_class_name)
        # Stash provenance + the master-Hasty dataset_name on the sample so
        # the download step can recover it without re-querying the master.
        # Map this tile back to its Hasty (dataset_name, image_name) — the
        # download step uses both to look up the right master record (image
        # names alone aren't unique across datasets, e.g. DJI_0141.JPG).
        route = image_to_dataset[hA_image.image_name]
        sample["hasty_dataset_name"] = route["dataset_name"]
        sample["hasty_image_name"] = route["hasty_image_name"]
        sample["phase15_tile_name"] = hA_image.image_name
        dataset.add_sample(sample)
        n_samples += 1
    logger.info(f"Added {n_samples} samples to fiftyone dataset '{config.dataset_name}'.")
    dataset.compute_metadata()

    # Write the report BEFORE launching CVAT so the download step can find
    # everything even if CVAT submission times out / needs a retry.
    report_config_path = report_path / "report.json"
    report_config.save(report_config_path)

    # ---- Push to CVAT --------------------------------------------------
    logger.info(
        f"Launching CVAT task: project={CVAT_PROJECT_NAME!r}, "
        f"organization={CVAT_ORGANIZATION!r}, anno_key={config.dataset_name!r}"
    )
    # task_size keeps each CVAT task small enough to avoid 504 Gateway
    # Timeouts on the data upload POST (the public CVAT instance times out
    # when a single task tries to ingest more than ~150-200 MB of imagery).
    # All tasks land in the same project (Hasty_Corr), so the reviewer sees
    # them grouped.
    dataset.annotate(
        anno_key=config.dataset_name,
        label_field="detection",
        label_type="keypoints",
        classes=provenance_classes,
        # Declare hasty_id as a CVAT attribute so it round-trips with the
        # label even if the reviewer relocates it. The "score" field is
        # historical; not load-bearing for the diff.
        attributes=["hasty_id", "score"],
        task_size=50,
        launch_editor=True,
        organization=CVAT_ORGANIZATION,
        project_name=CVAT_PROJECT_NAME,
    )
    logger.info(
        f"Done. Report config (download step input): {report_config_path}\n"
        f"  Image->Hasty-dataset mapping: {mapping_path}"
    )
    return report_config


if __name__ == "__main__":
    # Example for Phase-13 val (iteration 0 — the seed pass).
    # Edit these paths per iteration.
    val_base_path = Path("/home/christian/data/training_data/2026_05_08_data_scaling/val")
    cleanup_base = val_base_path / "phase15_iter0"
    report_dir = cleanup_base / "report"

    config = DatasetCorrectionConfig(
        analysis_date="2026-05-30_phase15_iter0_val",
        dataset_name="phase15_iter0_val",
        type="points",
        subset_base_path=val_base_path,
        reference_base_path=val_base_path,  # same path; reference annotation lives here
        # The full-size Hasty annotation for these val tiles. Replace with the
        # actual file name (the val/test dirs have hasty_format_full_size.json).
        hasty_reference_annotation_name="hasty_format_full_size.json",
        hasty_ground_truth_annotation_name="hasty_format_full_size.json",
        # GT csv used by the herdnet evaluator.
        herdnet_annotation_name="herdnet_format.csv",
        # Detections produced by Phase-13 single B4 at adapt_ts=0.20.
        detections_path=Path(
            "/home/christian/hnee/HerdNet/output/phase13_eval_20260508_142316/Nfull/ts_0.20/detections.csv"
        ),
        correct_fp_gt=True,
        box_size=400,
        radius=100,
        images_path=val_base_path / "Default",
        output_path=cleanup_base / "output",
        corrected_path=cleanup_base / "corrections",
    )

    phase15_cvat_upload(
        config=config,
        report_path=report_dir,
        pred_score_threshold=0.5,
        include_matched=False,  # iteration 0: only review disagreements
    )
