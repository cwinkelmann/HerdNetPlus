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
from pathlib import Path

import pandas as pd
from loguru import logger

from active_learning.config.dataset_filter import (
    DatasetCorrectionConfig,
    DatasetCorrectionReportConfig,
)
from active_learning.util.evaluation.evaluation import submit_for_cvat_evaluation
from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2

import fiftyone as fo

# Local helper (in this directory). When running directly add this dir to PYTHONPATH
# or set `python -m scripts.human_in_the_loop.100_HIT_phase15_upload` semantics.
import sys
sys.path.append(str(Path(__file__).parent))
from hit_phase15_merge import (
    merge_dataset,
    merge_summary,
    merge_results_to_hasty,
    SUFFIX_GT_ONLY,
    SUFFIX_PRED_ONLY,
    SUFFIX_MATCHED,
    SUFFIX_BORDERLINE,
)


def phase15_cvat_upload(
    config: DatasetCorrectionConfig,
    report_path: Path,
    pred_score_threshold: float = 0.5,
    include_matched: bool = False,
    base_class_name: str = "iguana",
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

    logger.info(f"Loading Hasty GT: {config.subset_base_path / config.hasty_ground_truth_annotation_name}")
    hA_ground_truth = HastyAnnotationV2.from_file(
        config.subset_base_path / config.hasty_ground_truth_annotation_name
    )
    logger.info(f"Loading reference Hasty (for label_classes): "
                f"{config.reference_base_path / config.hasty_reference_annotation_name}")
    hA_reference = HastyAnnotationV2.from_file(
        config.reference_base_path / config.hasty_reference_annotation_name
    )

    hA_merged = merge_results_to_hasty(
        results=results,
        hA_reference=hA_ground_truth,  # we need *image metadata* from GT, not reference
        dataset_name=config.dataset_name,
        base_class_name=base_class_name,
        include_matched=include_matched,
    )
    # Borrow label_classes from the production reference so CVAT recognises them.
    hA_merged.label_classes = hA_reference.label_classes

    # Persist the intermediate Hasty — this is what download time will read
    # to recover provenance and the pre-correction state.
    hA_intermediate_path = config.corrected_path / f"{config.dataset_name}_phase15_intermediate_hasty.json"
    hA_merged.save(hA_intermediate_path)
    report_config.hA_prediction_path = hA_intermediate_path
    logger.info(f"Wrote intermediate Hasty: {hA_intermediate_path}")

    # Build a fiftyone dataset + push to CVAT — same pattern as
    # helper.py::hit_fp_gt_cvat_upload.
    try:
        fo.delete_dataset(config.dataset_name)
    except Exception:
        pass
    dataset = fo.Dataset(name=config.dataset_name)
    dataset.default_classes = [
        f"{base_class_name}{SUFFIX_GT_ONLY}",
        f"{base_class_name}{SUFFIX_PRED_ONLY}",
        f"{base_class_name}{SUFFIX_MATCHED}",
        f"{base_class_name}{SUFFIX_BORDERLINE}",
    ]
    dataset.persistent = True

    submit_for_cvat_evaluation(
        config=config,
        report_config=report_config,
        hA_prediction=hA_merged,
        dataset=dataset,
    )
    report_config_path = report_path / "report.json"
    report_config.save(report_config_path)
    logger.info(f"Wrote report config (use this in the download step): {report_config_path}")
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
