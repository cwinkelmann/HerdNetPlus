"""
Phase 15 — iteration 7 launcher (al_v8 E7 on the same 14-dataset val).

Re-runs the al_v8 val (1,125 imgs across 8 non-Fern islands) with the
LATEST best_model (E7 snapshot, F2=0.860, R=0.873, P=0.814) AND the
new efficiency knobs landed in 100_HIT_phase15_upload.py:

  - pred_score_threshold=0.50  (item 1: bump from 0.30, drop low-conf FPs)
  - ranking_mode="score_sum"   (item 2: high-confidence FPs surface first)
  - min_gt_count=3             (item 3: only review dense images)

Hard-negative suppression (already in upload code) drops candidates
that sit on top of iter-6's H promotions, so iter-7 only surfaces
*new* pred_only candidates the user hasn't seen.

Expected impact on A/(A+H) ratio: ~3-4x vs iter-6's 11.8%.
"""

from __future__ import annotations

import shutil
import sys
from datetime import date
from importlib import import_module
from pathlib import Path

from loguru import logger

from active_learning.config.dataset_filter import DatasetCorrectionConfig

sys.path.append(str(Path(__file__).parent))
_upload_mod = import_module("100_HIT_phase15_upload")
phase15_cvat_upload = _upload_mod.phase15_cvat_upload


HASTY_MASTER = Path(
    "/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json"
)
AL_V8_ROOT = Path(
    "/home/christian/data/training_data/2026_06_03_al_v8_canonical/al_v8_2026_06_03"
)
VAL_IMG_DIR = AL_V8_ROOT / "val/Default"
VAL_GT_CSV = AL_V8_ROOT / "val/herdnet_format.csv"

DET_CSV: Path | None = None  # auto-pick newest al_v8_inference_e7_*

STAGE_ROOT = Path("/home/christian/hnee/HerdNet/output/phase15_iter7_al_v8_v2")
STAGE_IMAGES = STAGE_ROOT / "val_images"
STAGE_DET_CSV = STAGE_ROOT / "val_detections.csv"
STAGE_GT_CSV = STAGE_ROOT / "val_herdnet_format.csv"
REPORT_DIR = STAGE_ROOT / "report"

# Efficiency-knob settings (different from iter-5/6 defaults).
PRED_SCORE_THRESHOLD = 0.50     # ↑ from 0.30 — drop low-confidence FPs
INCLUDE_MATCHED = True
ONLY_FP_IMAGES = True
MATCH_RADIUS_PX = 100
BOX_SIZE_PX = 400
MAX_FP_IMAGES_DEFAULT = 500
MIN_GT_COUNT = 3                # only images with ≥3 GT iguanas
RANKING_MODE = "score_sum"      # high-confidence FPs first


def _find_det_csv() -> Path:
    if DET_CSV and DET_CSV.is_file():
        return DET_CSV
    candidates = sorted(
        Path("/home/christian/hnee/HerdNet/output").glob(
            "al_v8_inference_e7_*/detections.csv"
        )
    )
    if not candidates:
        raise FileNotFoundError("No al_v8 E7 detections.csv found.")
    chosen = candidates[-1]
    logger.info(f"Using detections csv: {chosen}")
    return chosen


def _symlink_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        tgt = dst_dir / f.name
        if tgt.exists() or tgt.is_symlink():
            continue
        try:
            tgt.symlink_to(f.resolve())
        except OSError:
            shutil.copy2(f, tgt)
        n += 1
    return n


def main() -> None:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    det_csv = _find_det_csv()

    n = _symlink_images(VAL_IMG_DIR, STAGE_IMAGES)
    logger.info(f"Symlinked {n} val tiles into {STAGE_IMAGES}")

    if not STAGE_DET_CSV.exists():
        shutil.copy2(det_csv, STAGE_DET_CSV)
    if not STAGE_GT_CSV.exists():
        shutil.copy2(VAL_GT_CSV, STAGE_GT_CSV)

    hasty_symlink = STAGE_ROOT / HASTY_MASTER.name
    if not hasty_symlink.exists():
        try:
            hasty_symlink.symlink_to(HASTY_MASTER.resolve())
        except OSError:
            shutil.copy2(HASTY_MASTER, hasty_symlink)

    dataset_name = f"phase15_iter7_al_v8_v2_{date.today().strftime('%Y_%m_%d')}"
    config = DatasetCorrectionConfig(
        analysis_date=date.today().isoformat(),
        dataset_name=dataset_name,
        type="points",
        subset_base_path=STAGE_ROOT,
        reference_base_path=HASTY_MASTER.parent,
        hasty_reference_annotation_name=HASTY_MASTER.name,
        hasty_ground_truth_annotation_name=HASTY_MASTER.name,
        herdnet_annotation_name=STAGE_GT_CSV.name,
        detections_path=STAGE_DET_CSV,
        correct_fp_gt=True,
        box_size=BOX_SIZE_PX,
        radius=MATCH_RADIUS_PX,
        images_path=STAGE_IMAGES,
        output_path=STAGE_ROOT / "output",
        corrected_path=STAGE_ROOT / "corrections",
    )

    phase15_cvat_upload(
        config=config,
        report_path=REPORT_DIR,
        pred_score_threshold=PRED_SCORE_THRESHOLD,
        include_matched=INCLUDE_MATCHED,
        only_fp_images=ONLY_FP_IMAGES,
        max_fp_images=MAX_FP_IMAGES_DEFAULT,
        min_gt_count=MIN_GT_COUNT,
        ranking_mode=RANKING_MODE,
    )
    logger.info("=== Phase15 iter7 (al_v8 E7 + efficiency knobs) upload complete ===")
    logger.info(f"  Report: {REPORT_DIR}")


if __name__ == "__main__":
    main()
