"""
Phase 15 — iteration 2 launcher (al_v3 image-level split, ~3000 new val images).

Pulls predictions from the al_v3 epoch-3 model (best recall, R=0.8559)
and uploads them to CVAT alongside the corrected-master GT for human
review.

The al_v3 split + corrected master are already in place upstream
(produced by tools/build_al_v3_split.py and
scripts/training_data_preparation/021_run_al_v3_nocrop.py).
This launcher just stages + calls phase15_cvat_upload.

Inputs (all expected to exist already):
  - corrected master Hasty: /data/.../2026_05_06_labels.json
  - val Default/: /home/christian/data/training_data/2026_06_01_al_v3_canonical/al_v3_2026_06_01/val/Default/
  - val herdnet_format.csv: same dir
  - detections.csv: produced by tools/infer.py on val/Default with
    epoch3_model.pth (set DET_CSV below to the inference output path).

Outputs:
  - CVAT task in project 'Hasty_Corr', org 'IguanasFromAbove'
  - Intermediate Hasty + report in al_v3/iter2/
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


# --- Inputs ---------------------------------------------------------------
HASTY_MASTER = Path(
    "/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json"
)

AL_V3_ROOT = Path(
    "/home/christian/data/training_data/2026_06_01_al_v3_canonical/al_v3_2026_06_01"
)
VAL_IMG_DIR = AL_V3_ROOT / "val/Default"
VAL_GT_CSV = AL_V3_ROOT / "val/herdnet_format.csv"

# Set to the actual inference output dir produced by tools/infer.py.
# We'll detect it dynamically if not explicitly set.
DET_CSV: Path | None = None  # If None, picks newest al_v3_inference_e3_* output.

# --- Staging --------------------------------------------------------------
STAGE_ROOT = Path("/home/christian/hnee/HerdNet/output/phase15_iter2_al_v3")
STAGE_IMAGES = STAGE_ROOT / "val_images"
STAGE_DET_CSV = STAGE_ROOT / "val_detections.csv"
STAGE_GT_CSV = STAGE_ROOT / "val_herdnet_format.csv"
REPORT_DIR = STAGE_ROOT / "report"

# --- Upload knobs ---------------------------------------------------------
PRED_SCORE_THRESHOLD = 0.5
# Cap how many images go to CVAT review per upload. Phase-15 ranks by
# pred_only count (most-candidate images first) and truncates. With ~200
# images we get 1 CVAT task × 4 jobs of 50, which is the iter-1 working
# shape.
MAX_FP_IMAGES_DEFAULT = 200
INCLUDE_MATCHED = True
ONLY_FP_IMAGES = True
MATCH_RADIUS_PX = 100
BOX_SIZE_PX = 400


def _find_det_csv() -> Path:
    if DET_CSV and DET_CSV.is_file():
        return DET_CSV
    # Prefer the dscore-filtered file if it exists (smaller review set).
    filtered = sorted(
        Path("/home/christian/hnee/HerdNet/output").glob(
            "al_v3_inference_e3_*/detections_dscore*.csv"
        )
    )
    if filtered:
        chosen = filtered[-1]
        logger.info(f"Using dscore-filtered detections csv: {chosen}")
        return chosen
    candidates = sorted(
        Path("/home/christian/hnee/HerdNet/output").glob(
            "al_v3_inference_e3_*/detections.csv"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "No detections.csv found. Run tools/infer.py first, or set DET_CSV."
        )
    chosen = candidates[-1]
    logger.info(f"Auto-picked detections csv: {chosen}")
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

    logger.info(f"Symlinking val Default tiles into {STAGE_IMAGES}")
    n = _symlink_images(VAL_IMG_DIR, STAGE_IMAGES)
    logger.info(f"  {n} new symlinks")

    if not STAGE_DET_CSV.exists():
        shutil.copy2(det_csv, STAGE_DET_CSV)
        logger.info(f"Copied detections to {STAGE_DET_CSV}")
    if not STAGE_GT_CSV.exists():
        shutil.copy2(VAL_GT_CSV, STAGE_GT_CSV)
        logger.info(f"Copied GT to {STAGE_GT_CSV}")

    # Symlink master JSON into stage_root so DatasetCorrectionConfig can find it.
    hasty_symlink = STAGE_ROOT / HASTY_MASTER.name
    if not hasty_symlink.exists():
        try:
            hasty_symlink.symlink_to(HASTY_MASTER.resolve())
        except OSError:
            shutil.copy2(HASTY_MASTER, hasty_symlink)

    dataset_name = f"phase15_iter2_al_v3_{date.today().strftime('%Y_%m_%d')}"
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
    )
    logger.info("=== Phase15 iter2 (al_v3) upload complete ===")
    logger.info(f"  Report: {REPORT_DIR}")


if __name__ == "__main__":
    main()
