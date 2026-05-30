"""
Phase 15 — iteration 0 combined upload of val + test predictions to CVAT.

Why a separate script vs ``100_HIT_phase15_upload.py``: that one is the
generic upload helper. This is the *iteration-0 launcher* that
combines the val and test sets (both already evaluated at ts=0.20 by
the Phase-13 single B4 production candidate) into a single CVAT task,
matching predictions against the Hasty master at
``/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json``.

It:
  1. Concatenates the val + test herdnet-format GT CSVs.
  2. Concatenates the val + test detections (ts=0.20).
  3. Symlinks all 978 image tiles into a single staging directory
     (the CVAT upload helper expects one images_path).
  4. Builds a DatasetCorrectionConfig pointing at the staging dir + the
     production Hasty master.
  5. Calls phase15_cvat_upload() — Hungarian-merges and pushes to CVAT.
  6. Writes report.json so that ``101_HIT_phase15_download.py`` can apply
     the reviewer's corrections back to the Hasty master.

After the human reviews in CVAT, run the download script with:
   master_hasty_path = the same Hasty file referenced below
   report_path       = the report dir printed by this script
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from active_learning.config.dataset_filter import DatasetCorrectionConfig

import sys
sys.path.append(str(Path(__file__).parent))
from importlib import import_module
# 100_HIT_phase15_upload.py has a leading digit; import via importlib.
_upload_mod = import_module("100_HIT_phase15_upload")
phase15_cvat_upload = _upload_mod.phase15_cvat_upload


# ---- inputs ----
HASTY_MASTER = Path(
    "/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json"
)
DATA_ROOT = Path("/home/christian/data/training_data/2026_05_08_data_scaling")

VAL_GT_CSV = DATA_ROOT / "val/herdnet_format.csv"
VAL_IMG_DIR = DATA_ROOT / "val/Default"
VAL_DET_CSV = Path(
    "/home/christian/hnee/HerdNet/output/phase13_eval_20260508_142316/Nfull/ts_0.20/detections.csv"
)

TEST_GT_CSV = DATA_ROOT / "test/herdnet_format.csv"
TEST_IMG_DIR = DATA_ROOT / "test/Default"
# Phase-13 single B4 at ts=0.20 on the test set — produced by run_test_eval_sweep.sh.
TEST_DET_CSV = Path(
    "/home/christian/hnee/HerdNet/output/test_sweep_20260530_101438/phase13_single_b4_ts0.20/detections.csv"
)

# ---- staging ----
STAGE_ROOT = Path("/home/christian/hnee/HerdNet/output/phase15_iter0_val_test")
STAGE_IMAGES = STAGE_ROOT / "combined_images"
STAGE_GT_CSV = STAGE_ROOT / "combined_herdnet_format.csv"
STAGE_DET_CSV = STAGE_ROOT / "combined_detections.csv"
REPORT_DIR = STAGE_ROOT / "report"

# ---- knobs ----
PRED_SCORE_THRESHOLD = 0.5   # filter low-confidence noise; bump higher for fewer candidates
INCLUDE_MATCHED = False      # iter 0: review disagreements only
ONLY_FP_IMAGES = True        # only upload images that have at least one pred_only candidate
MATCH_RADIUS_PX = 100        # matches the evaluator's threshold
BOX_SIZE_PX = 400            # CVAT crop window per label


def _symlink_images(src_dirs: list[Path], dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in src_dirs:
        if not src.is_dir():
            raise FileNotFoundError(src)
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            target = dst_dir / f.name
            if target.exists() or target.is_symlink():
                continue
            try:
                target.symlink_to(f.resolve())
            except OSError:
                # Fallback to copy if symlinking is disallowed (e.g. across mounts).
                shutil.copy2(f, target)
            n += 1
    return n


def _concat_csvs(paths: list[Path], dst: Path) -> int:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(dst, index=False)
    return len(df)


def main() -> None:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("Symlinking val + test images into combined staging dir...")
    n_symlinked = _symlink_images([VAL_IMG_DIR, TEST_IMG_DIR], STAGE_IMAGES)
    logger.info(f"  {n_symlinked} image files in {STAGE_IMAGES}")

    n_gt = _concat_csvs([VAL_GT_CSV, TEST_GT_CSV], STAGE_GT_CSV)
    logger.info(f"Wrote combined GT: {STAGE_GT_CSV} ({n_gt} rows)")
    n_det = _concat_csvs([VAL_DET_CSV, TEST_DET_CSV], STAGE_DET_CSV)
    logger.info(f"Wrote combined detections: {STAGE_DET_CSV} ({n_det} rows)")

    dataset_name = f"phase15_iter0_val_test_{date.today().isoformat()}"
    config = DatasetCorrectionConfig(
        analysis_date=date.today().isoformat(),
        dataset_name=dataset_name,
        type="points",
        subset_base_path=STAGE_ROOT,
        reference_base_path=HASTY_MASTER.parent,
        # The production Hasty master we'll eventually update.
        hasty_reference_annotation_name=HASTY_MASTER.name,
        # We don't have a separate "GT Hasty for these tiles" — use the master.
        # Image metadata (name, width, height) for the val+test tiles must be present
        # in this file. If not, hA_reference fallback in merge_results_to_hasty
        # will skip those images with a warning.
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

    report_config = phase15_cvat_upload(
        config=config,
        report_path=REPORT_DIR,
        pred_score_threshold=PRED_SCORE_THRESHOLD,
        include_matched=INCLUDE_MATCHED,
        only_fp_images=ONLY_FP_IMAGES,
    )
    logger.info("=== Upload complete ===")
    logger.info(f"  report dir:       {REPORT_DIR}")
    logger.info(f"  master Hasty:     {HASTY_MASTER}")
    logger.info("Next step: have the human reviewer go through the CVAT task,")
    logger.info("then run 101_HIT_phase15_download.py with:")
    logger.info(f"  report_path             = {REPORT_DIR}")
    logger.info(f"  master_hasty_path       = {HASTY_MASTER}")
    logger.info(f"  output_master_hasty_path= {HASTY_MASTER.parent / (HASTY_MASTER.stem + '_phase15_iter0_corrected.json')}")
    logger.info(f"  edit_log_path           = /home/christian/hnee/HerdNet/data/phase15_edit_log.csv")


if __name__ == "__main__":
    main()
