"""
Phase 15 — iteration 1 launcher.

End-to-end:
  1. Backup the current master Hasty, promote the iteration-0 corrected
     file to master.
  2. Regenerate per-tile herdnet_format.csv files for val and test from
     the new master — so the upload's GT reflects the iteration-0
     corrections.
  3. Launch a fresh CVAT upload of the same val+test image set against
     the corrected GT. With the recent code update (commit 2708eca),
     every uploaded fo.Keypoint carries kp["hasty_id"] = label.id, so
     the iteration-1 download can match pre <-> post by ID directly.

For the 50 images you reviewed in iteration 0: the corrected GT is now
master, so the merge result will show *fewer* disagreements (most
candidate FPs you already accepted are now matched pairs). You shouldn't
have to re-review those.

For the remaining 175 images: same upload as iteration 0, but now with
hasty_id round-tripped.
"""

from __future__ import annotations

import shutil
import sys
from datetime import date
from importlib import import_module
from pathlib import Path

import pandas as pd
from loguru import logger

from active_learning.config.dataset_filter import DatasetCorrectionConfig

sys.path.append(str(Path(__file__).parent))

_upload_mod = import_module("100_HIT_phase15_upload")
phase15_cvat_upload = _upload_mod.phase15_cvat_upload

from hasty_to_herdnet_csv import hasty_to_herdnet_csv


# ---- inputs ----
HASTY_MASTER = Path(
    "/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json"
)
CORRECTED_HASTY = Path(
    "/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels_phase15_iter0_corrected.json"
)
HASTY_BACKUP = Path(
    "/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.pre_phase15_iter0.bak.json"
)

DATA_ROOT = Path("/home/christian/data/training_data/2026_05_08_data_scaling")
VAL_IMG_DIR = DATA_ROOT / "val/Default"
TEST_IMG_DIR = DATA_ROOT / "test/Default"

# Iteration-0 detections, unchanged (the model didn't re-predict).
VAL_DET_CSV = Path(
    "/home/christian/hnee/HerdNet/output/phase13_eval_20260508_142316/Nfull/ts_0.20/detections.csv"
)
TEST_DET_CSV = Path(
    "/home/christian/hnee/HerdNet/output/test_sweep_20260530_101438/phase13_single_b4_ts0.20/detections.csv"
)

# ---- staging ----
STAGE_ROOT = Path("/home/christian/hnee/HerdNet/output/phase15_iter1_val_test")
STAGE_IMAGES = STAGE_ROOT / "combined_images"
STAGE_GT_CSV = STAGE_ROOT / "combined_herdnet_format.csv"
STAGE_DET_CSV = STAGE_ROOT / "combined_detections.csv"
REPORT_DIR = STAGE_ROOT / "report"

# ---- knobs ----
PRED_SCORE_THRESHOLD = 0.5
INCLUDE_MATCHED = True
ONLY_FP_IMAGES = True
MATCH_RADIUS_PX = 100
BOX_SIZE_PX = 400


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
                shutil.copy2(f, target)
            n += 1
    return n


def _concat_csvs(paths: list[Path], dst: Path) -> int:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(dst, index=False)
    return len(df)


def promote_corrected_to_master() -> None:
    """Move the iter-0 corrected file into the master slot (with backup)."""
    if not CORRECTED_HASTY.is_file():
        raise FileNotFoundError(
            f"Corrected Hasty not found: {CORRECTED_HASTY}. Run "
            f"103_HIT_phase15_download_val_test.py first."
        )

    # Backup the current master (only on the first promotion).
    if HASTY_MASTER.is_file() and not HASTY_BACKUP.exists():
        shutil.copy2(HASTY_MASTER, HASTY_BACKUP)
        logger.info(f"Backed up master -> {HASTY_BACKUP}")
    elif HASTY_BACKUP.exists():
        logger.info(f"Backup already exists ({HASTY_BACKUP}), not overwriting")

    # Promote: copy corrected over master.
    shutil.copy2(CORRECTED_HASTY, HASTY_MASTER)
    logger.info(f"Promoted iter-0 corrected -> master: {HASTY_MASTER}")


def regenerate_herdnet_gt() -> tuple[Path, Path]:
    """Rebuild val + test herdnet_format.csv from the (newly promoted) master."""
    val_out = STAGE_ROOT / "val_herdnet_format.from_master.csv"
    test_out = STAGE_ROOT / "test_herdnet_format.from_master.csv"
    hasty_to_herdnet_csv(HASTY_MASTER, VAL_IMG_DIR, val_out)
    hasty_to_herdnet_csv(HASTY_MASTER, TEST_IMG_DIR, test_out)
    return val_out, test_out


def main() -> None:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Promote corrected -> master.
    promote_corrected_to_master()

    # 2. Regenerate per-tile GT csv from the new master.
    val_gt_csv, test_gt_csv = regenerate_herdnet_gt()

    # 3. Stage combined images + concatenated GT + concatenated detections.
    hasty_symlink = STAGE_ROOT / HASTY_MASTER.name
    if not hasty_symlink.exists():
        try:
            hasty_symlink.symlink_to(HASTY_MASTER.resolve())
        except OSError:
            shutil.copy2(HASTY_MASTER, hasty_symlink)
        logger.info(f"Symlinked Hasty master into staging: {hasty_symlink}")

    logger.info("Symlinking val + test images into combined staging dir...")
    n_symlinked = _symlink_images([VAL_IMG_DIR, TEST_IMG_DIR], STAGE_IMAGES)
    logger.info(f"  {n_symlinked} image files in {STAGE_IMAGES}")

    n_gt = _concat_csvs([val_gt_csv, test_gt_csv], STAGE_GT_CSV)
    logger.info(f"Wrote combined GT: {STAGE_GT_CSV} ({n_gt} rows)")
    n_det = _concat_csvs([VAL_DET_CSV, TEST_DET_CSV], STAGE_DET_CSV)
    logger.info(f"Wrote combined detections: {STAGE_DET_CSV} ({n_det} rows)")

    # 4. Build the DatasetCorrectionConfig with an iter-1 name (different
    # anno_key -> doesn't clash with the iter-0 CVAT task).
    dataset_name = f"phase15_iter1_val_test_{date.today().strftime('%Y_%m_%d')}"
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

    # 5. Launch upload — adds hasty_id to every fo.Keypoint (commit 2708eca).
    phase15_cvat_upload(
        config=config,
        report_path=REPORT_DIR,
        pred_score_threshold=PRED_SCORE_THRESHOLD,
        include_matched=INCLUDE_MATCHED,
        only_fp_images=ONLY_FP_IMAGES,
    )
    logger.info("=== Iteration 1 upload complete ===")
    logger.info(f"  Report:  {REPORT_DIR}")
    logger.info(f"  Backup:  {HASTY_BACKUP}")
    logger.info(f"  Master:  {HASTY_MASTER}")


if __name__ == "__main__":
    main()
