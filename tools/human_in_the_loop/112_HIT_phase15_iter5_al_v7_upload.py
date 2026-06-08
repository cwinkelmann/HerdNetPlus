"""
Phase 15 — iteration 5 launcher (al_v7 held-out val).

Uploads predictions from the al_v7 best_model (epoch 2 by F2;
R=0.880, P=0.749, F2=0.850) for review of the 13 held-out
val datasets (~610 images, all net-new Espanola+Floreana
from iter-v2). Detections from
output/al_v7_inference_e2_<ts>/detections.csv.

Inference recipe: `adapt_ts=0.20` + `score_threshold=0.30`,
matching the training-time eval. Same task_size=25 reliable
config + `max_fp_images=200` density-ranked cap as iter-3/4.
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
AL_V7_ROOT = Path(
    "/home/christian/data/training_data/2026_06_03_al_v7_canonical/al_v7_2026_06_03"
)
VAL_IMG_DIR = AL_V7_ROOT / "val/Default"
VAL_GT_CSV = AL_V7_ROOT / "val/herdnet_format.csv"

DET_CSV: Path | None = None  # auto-pick newest al_v7_inference_e2_*

STAGE_ROOT = Path("/home/christian/hnee/HerdNet/output/phase15_iter5_al_v7")
STAGE_IMAGES = STAGE_ROOT / "val_images"
STAGE_DET_CSV = STAGE_ROOT / "val_detections.csv"
STAGE_GT_CSV = STAGE_ROOT / "val_herdnet_format.csv"
REPORT_DIR = STAGE_ROOT / "report"

PRED_SCORE_THRESHOLD = 0.3
INCLUDE_MATCHED = True
ONLY_FP_IMAGES = True
MATCH_RADIUS_PX = 100
BOX_SIZE_PX = 400
MAX_FP_IMAGES_DEFAULT = 500


def _find_det_csv() -> Path:
    if DET_CSV and DET_CSV.is_file():
        return DET_CSV
    candidates = sorted(
        Path("/home/christian/hnee/HerdNet/output").glob(
            "al_v7_inference_e*_*/detections.csv"
        )
    )
    if not candidates:
        raise FileNotFoundError("No al_v7 detections.csv found.")
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

    dataset_name = f"phase15_iter5_al_v7_{date.today().strftime('%Y_%m_%d')}"
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
    logger.info("=== Phase15 iter5 (al_v7) upload complete ===")
    logger.info(f"  Report: {REPORT_DIR}")


if __name__ == "__main__":
    main()
