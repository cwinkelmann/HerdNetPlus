"""Phase 15 iteration 5 download (al_v7 held-out val: Espanola+Floreana).

Run this once the iter-5 CVAT tasks (anno key
`phase15_iter5_al_v7_2026_06_03`) are reviewed. Pulls corrected
annotations back from CVAT, classifies each edited candidate into
A/B/C/H/kept_unchanged, appends rows to the phase15_edit_log, and
writes a corrected-master copy to
`2026_05_06_labels_phase15_iter5_corrected.json` (the in-place
master is NOT overwritten — review the corrected file, then promote
manually with the same `cp` pattern used in previous iterations).
"""
from __future__ import annotations
import sys
from pathlib import Path
from importlib import import_module

sys.path.append(str(Path(__file__).parent))
_dl_mod = import_module("101_HIT_phase15_download")
phase15_cvat_download_and_update = _dl_mod.phase15_cvat_download_and_update


HASTY_MASTER = Path(
    "/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json"
)
STAGE_ROOT = Path("/home/christian/hnee/HerdNet/output/phase15_iter5_al_v7")
REPORT_DIR = STAGE_ROOT / "report"
REPORT_JSON = REPORT_DIR / "report.json"

OUTPUT_HASTY = HASTY_MASTER.parent / f"{HASTY_MASTER.stem}_phase15_iter5_corrected.json"
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")
ITERATION_ID = "phase15_iter5_al_v7"


if __name__ == "__main__":
    counts = phase15_cvat_download_and_update(
        report_path=REPORT_JSON,
        master_hasty_path=HASTY_MASTER,
        output_master_hasty_path=OUTPUT_HASTY,
        edit_log_path=EDIT_LOG,
        iteration_id=ITERATION_ID,
    )
    print()
    print(f"Iteration {ITERATION_ID} complete.")
    print(f"  A={counts.A} B={counts.B} C={counts.C} H={counts.H} kept_unchanged={counts.kept_unchanged}")
    print(f"Updated Hasty: {OUTPUT_HASTY}")
    print(f"Edit log:      {EDIT_LOG}")
