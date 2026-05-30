"""
Phase 15 — iteration 0 download. Pulls the human-corrected annotations
from the CVAT task launched by ``102_HIT_phase15_upload_val_test.py``
and applies them to the Hasty master at
``/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json``.

Run after the human reviewer has worked through the CVAT task. Writes:
  - <Hasty master>_phase15_iter0_corrected.json  (the updated master)
  - data/phase15_edit_log.csv                    (append-only audit trail)
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
STAGE_ROOT = Path("/home/christian/hnee/HerdNet/output/phase15_iter0_val_test")
REPORT_DIR = STAGE_ROOT / "report"
REPORT_JSON = REPORT_DIR / "report.json"

OUTPUT_HASTY = HASTY_MASTER.parent / f"{HASTY_MASTER.stem}_phase15_iter0_corrected.json"
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")
ITERATION_ID = "phase15_iter0_val_test"


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
    print(f"  A={counts.A} (kept as iguana / promoted to GT)")
    print(f"  B={counts.B} (false GT removed)")
    print(f"  C={counts.C} (relocated)")
    print(f"  D={counts.D} (borderline — flag for second opinion)")
    print(f"  E={counts.E} (confirmed FPs — no master change)")
    print(f"  H={counts.H} (hard-negative promoted: kept as non-iguana class)")
    print(f"  kept_unchanged={counts.kept_unchanged}")
    print()
    print(f"Updated Hasty:  {OUTPUT_HASTY}")
    print(f"Edit log:       {EDIT_LOG}")
    print()
    print("Once you've validated the result, copy/rename the updated Hasty over")
    print("the original to make it the new production GT, then move on to iteration 1.")
