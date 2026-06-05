"""
Phase 15 — iteration 2 download for the al_v3 split.

Pulls human-corrected annotations from the CVAT tasks launched by
106_HIT_phase15_iter2_al_v3_upload.py. Under the explicit-positive
selection rule (iter-2+): only predictions whose post-class is
`iguana_point_gt` get promoted to master. Untouched red `iguana_point_pred`
points are recorded as `kept_unchanged` (no master change).

Writes:
  - <master>_phase15_iter2_corrected.json   (updated master; safe — doesn't overwrite original)
  - data/phase15_edit_log.csv               (append-only audit trail, tagged phase15_iter2_al_v3)
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
STAGE_ROOT = Path("/home/christian/hnee/HerdNet/output/phase15_iter2_al_v3")
REPORT_DIR = STAGE_ROOT / "report"
REPORT_JSON = REPORT_DIR / "report.json"

OUTPUT_HASTY = HASTY_MASTER.parent / f"{HASTY_MASTER.stem}_phase15_iter2_corrected.json"
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")
ITERATION_ID = "phase15_iter2_al_v3"


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
    print(f"  A={counts.A} (re-classed to iguana_point_gt → added to master)")
    print(f"  B={counts.B} (false GT removed)")
    print(f"  C={counts.C} (relocated)")
    print(f"  D={counts.D} (borderline — flag for second opinion)")
    print(f"  E={counts.E} (confirmed FPs — no master change)")
    print(f"  H={counts.H} (hard-negative: deleted or non-iguana class)")
    print(f"  kept_unchanged={counts.kept_unchanged} (untouched red pred_only)")
    print()
    print(f"Updated Hasty:  {OUTPUT_HASTY}")
    print(f"Edit log:       {EDIT_LOG}")
