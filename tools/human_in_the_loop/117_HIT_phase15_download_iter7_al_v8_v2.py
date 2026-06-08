"""Phase 15 iteration 7 download (al_v8 E7 + efficiency knobs).

Special rule for THIS iteration: ``untouched_pred_is_hard_neg=True``.
The reviewer for iter-7 only re-classed confirmed iguanas to
`iguana_point_gt` and skipped deletions to save time. So any
pred_only/borderline left untouched gets promoted to H (hard
negative) with a `not_iguana_but_similar_look` master label, instead
of the default "kept_unchanged" interpretation.

Effect: this iteration's H count will be much larger than iter-5/6
because every NOT-deleted-NOT-confirmed candidate counts as H.
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
STAGE_ROOT = Path("/home/christian/hnee/HerdNet/output/phase15_iter7_al_v8_v2")
REPORT_DIR = STAGE_ROOT / "report"
REPORT_JSON = REPORT_DIR / "report.json"

OUTPUT_HASTY = HASTY_MASTER.parent / f"{HASTY_MASTER.stem}_phase15_iter7_corrected.json"
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")
ITERATION_ID = "phase15_iter7_al_v8_v2"


if __name__ == "__main__":
    counts = phase15_cvat_download_and_update(
        report_path=REPORT_JSON,
        master_hasty_path=HASTY_MASTER,
        output_master_hasty_path=OUTPUT_HASTY,
        edit_log_path=EDIT_LOG,
        iteration_id=ITERATION_ID,
        untouched_pred_is_hard_neg=True,
    )
    print()
    print(f"Iteration {ITERATION_ID} complete.")
    print(f"  A={counts.A} B={counts.B} C={counts.C} H={counts.H} kept_unchanged={counts.kept_unchanged}")
    print(f"Updated Hasty: {OUTPUT_HASTY}")
    print(f"Edit log:      {EDIT_LOG}")
