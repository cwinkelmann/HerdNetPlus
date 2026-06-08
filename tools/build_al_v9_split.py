"""
Build al_v9 split (2026-06-04, post phase-15 iter-7 promotion).

Two train variants for a 2×2 sweep:
  --fern-mode in    Train includes phase-15-reviewed Fer + FPM01_24012023
                    (same as al_v8 default).
  --fern-mode out   Train fully excludes all Fer datasets.

Shared val across both modes: the al_v8 val (1,125 imgs / 14 non-Fern
datasets / 8 islands).

Outputs to /home/christian/data/training_data/2026_06_04_al_v9_canonical/:
  - al_v9_train_fer_in.txt  / al_v9_train_fer_out.txt
  - al_v9_val.txt (single shared val image-name list)
  - al_v9_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from loguru import logger


MASTER = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json")
RSYNC = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/unzipped_images_rsync")
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")

OUT_DIR = Path("/home/christian/data/training_data/2026_06_04_al_v9_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

master_link = OUT_DIR / MASTER.name
if not master_link.exists():
    master_link.symlink_to(MASTER)

OUT_VAL = OUT_DIR / "al_v9_val.txt"
OUT_TRAIN_IN = OUT_DIR / "al_v9_train_fer_in.txt"
OUT_TRAIN_OUT = OUT_DIR / "al_v9_train_fer_out.txt"
OUT_MANIFEST = OUT_DIR / "al_v9_manifest.json"

DONE_STATUSES = {"COMPLETED", "DONE", "Done"}

# Same val datasets as al_v8 — 14 untouched non-Fern across 8 islands.
VAL_DATASETS = [
    "iswspb02_26012023", "isa_isvi01_27012023", "isa_ispda02_17012023",
    "ismu01_04_05_16122021",
    "flprs01_02_28012023", "flscj02_03022021", "flpl02_28012023",
    "scpe01_15012023", "sceab01_15012023",
    "esp_epc03_10022023_body",
    "pin_pwc02_11122021",
    "gwa02_med_05122021",
    "mwbbc01_med_07122021",
    "Zooniverse_expert_phase_2",
]

FERN_INCLUDE_IN_TRAIN = ["FPM01_24012023"]


def is_fernandina(ds: str) -> bool:
    if not ds:
        return False
    ds_lo = ds.lower()
    if ds_lo.startswith("fer_") or ds.startswith("Fer_"):
        return True
    if ds in {
        "Fer_FCD01-02-03_20122021",
        "Fer_FCD01-02-03_20122021_single_images",
        "Fer_FPM01-02_20122023",
        "Fer_FPE02_07052024",
        "FPM01_24012023",
        "floreana_FPE01_FECA01",
    }:
        return True
    for pfx in ("fpe", "fsa", "fsd", "fsg", "fsj", "fso", "fwe"):
        if ds_lo.startswith(pfx):
            return True
    return False


def labeled_pool(blob) -> dict[tuple[str, str], int]:
    out = {}
    for idx, img in enumerate(blob.get("images", [])):
        ds = img.get("dataset_name")
        nm = img.get("image_name")
        if not ds or not nm:
            continue
        if str(img.get("image_status")) not in DONE_STATUSES:
            continue
        if not any(l.get("class_name") == "iguana_point" and (l.get("keypoints") or [])
                   for l in (img.get("labels") or [])):
            continue
        if not (RSYNC / ds).is_dir():
            continue
        out[(ds, nm)] = idx
    return out


def main() -> int:
    logger.info(f"Reading master: {MASTER}")
    master_blob = json.loads(MASTER.read_text())
    pool = labeled_pool(master_blob)
    logger.info(f"Labeled pool: {len(pool)} pairs")

    reviewed_pairs: set[tuple[str, str]] = set()
    if EDIT_LOG.exists():
        with EDIT_LOG.open(newline="") as f:
            for row in csv.DictReader(f):
                tile = row.get("image", "")
                if "___" not in tile:
                    continue
                ds, name = tile.split("___", 1)
                reviewed_pairs.add((ds, name))
    reviewed_fern = {p for p in reviewed_pairs if is_fernandina(p[0])}
    logger.info(f"phase15 reviewed: {len(reviewed_pairs)} (Fer: {len(reviewed_fern)})")

    val_set = set(VAL_DATASETS)
    fern_keep_in_train = set(FERN_INCLUDE_IN_TRAIN)

    # Val (shared).
    val_pairs = {(ds, nm) for (ds, nm) in pool.keys() if ds in val_set}
    val_names = sorted({nm for _, nm in val_pairs})
    OUT_VAL.write_text("\n".join(val_names) + "\n")

    # Train — fer_in variant.
    train_in: set[tuple[str, str]] = set()
    for (ds, nm) in pool.keys():
        if ds in val_set:
            continue
        if is_fernandina(ds):
            if ds in fern_keep_in_train or (ds, nm) in reviewed_fern:
                train_in.add((ds, nm))
        else:
            train_in.add((ds, nm))

    # Train — fer_out variant: same as in, minus ALL Fer.
    train_out: set[tuple[str, str]] = {p for p in train_in if not is_fernandina(p[0])}

    # Strip cross-dataset name collisions w/ val (rare).
    val_name_set = set(val_names)
    for name_set, label in [(train_in, "fer_in"), (train_out, "fer_out")]:
        names = {nm for _, nm in name_set}
        overlap = names & val_name_set
        if overlap:
            logger.warning(f"{label}: {len(overlap)} train↔val name collisions removed")
            name_set -= {(d, n) for (d, n) in name_set if n in overlap}

    train_in_names = sorted({nm for _, nm in train_in})
    train_out_names = sorted({nm for _, nm in train_out})
    OUT_TRAIN_IN.write_text("\n".join(train_in_names) + "\n")
    OUT_TRAIN_OUT.write_text("\n".join(train_out_names) + "\n")

    # Manifest summary.
    val_by_ds = defaultdict(int)
    for ds, _ in val_pairs:
        val_by_ds[ds] += 1
    in_by_ds = defaultdict(int)
    out_by_ds = defaultdict(int)
    for ds, _ in train_in:
        in_by_ds[ds] += 1
    for ds, _ in train_out:
        out_by_ds[ds] += 1

    OUT_MANIFEST.write_text(json.dumps({
        "master": str(MASTER),
        "pool_pairs": len(pool),
        "phase15_reviewed_pairs": len(reviewed_pairs),
        "reviewed_fern_pairs": len(reviewed_fern),
        "fern_explicitly_included_in_train": FERN_INCLUDE_IN_TRAIN,
        "val_datasets": VAL_DATASETS,
        "val_pairs": len(val_pairs),
        "val_names": len(val_names),
        "val_per_dataset": dict(sorted(val_by_ds.items(), key=lambda x: -x[1])),
        "train_fer_in_pairs": len(train_in),
        "train_fer_in_names": len(train_in_names),
        "train_fer_in_n_datasets": len(in_by_ds),
        "train_fer_out_pairs": len(train_out),
        "train_fer_out_names": len(train_out_names),
        "train_fer_out_n_datasets": len(out_by_ds),
    }, indent=2))

    logger.info(f"Val:           {len(val_pairs)} pairs / {len(val_names)} names "
                f"({len(val_by_ds)} datasets)")
    logger.info(f"Train fer_in:  {len(train_in)} pairs / {len(train_in_names)} names "
                f"({len(in_by_ds)} datasets)")
    logger.info(f"Train fer_out: {len(train_out)} pairs / {len(train_out_names)} names "
                f"({len(out_by_ds)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
