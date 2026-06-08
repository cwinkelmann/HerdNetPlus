"""
Build al_v8 split (2026-06-03, post phase-15 iter-5).

Strategy:
  - Val = ~1,100 fully-or-mostly UNREVIEWED non-Fernandina images,
          deliberately spread across Isabela, Floreana, Santa Cruz,
          Espanola, Pinta, Genovesa, Marchena and Zooniverse_expert_phase_2.
          User wants to correct the other islands before doing more Fernandina.
  - Train = master \\ val, including:
             * non-Fern everything else
             * phase15-reviewed Fer pairs
             * FPM01_24012023 (clean Fer reflight)

Output: /home/christian/data/training_data/2026_06_03_al_v8_canonical/
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from loguru import logger


MASTER = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json")
RSYNC = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/unzipped_images_rsync")
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")

OUT_DIR = Path("/home/christian/data/training_data/2026_06_03_al_v8_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

master_link = OUT_DIR / MASTER.name
if not master_link.exists():
    master_link.symlink_to(MASTER)

OUT_TRAIN = OUT_DIR / "al_v8_train_image_names.txt"
OUT_VAL = OUT_DIR / "al_v8_val_image_names.txt"
OUT_MANIFEST = OUT_DIR / "al_v8_manifest.json"

DONE_STATUSES = {"COMPLETED", "DONE", "Done"}

FERN_INCLUDE_IN_TRAIN = ["FPM01_24012023"]

# Val: untouched (or mostly untouched) non-Fern datasets, ~1,100 imgs.
VAL_DATASETS = [
    # Isabela (lots of unreviewed)
    "iswspb02_26012023",            # 146 (★ fresh)
    "isa_isvi01_27012023",          # 72  (★ fresh)
    "isa_ispda02_17012023",         # 70  (★ fresh)
    "ismu01_04_05_16122021",        # 61  (★ fresh)
    # Floreana (fresh)
    "flprs01_02_28012023",          # 123 (★)
    "flscj02_03022021",             # 102 (★)
    "flpl02_28012023",              # 81  (★)
    # Santa Cruz (whole island untouched)
    "scpe01_15012023",              # 80  (★)
    "sceab01_15012023",             # 59  (★)
    # Espanola
    "esp_epc03_10022023_body",      # 65  (★)
    # Pinta
    "pin_pwc02_11122021",           # 40  (★)
    # Genovesa
    "gwa02_med_05122021",           # 35  (★)
    # Marchena
    "mwbbc01_med_07122021",         # 30  (★)
    # Zooniverse expert phase 2 (159 imgs, only 2 reviewed)
    "Zooniverse_expert_phase_2",    # 159
]


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

    # Pre-flight: report sizes per val dataset
    logger.info("Val datasets (target ~1,100 imgs total):")
    val_pool_sum = 0
    for d in VAL_DATASETS:
        n = sum(1 for (ds, _) in pool.keys() if ds == d)
        val_pool_sum += n
        if n == 0:
            logger.warning(f"  {d:<40} NOT IN POOL — will contribute 0 images")
        else:
            logger.info(f"  {d:<40} {n}")
    logger.info(f"  TOTAL val pool: {val_pool_sum}")

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

    fern_keep_in_train = set(FERN_INCLUDE_IN_TRAIN)
    val_set = set(VAL_DATASETS)

    val_pairs: set[tuple[str, str]] = set()
    train_pairs: set[tuple[str, str]] = set()
    for (ds, name) in pool.keys():
        if ds in val_set:
            val_pairs.add((ds, name))
            continue
        if is_fernandina(ds):
            if ds in fern_keep_in_train or (ds, name) in reviewed_fern:
                train_pairs.add((ds, name))
        else:
            train_pairs.add((ds, name))

    train_names = {n for _, n in train_pairs}
    val_names = {n for _, n in val_pairs}
    overlap = train_names & val_names
    if overlap:
        logger.warning(f"{len(overlap)} cross-dataset name collisions removed from train")
        train_names -= overlap
    train_pairs_final = {(d, n) for (d, n) in train_pairs if n not in overlap}

    OUT_TRAIN.write_text("\n".join(sorted(train_names)) + "\n")
    OUT_VAL.write_text("\n".join(sorted(val_names)) + "\n")

    val_by_ds = defaultdict(int)
    for ds, _ in val_pairs:
        val_by_ds[ds] += 1
    val_by_ds_sorted = dict(sorted(val_by_ds.items(), key=lambda x: -x[1]))

    train_by_ds = defaultdict(int)
    for ds, _ in train_pairs_final:
        train_by_ds[ds] += 1

    OUT_MANIFEST.write_text(json.dumps({
        "master": str(MASTER),
        "pool_pairs": len(pool),
        "phase15_reviewed_pairs": len(reviewed_pairs),
        "reviewed_fern_pairs": len(reviewed_fern),
        "fern_explicitly_included_in_train": FERN_INCLUDE_IN_TRAIN,
        "val_datasets": VAL_DATASETS,
        "train_pairs": len(train_pairs_final),
        "train_distinct_image_names": len(train_names),
        "val_pairs": len(val_pairs),
        "val_distinct_image_names": len(val_names),
        "val_per_dataset": val_by_ds_sorted,
        "train_n_datasets": len(train_by_ds),
        "name_overlap_removed_from_train": sorted(overlap),
    }, indent=2))

    logger.info(f"Train: {len(train_pairs_final)} pairs / {len(train_names)} names "
                f"({len(train_by_ds)} datasets)")
    logger.info(f"Val:   {len(val_pairs)} pairs / {len(val_names)} names "
                f"({len(val_by_ds)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
