"""
Build al_v10 split (2026-06-08, post phase-15 iter-7 master).

Strategy:
  - Val = 991 untouched ortho-like images across 5 islands (Marchena,
    Floreana, Fernandina, Isabela, Pinta). All datasets are fully fresh
    in phase-15 history. They represent "natural" full-orthomosaic
    distributions the model has never seen.
  - Train = master \\ val, includes everything else — phase-15-reviewed
    Fer, FPM01_24012023, non-Fer, etc. (i.e. "mostly everything").

Goal: measure how close the predictions on these holdout orthomosaics
match a human-labeler's expected counts/positions. Used later for
deciding the production model's census-readiness.
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

OUT_DIR = Path("/home/christian/data/training_data/2026_06_08_al_v10_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

master_link = OUT_DIR / MASTER.name
if not master_link.exists():
    master_link.symlink_to(MASTER)

OUT_TRAIN = OUT_DIR / "al_v10_train_image_names.txt"
OUT_VAL = OUT_DIR / "al_v10_val_image_names.txt"
OUT_MANIFEST = OUT_DIR / "al_v10_manifest.json"

DONE_STATUSES = {"COMPLETED", "DONE", "Done"}

# 5-island ortho-like val (all fully untouched in phase-15 history).
VAL_DATASETS = [
    # Marchena
    "mbbd04_med_07122021",          # 82
    "mbbd02_med_07122021",          # 68
    # Floreana
    "flscj01_03022021",             # 83
    "flpvin02_28012023",            # 60
    # Fernandina (one big dense-colony ortho)
    "fer_fef01_02_20012023",        # 415
    # Isabela
    "islcb01_02_21012023",          # 78
    "iseb04_19012023",              # 76
    # Pinta
    "pcie08_medium_10122021",       # 72
    "pwc01_med_11122021",           # 57
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

    logger.info("Val datasets (5 islands, all ★ untouched):")
    val_total = 0
    for d in VAL_DATASETS:
        n = sum(1 for (ds, _) in pool.keys() if ds == d)
        val_total += n
        if n == 0:
            logger.warning(f"  {d:<40} NOT IN POOL (will contribute 0)")
        else:
            logger.info(f"  {d:<40} {n}")
    logger.info(f"  TOTAL val pool: {val_total}")

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

    val_pairs = {(ds, nm) for (ds, nm) in pool.keys() if ds in val_set}

    # Train = master \ val (mostly everything). For Fer datasets NOT
    # in val we include all phase-15-reviewed images + FPM01_24012023
    # (small clean reflight), same as al_v8/al_v9 fer_in.
    fern_keep = {"FPM01_24012023"}
    train_pairs: set[tuple[str, str]] = set()
    for (ds, nm) in pool.keys():
        if ds in val_set:
            continue
        if is_fernandina(ds):
            if ds in fern_keep or (ds, nm) in reviewed_fern:
                train_pairs.add((ds, nm))
        else:
            train_pairs.add((ds, nm))

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
    train_by_ds = defaultdict(int)
    for ds, _ in train_pairs_final:
        train_by_ds[ds] += 1

    OUT_MANIFEST.write_text(json.dumps({
        "master": str(MASTER),
        "pool_pairs": len(pool),
        "phase15_reviewed_pairs": len(reviewed_pairs),
        "reviewed_fern_pairs": len(reviewed_fern),
        "fern_explicitly_included_in_train": sorted(fern_keep),
        "val_datasets": VAL_DATASETS,
        "val_per_dataset": dict(sorted(val_by_ds.items(), key=lambda x: -x[1])),
        "val_pairs": len(val_pairs),
        "val_distinct_image_names": len(val_names),
        "train_pairs": len(train_pairs_final),
        "train_distinct_image_names": len(train_names),
        "train_n_datasets": len(train_by_ds),
        "name_overlap_removed_from_train": sorted(overlap),
    }, indent=2))

    logger.info(f"Val:   {len(val_pairs)} pairs / {len(val_names)} names ({len(val_by_ds)} datasets)")
    logger.info(f"Train: {len(train_pairs_final)} pairs / {len(train_names)} names "
                f"({len(train_by_ds)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
