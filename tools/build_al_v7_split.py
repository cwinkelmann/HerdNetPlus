"""
Build al_v7 split (2026-06-03, post iter-v2 merge).

Strategy:
  - Val  = the 709 NET-NEW images introduced by 2026_06_03_iteration_v2
           (mostly Espanola + Floreana, never reviewed in any phase-15 round).
           These are the freshest, most-likely-to-have-annotation-gaps
           images we have right now, so they're the right place to look
           for next-round CVAT corrections.
  - Train = master \ val, including:
             * non-Fer everything else
             * phase-15-reviewed Fer pairs (kept from al_v5/v6 strategy)
             * FPM01_24012023 (clean Fer reflight)

Inputs:
  - Promoted master at /data/.../2026_05_06_labels.json (post merge)
  - Pre-merge backup: 2026_05_06_labels.pre_iterv2_merge.bak.json
    Used to identify net-new (ds, name) pairs added by iter-v2.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger


MASTER = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json")
PRE_MERGE = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.pre_iterv2_merge.bak.json")
RSYNC = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/unzipped_images_rsync")
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")

OUT_DIR = Path("/home/christian/data/training_data/2026_06_03_al_v7_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

master_link = OUT_DIR / MASTER.name
if not master_link.exists():
    master_link.symlink_to(MASTER)

OUT_TRAIN = OUT_DIR / "al_v7_train_image_names.txt"
OUT_VAL = OUT_DIR / "al_v7_val_image_names.txt"
OUT_MANIFEST = OUT_DIR / "al_v7_manifest.json"

DONE_STATUSES = {"COMPLETED", "DONE", "Done"}

# Explicit small Fer dataset kept in train (the clean Jan-2023 reflight).
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
    """{(ds, name): idx} for images that are COMPLETED, have a iguana_point label
    with keypoints, and whose dataset dir exists on disk."""
    out: dict[tuple[str, str], int] = {}
    for idx, img in enumerate(blob.get("images", [])):
        ds = img.get("dataset_name")
        nm = img.get("image_name")
        if not ds or not nm:
            continue
        status = str(img.get("image_status"))
        if status not in DONE_STATUSES:
            continue
        has_pts = False
        for l in (img.get("labels") or []):
            if l.get("class_name") == "iguana_point" and (l.get("keypoints") or []):
                has_pts = True
                break
        if not has_pts:
            continue
        if not (RSYNC / ds).is_dir():
            continue
        out[(ds, nm)] = idx
    return out


def main() -> int:
    logger.info(f"Reading master: {MASTER}")
    master_blob = json.loads(MASTER.read_text())
    logger.info(f"Reading pre-merge backup: {PRE_MERGE}")
    pre_blob = json.loads(PRE_MERGE.read_text())

    pool = labeled_pool(master_blob)
    pre_pool_keys = set(
        (img.get("dataset_name"), img.get("image_name"))
        for img in pre_blob.get("images", [])
        if img.get("dataset_name") and img.get("image_name")
    )
    logger.info(f"Labeled pool: {len(pool)} pairs")
    logger.info(f"Pre-merge pool size: {len(pre_pool_keys)}")

    # Net-new images from iter-v2.
    net_new = {key for key in pool.keys() if key not in pre_pool_keys}
    logger.info(f"Net-new from iter-v2 (val candidates): {len(net_new)}")

    # Validate: a few net-new datasets should be the espanola/floreana ones.
    by_ds = Counter(ds for ds, _ in net_new)
    logger.info("Top net-new datasets (val):")
    for ds, n in by_ds.most_common(15):
        logger.info(f"  {ds:<40} {n}")

    # Read phase-15 reviewed pairs for Fer-keep-in-train logic.
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
    logger.info(f"phase-15 reviewed pairs: {len(reviewed_pairs)} (Fer: {len(reviewed_fern)})")

    fern_keep_in_train = set(FERN_INCLUDE_IN_TRAIN)

    val_pairs = set(net_new)
    train_pairs: set[tuple[str, str]] = set()
    for (ds, name) in pool.keys():
        if (ds, name) in val_pairs:
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
        "pre_merge_backup": str(PRE_MERGE),
        "pool_pairs": len(pool),
        "net_new_pairs_from_iterv2": len(net_new),
        "phase15_reviewed_pairs": len(reviewed_pairs),
        "reviewed_fern_pairs": len(reviewed_fern),
        "fern_explicitly_included_in_train": FERN_INCLUDE_IN_TRAIN,
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
