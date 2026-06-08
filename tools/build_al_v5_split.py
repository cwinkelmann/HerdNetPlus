"""
Build active-learning iteration-5 split (2026-06-02 evening).

Master state: post-phase15-iter3 (boxes=20692, iguana_point=73490,
not_iguana_but_similar_look=309).

Strategy:
- Train = all non-Fernandina datasets (minus VAL) + iter-2/3-reviewed
  Fernandina images + one explicitly-included small Fernandina dataset
  (FPM01_24012023, user-rated quality 4 in dataset_quality.md).
- Val   = 7 fresh non-Fernandina datasets (~430 images), DIFFERENT
  from al_v4's val so we get clean new review material for iter-4.

Outputs into /home/christian/data/training_data/2026_06_02_al_v5_canonical/
  al_v5_train_image_names.txt
  al_v5_val_image_names.txt
  al_v5_manifest.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from loguru import logger

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


MASTER = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json")
RSYNC = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/unzipped_images_rsync")
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")

OUT_DIR = Path("/home/christian/data/training_data/2026_06_02_al_v5_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

master_link = OUT_DIR / MASTER.name
if not master_link.exists():
    master_link.symlink_to(MASTER)

OUT_TRAIN = OUT_DIR / "al_v5_train_image_names.txt"
OUT_VAL = OUT_DIR / "al_v5_val_image_names.txt"
OUT_MANIFEST = OUT_DIR / "al_v5_manifest.json"

SEED = 42
DONE_STATUSES = {"COMPLETED", "DONE", "Done"}

# Fresh val: non-Fernandina, geographically diverse, modest sizes,
# none in al_v4's val list. Targets ~430-500 review images.
VAL_DATASETS = [
    "iseb05_19012023",          # Isabela east
    "flpr02_28012023",          # Floreana
    "pwc03_med_11122021",       # Pinta-Wolf-Cousteau
    "mbbe03_med_09122021",      # Marchena
    "iswc01_02_19012023",       # Isabela southwest coast
    "srecc02_12022023",         # San Cristobal
    "ispa05_15122021",          # Isabela Punta Albemarle
]

# Small Fernandina to explicitly include in training (user-rated 4/5).
FERN_INCLUDE = ["FPM01_24012023"]

# Reuse the al_v4 Fernandina classifier so behaviour is consistent.
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


def main() -> int:
    logger.info(f"Reading promoted master: {MASTER}")
    hA = HastyAnnotationV2.from_file(MASTER)
    logger.info(f"  {len(hA.images)} image records")

    pool_pairs: dict[tuple[str, str], int] = {}
    for idx, img in enumerate(hA.images):
        if not img.dataset_name or not img.image_name:
            continue
        if str(img.image_status) not in DONE_STATUSES:
            continue
        if not any(l.class_name == "iguana_point" and l.keypoints for l in img.labels):
            continue
        if not (RSYNC / img.dataset_name).is_dir():
            continue
        pool_pairs[(img.dataset_name, img.image_name)] = idx
    logger.info(f"Labeled pool: {len(pool_pairs)} (dataset, image_name) pairs")

    all_datasets = sorted({d for d, _ in pool_pairs.keys()})
    fern_datasets = {d for d in all_datasets if is_fernandina(d)}
    non_fern_datasets = {d for d in all_datasets if not is_fernandina(d)}
    logger.info(f"Datasets: Fern={len(fern_datasets)}, non-Fern={len(non_fern_datasets)}")

    # Iter-2 + iter-3 reviewed tiles from edit log.
    reviewed_pairs: set[tuple[str, str]] = set()
    if EDIT_LOG.exists():
        with EDIT_LOG.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("iteration_id") not in (
                    "phase15_iter0_val_test",
                    "phase15_iter1_val_test",
                    "phase15_iter2_al_v3",
                    "phase15_iter3_al_v4",
                ):
                    continue
                tile = row.get("image", "")
                if "___" not in tile:
                    continue
                ds, name = tile.split("___", 1)
                reviewed_pairs.add((ds, name))
    logger.info(f"All phase15-reviewed (dataset, image_name) pairs: {len(reviewed_pairs)}")
    reviewed_fern = {p for p in reviewed_pairs if is_fernandina(p[0])}
    logger.info(f"  of which Fernandina: {len(reviewed_fern)}")

    # Sanity-check VAL_DATASETS and FERN_INCLUDE.
    for d in VAL_DATASETS:
        if d not in all_datasets:
            logger.warning(f"VAL dataset {d!r} not in labeled pool — will yield 0 images")
        if is_fernandina(d):
            logger.warning(f"VAL dataset {d!r} is Fernandina — re-check intent")
    for d in FERN_INCLUDE:
        if d not in all_datasets:
            logger.warning(f"FERN_INCLUDE {d!r} not in labeled pool")
        if not is_fernandina(d):
            logger.warning(f"FERN_INCLUDE {d!r} is NOT Fernandina — re-check intent")

    val_set = set(VAL_DATASETS)
    fern_keep = set(FERN_INCLUDE)

    train_pairs: set[tuple[str, str]] = set()
    for (ds, name) in pool_pairs.keys():
        if ds in val_set:
            continue
        if is_fernandina(ds):
            if ds in fern_keep or (ds, name) in reviewed_fern:
                train_pairs.add((ds, name))
        else:
            train_pairs.add((ds, name))

    val_pairs = {(ds, name) for (ds, name) in pool_pairs.keys() if ds in val_set}

    train_names = {n for _, n in train_pairs}
    val_names = {n for _, n in val_pairs}

    overlap = train_names & val_names
    if overlap:
        logger.warning(f"{len(overlap)} cross-dataset name collisions removed from train")
        train_names -= overlap
    train_pairs_final = {(d, n) for (d, n) in train_pairs if n not in overlap}

    OUT_TRAIN.write_text("\n".join(sorted(train_names)) + "\n")
    OUT_VAL.write_text("\n".join(sorted(val_names)) + "\n")

    train_by_ds = defaultdict(int)
    for ds, _ in train_pairs_final:
        train_by_ds[ds] += 1
    val_by_ds = defaultdict(int)
    for ds, _ in val_pairs:
        val_by_ds[ds] += 1

    fern_in_train = {d for d, _ in train_pairs_final if is_fernandina(d)}
    OUT_MANIFEST.write_text(json.dumps({
        "master": str(MASTER),
        "pool_pairs": len(pool_pairs),
        "non_fernandina_dataset_count": len(non_fern_datasets),
        "fernandina_dataset_count": len(fern_datasets),
        "reviewed_pairs_all_iters": len(reviewed_pairs),
        "reviewed_fern_pairs": len(reviewed_fern),
        "val_datasets": VAL_DATASETS,
        "fern_explicitly_included": FERN_INCLUDE,
        "fern_datasets_in_train": sorted(fern_in_train),
        "train_pairs": len(train_pairs_final),
        "train_distinct_image_names": len(train_names),
        "val_pairs": len(val_pairs),
        "val_distinct_image_names": len(val_names),
        "val_per_dataset": dict(val_by_ds),
        "train_per_dataset_top20": dict(sorted(train_by_ds.items(),
                                                key=lambda x: -x[1])[:20]),
        "name_overlap_removed_from_train": sorted(overlap),
        "seed": SEED,
    }, indent=2))

    logger.info(f"Train: {len(train_pairs_final)} pairs / {len(train_names)} names")
    logger.info(f"  Fernandina datasets in train: {sorted(fern_in_train)}")
    logger.info(f"Val:   {len(val_pairs)} pairs / {len(val_names)} names")
    logger.info(f"Wrote {OUT_TRAIN}, {OUT_VAL}, {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
