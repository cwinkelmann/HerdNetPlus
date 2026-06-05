"""
Build active-learning iteration-6 split (2026-06-03).

Strategy:
- Val is BIGGER and more diverse than al_v5: ~1300 images across both
  Fernandina and other-island datasets that haven't been heavily
  touched by previous phase-15 iterations.
- Train: same shape as al_v5 — non-Fer minus val + iter-2/3-reviewed
  Fer + FPM01_24012023.

Picks for val:
- Fernandina batch (~250 images): a few moderate-sized fer_* datasets we
  haven't reviewed (or only lightly reviewed).
- Other islands (~1000 images): mix of Floreana, Isabela, Espanola,
  Santa Cruz, San Cristobal, Pinta, Genovesa that haven't been in
  iter-0..iter-4 review pool.
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

OUT_DIR = Path("/home/christian/data/training_data/2026_06_03_al_v6_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

master_link = OUT_DIR / MASTER.name
if not master_link.exists():
    master_link.symlink_to(MASTER)

OUT_TRAIN = OUT_DIR / "al_v6_train_image_names.txt"
OUT_VAL = OUT_DIR / "al_v6_val_image_names.txt"
OUT_MANIFEST = OUT_DIR / "al_v6_manifest.json"

SEED = 42
DONE_STATUSES = {"COMPLETED", "DONE", "Done"}

# Val datasets — bigger, mix of Fer + non-Fer, picked to be fresh
# (low or no overlap with iter-0..iter-4 reviewed pairs).
VAL_DATASETS_FERN = [
    "fer_fnj02_03_04_19122021",     # ~164
    "fer_fwk04_21122021",           # ~163
    "fer_fnf02_19122021",           # ~62
    "fer_fpe08_18122021",           # ~53
]

VAL_DATASETS_OTHER = [
    # Floreana
    "flscj02_03022021",             # ~112
    "flpvin02_28012023",            # ~60
    "flsca03_23012021",             # ~46
    # Espanola
    "egi04_05_27012024",            # ~106
    "esck05_10022023",              # ~106
    "esp_em06_26012021",            # ~10
    # Isabela
    "ispa11_12_13_15122021",        # ~439 (big — densest colonies)
    "isa_isvi01_27012023",          # ~86
    "isa_isvb04_27012023",          # ~20
    # Santa Cruz
    "scpa01_06012023",              # ~33
    "sccd01_02_13012023",           # ~51
    # San Cristobal
    "SCris_SRIL01_04022023",        # ~36
    # Marchena
    "mar_mbbe04_09122021",          # ~18
    # Pinta/Pinzon
    "pcie08_medium_10122021",       # ~72
    # Genovesa/wolf
    "gwa01_med_05122021",           # ~77
]

VAL_DATASETS = VAL_DATASETS_FERN + VAL_DATASETS_OTHER

# Small Fernandina explicitly included in TRAIN even though we're
# adding more Fer to val now (it's the clean Jan-2023 reflight,
# quality 4/5 per dataset_quality.md).
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


def main() -> int:
    logger.info(f"Reading promoted master: {MASTER}")
    hA = HastyAnnotationV2.from_file(MASTER)

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
    logger.info(f"Labeled pool: {len(pool_pairs)} pairs")

    reviewed_pairs: set[tuple[str, str]] = set()
    if EDIT_LOG.exists():
        with EDIT_LOG.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("iteration_id") not in (
                    "phase15_iter0_val_test", "phase15_iter1_val_test",
                    "phase15_iter2_al_v3", "phase15_iter3_al_v4",
                    "phase15_iter4_al_v5",
                ):
                    continue
                tile = row.get("image", "")
                if "___" not in tile:
                    continue
                ds, name = tile.split("___", 1)
                reviewed_pairs.add((ds, name))
    reviewed_fern = {p for p in reviewed_pairs if is_fernandina(p[0])}
    logger.info(f"All phase15-reviewed pairs: {len(reviewed_pairs)} (Fer: {len(reviewed_fern)})")

    # Pre-flight: warn if any val dataset overlaps with reviewed
    for d in VAL_DATASETS:
        n_reviewed_in_d = sum(1 for (ds, _) in reviewed_pairs if ds == d)
        if n_reviewed_in_d > 0:
            logger.warning(f"VAL {d!r} already has {n_reviewed_in_d} reviewed pairs in phase15 history")
        if d not in {ds for ds, _ in pool_pairs.keys()}:
            logger.warning(f"VAL {d!r} not in labeled pool — will yield 0 images")

    val_set = set(VAL_DATASETS)
    fern_keep_in_train = set(FERN_INCLUDE_IN_TRAIN)

    train_pairs: set[tuple[str, str]] = set()
    for (ds, name) in pool_pairs.keys():
        if ds in val_set:
            continue
        if is_fernandina(ds):
            if ds in fern_keep_in_train or (ds, name) in reviewed_fern:
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

    val_by_ds = defaultdict(int)
    for ds, _ in val_pairs:
        val_by_ds[ds] += 1
    val_by_ds_sorted = dict(sorted(val_by_ds.items(), key=lambda x: -x[1]))

    OUT_MANIFEST.write_text(json.dumps({
        "master": str(MASTER),
        "pool_pairs": len(pool_pairs),
        "reviewed_pairs_all_iters": len(reviewed_pairs),
        "reviewed_fern_pairs": len(reviewed_fern),
        "val_datasets_fern": VAL_DATASETS_FERN,
        "val_datasets_other": VAL_DATASETS_OTHER,
        "fern_explicitly_included_in_train": FERN_INCLUDE_IN_TRAIN,
        "train_pairs": len(train_pairs_final),
        "train_distinct_image_names": len(train_names),
        "val_pairs": len(val_pairs),
        "val_distinct_image_names": len(val_names),
        "val_per_dataset": val_by_ds_sorted,
        "name_overlap_removed_from_train": sorted(overlap),
        "seed": SEED,
    }, indent=2))

    logger.info(f"Train: {len(train_pairs_final)} pairs / {len(train_names)} names")
    logger.info(f"Val:   {len(val_pairs)} pairs / {len(val_names)} names")
    logger.info("Val per dataset (top 20):")
    for ds, n in list(val_by_ds_sorted.items())[:20]:
        logger.info(f"  {ds:<35} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
