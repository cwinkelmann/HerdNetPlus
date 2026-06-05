"""
Build active-learning iteration-4 split.

Strategy (per user 2026-06-02):
- INCLUDE all non-Fernandina datasets in training.
- EXCLUDE all UN-reviewed Fernandina datasets (they're poison until cleaned).
- INCLUDE only the Fernandina images that were hand-reviewed in
  phase-15 iter-2 (the 200 we just promoted).
- Set aside a handful of non-Fernandina datasets for VAL (next iter-3
  correction target).

The classification rule for "Fernandina":
  * any dataset_name starting with `fer_` or `Fer_`
  * `Fer_FCD01-02-03_20122021*`, `Fer_FPM01-02_20122023`, `Fer_FPE02_07052024`,
    `FPM01_24012023` — explicit Fer site codes
  * `fpe*` — FPE site is Fernandina (per active-learning dataset_mapping.py)
  * `floreana_FPE01_FECA01` — explicitly noted as Fernandina-despite-name in mapping
  * Other `fs*` / `fwe*` / `fso*` / `fsa*` / `fsd*` / `fsg*` / `fsj*` —
    ASSUMED Fernandina based on naming convention; user should verify.

Outputs (written to OUT_DIR):
  al_v4_train_image_names.txt — bare image_names for train
  al_v4_val_image_names.txt   — bare image_names for val
  al_v4_manifest.json         — full provenance breakdown
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from loguru import logger

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


MASTER = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json")
RSYNC = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/unzipped_images_rsync")
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")

OUT_DIR = Path("/home/christian/data/training_data/2026_06_02_al_v4_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Symlink master into output dir so the canonical 021 pipeline can find it.
master_link = OUT_DIR / MASTER.name
if not master_link.exists():
    master_link.symlink_to(MASTER)

OUT_TRAIN = OUT_DIR / "al_v4_train_image_names.txt"
OUT_VAL = OUT_DIR / "al_v4_val_image_names.txt"
OUT_MANIFEST = OUT_DIR / "al_v4_manifest.json"

SEED = 42
DONE_STATUSES = {"COMPLETED", "DONE", "Done"}

# Held-out for next iteration's CVAT correction. Mix of geographic diversity,
# moderate sizes, all non-Fernandina. Update if you want a different
# review target.
VAL_DATASETS = [
    "isa_isvp01_27012023",   # Isabela
    "iscw01_25012023",       # Isabela coastal
    "isncw02_25012023",      # Isabela north coast
    "scbb02_03_13012023",    # Santa Cruz
    "mbn09_med_06122021",    # Marchena
    "pcie05_medium_09122021",  # Pinta
]


def is_fernandina(ds: str) -> bool:
    """Best-effort Fernandina classifier. User should verify."""
    if not ds:
        return False
    ds_lo = ds.lower()
    if ds_lo.startswith("fer_") or ds.startswith("Fer_"):
        return True
    # Explicit single-image / orthomosaic Fernandina names
    if ds in {
        "Fer_FCD01-02-03_20122021",
        "Fer_FCD01-02-03_20122021_single_images",
        "Fer_FPM01-02_20122023",
        "Fer_FPE02_07052024",
        "FPM01_24012023",
        "floreana_FPE01_FECA01",  # mapping says "yes Fernandina despite the name"
    }:
        return True
    # FPE / FSA / FSD / FSG / FSJ / FSO / FWE site codes — confirmed
    # Fernandina per dataset_mapping.py's Fernandina_m_fpe entry, and
    # naming-convention extrapolation for the other f* siblings.
    for pfx in ("fpe", "fsa", "fsd", "fsg", "fsj", "fso", "fwe"):
        if ds_lo.startswith(pfx):
            return True
    return False


def main() -> int:
    logger.info(f"Reading promoted master: {MASTER}")
    hA = HastyAnnotationV2.from_file(MASTER)
    logger.info(f"  {len(hA.images)} image records")

    # Build the labeled pool.
    pool_pairs: dict[tuple[str, str], int] = {}
    pool_name_to_datasets: dict[str, set[str]] = defaultdict(set)
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
        pool_name_to_datasets[img.image_name].add(img.dataset_name)
    logger.info(f"Labeled pool: {len(pool_pairs)} (dataset, image_name) pairs")

    # Classify each dataset.
    all_datasets = sorted({d for d, _ in pool_pairs.keys()})
    fern_datasets = {d for d in all_datasets if is_fernandina(d)}
    non_fern_datasets = {d for d in all_datasets if not is_fernandina(d)}
    logger.info(
        f"Datasets in pool: Fernandina={len(fern_datasets)}, "
        f"non-Fernandina={len(non_fern_datasets)}"
    )

    # Collect iter-2 reviewed image tiles from edit log.
    iter2_pairs: set[tuple[str, str]] = set()
    if EDIT_LOG.exists():
        with EDIT_LOG.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("iteration_id") != "phase15_iter2_al_v3":
                    continue
                tile = row.get("image", "")
                if "___" not in tile:
                    continue
                ds, name = tile.split("___", 1)
                iter2_pairs.add((ds, name))
    logger.info(f"Iter-2 reviewed (dataset, image_name) pairs: {len(iter2_pairs)}")

    # Of the iter-2 reviewed, which are Fernandina?
    iter2_fern_pairs = {p for p in iter2_pairs if is_fernandina(p[0])}
    logger.info(f"  of which Fernandina: {len(iter2_fern_pairs)}")

    # Val datasets — pre-flight check.
    for d in VAL_DATASETS:
        if d not in non_fern_datasets:
            logger.warning(
                f"VAL candidate {d!r} is NOT in non-Fernandina pool "
                f"(is_fernandina={is_fernandina(d)}, in pool={d in all_datasets})"
            )

    # ----- Build train set -----
    # Train = all non-Fernandina pool images, MINUS val datasets,
    # PLUS iter-2 reviewed Fernandina images.
    val_dataset_set = set(VAL_DATASETS)
    train_pairs: set[tuple[str, str]] = set()
    for (ds, name) in pool_pairs.keys():
        if ds in val_dataset_set:
            continue
        if is_fernandina(ds):
            if (ds, name) in iter2_fern_pairs:
                train_pairs.add((ds, name))
            # else: excluded (un-reviewed Fernandina)
        else:
            train_pairs.add((ds, name))

    val_pairs = {(ds, name) for (ds, name) in pool_pairs.keys() if ds in val_dataset_set}

    train_names = {n for _, n in train_pairs}
    val_names = {n for _, n in val_pairs}

    # Safety: a bare image_name might appear in BOTH a train dataset and
    # a val dataset (cross-dataset collisions exist — 28 total in master).
    # Drop any colliding name from train (val wins).
    name_overlap = train_names & val_names
    if name_overlap:
        logger.warning(
            f"{len(name_overlap)} image_names collide across train+val. "
            f"Removing from train so val stays clean."
        )
        train_names -= name_overlap

    train_pairs_final = {(ds, n) for (ds, n) in train_pairs if n not in name_overlap}

    OUT_TRAIN.write_text("\n".join(sorted(train_names)) + "\n")
    OUT_VAL.write_text("\n".join(sorted(val_names)) + "\n")

    # Per-island breakdown for the manifest.
    train_by_ds = defaultdict(int)
    for ds, _ in train_pairs_final:
        train_by_ds[ds] += 1
    val_by_ds = defaultdict(int)
    for ds, _ in val_pairs:
        val_by_ds[ds] += 1

    OUT_MANIFEST.write_text(json.dumps({
        "master": str(MASTER),
        "pool_pairs": len(pool_pairs),
        "fernandina_datasets": sorted(fern_datasets),
        "fernandina_dataset_count": len(fern_datasets),
        "non_fernandina_datasets": sorted(non_fern_datasets),
        "non_fernandina_dataset_count": len(non_fern_datasets),
        "iter2_reviewed_pairs": len(iter2_pairs),
        "iter2_reviewed_fern_pairs": len(iter2_fern_pairs),
        "val_datasets": VAL_DATASETS,
        "val_pairs": len(val_pairs),
        "val_distinct_image_names": len(val_names),
        "val_per_dataset": dict(val_by_ds),
        "train_pairs": len(train_pairs_final),
        "train_distinct_image_names": len(train_names),
        "train_per_dataset_top20": dict(sorted(train_by_ds.items(),
                                                key=lambda x: -x[1])[:20]),
        "name_overlap_removed_from_train": sorted(name_overlap),
        "seed": SEED,
    }, indent=2))

    logger.info(f"Train: {len(train_pairs_final)} pairs, {len(train_names)} distinct image_names")
    logger.info(f"Val:   {len(val_pairs)} pairs, {len(val_names)} distinct image_names")
    logger.info(f"Wrote {OUT_TRAIN}, {OUT_VAL}, {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
