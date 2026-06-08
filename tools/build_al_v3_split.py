"""
Build active-learning iteration-3 split: image-level sample.

  Pool          = master images with iguana_point kp + COMPLETED/DONE +
                  raw on disk
  Corrected     = ~225 tiles already reviewed in phase-15 iter-0 + iter-1
                  (extracted from data/phase15_edit_log.csv).
                  Their image_names are EXCLUDED from val sampling so
                  the corrected instances flow into train.
  Val           = 3000 random image_names sampled from
                  (Pool image_names - Corrected image_names)
  Train         = Pool image_names - Val image_names
                  (includes the 225 corrected, by construction)

Outputs are bare image_name lists (one per line). The canonical 021
pipeline matches on bare image_name via DatasetFilterConfig.images_filter.
There are 28 cross-dataset image_name collisions; since the same name
goes into the same split for ALL its dataset instances, no cross-split
contamination occurs.

Writes:
  <labels_dir>/al_v3_train_image_names.txt   (~7,310 names)
  <labels_dir>/al_v3_val_image_names.txt     (3,000 names)
  <labels_dir>/al_v3_manifest.json
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from loguru import logger

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2
from com.biospheredata.types.status import LabelingStatus


MASTER = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json")
RSYNC = Path("/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/unzipped_images_rsync")
EDIT_LOG = Path("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")

OUT_DIR = Path("/home/christian/data/training_data/2026_06_01_al_v3_canonical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Symlink master into output dir so the canonical pipeline can find it
# under labels_path/labels_name.
master_link = OUT_DIR / MASTER.name
if not master_link.exists():
    master_link.symlink_to(MASTER)

OUT_TRAIN_LIST = OUT_DIR / "al_v3_train_image_names.txt"
OUT_VAL_LIST = OUT_DIR / "al_v3_val_image_names.txt"
OUT_MANIFEST = OUT_DIR / "al_v3_manifest.json"

SEED = 42
VAL_SIZE = 3000

DONE_STATUSES = {"COMPLETED", "DONE", "Done"}


def parse_tile(tile_name: str) -> tuple[str, str] | None:
    if "___" not in tile_name:
        return None
    ds, name = tile_name.split("___", 1)
    return ds, name


def main() -> int:
    logger.info(f"Reading master: {MASTER}")
    hA = HastyAnnotationV2.from_file(MASTER)
    logger.info(f"  {len(hA.images)} image records")

    # Pool: iguana_point kp + COMPLETED/DONE + raw on disk.
    pool_pairs: set[tuple[str, str]] = set()
    pool_names: set[str] = set()
    name_to_datasets: dict[str, set[str]] = defaultdict(set)
    for img in hA.images:
        if not img.dataset_name or not img.image_name:
            continue
        if str(img.image_status) not in DONE_STATUSES:
            continue
        if not any(lab.class_name == "iguana_point" and lab.keypoints for lab in img.labels):
            continue
        if not (RSYNC / img.dataset_name).is_dir():
            continue
        pool_pairs.add((img.dataset_name, img.image_name))
        pool_names.add(img.image_name)
        name_to_datasets[img.image_name].add(img.dataset_name)
    logger.info(f"Pool: {len(pool_pairs)} (dataset, image_name) pairs, "
                f"{len(pool_names)} distinct image_names")

    # Already-corrected pairs from phase-15 edit log.
    corrected_pairs: set[tuple[str, str]] = set()
    with EDIT_LOG.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("iteration_id") not in ("phase15_iter0_val_test", "phase15_iter1_val_test"):
                continue
            parsed = parse_tile(row.get("image", ""))
            if parsed:
                corrected_pairs.add(parsed)
    corrected_in_pool = corrected_pairs & pool_pairs
    corrected_names = {n for _, n in corrected_in_pool}
    logger.info(f"Corrected pairs: {len(corrected_pairs)}; "
                f"in pool: {len(corrected_in_pool)}; "
                f"distinct image_names: {len(corrected_names)}")

    # Val sampling: only names that have NO corrected instance.
    val_candidate_names = sorted(pool_names - corrected_names)
    rng = random.Random(SEED)
    rng.shuffle(val_candidate_names)
    val_names = set(val_candidate_names[:VAL_SIZE])
    train_names = pool_names - val_names
    logger.info(f"Sampled: val={len(val_names)} names, train={len(train_names)} names")

    # Sanity asserts
    assert not (val_names & train_names), "val/train overlap"
    assert corrected_names.issubset(train_names), "corrected leaked to val"
    assert val_names | train_names == pool_names, "names not partitioned"

    # Count (dataset, image_name) pairs that will land in each split.
    train_pairs = {(d, n) for (d, n) in pool_pairs if n in train_names}
    val_pairs = {(d, n) for (d, n) in pool_pairs if n in val_names}
    logger.info(f"Will produce: train={len(train_pairs)} pairs, val={len(val_pairs)} pairs")

    OUT_TRAIN_LIST.write_text("\n".join(sorted(train_names)) + "\n")
    OUT_VAL_LIST.write_text("\n".join(sorted(val_names)) + "\n")
    logger.info(f"Wrote {OUT_TRAIN_LIST}")
    logger.info(f"Wrote {OUT_VAL_LIST}")

    OUT_MANIFEST.write_text(json.dumps({
        "pool_pairs": len(pool_pairs),
        "pool_distinct_names": len(pool_names),
        "corrected_pairs_in_pool": len(corrected_in_pool),
        "corrected_distinct_names": len(corrected_names),
        "val_names": len(val_names),
        "val_pairs": len(val_pairs),
        "train_names": len(train_names),
        "train_pairs": len(train_pairs),
        "seed": SEED,
        "master_source": str(MASTER),
    }, indent=2))
    logger.info(f"Wrote {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
