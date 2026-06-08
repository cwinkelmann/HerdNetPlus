"""
Slice the active-learning iteration-2 train/val from current corrected master.

New layout (/home/christian/data/training_data/2026_06_01_al_v2/):
  train/Default/  — symlinks to all of phase-13 train_Nfull + val + test
                    (the previously corrected pool, now used as training)
  val/Default/    — symlinks to ~1028 tiles carved from train_Nfull,
                    stratified by dataset (within-dataset random sample
                    proportional to dataset size). This is the next
                    chunk the human will correct.
  train/herdnet_format.csv  — regenerated from the current Hasty master
  val/herdnet_format.csv    — regenerated from the current Hasty master

The corrected master at /data/.../2026_05_06_labels.json is the source
of truth for labels. Tiles in train_Nfull whose source images aren't in
the master will simply have no labels (and so contribute no positives).

Stratification: per-dataset random sample. Source-image grouping is
trivial because in this dataset each tile filename IS one source image
(no multi-tile-per-image leakage to worry about within train_Nfull).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

sys.path.append(str(Path(__file__).parent / "human_in_the_loop"))
from hasty_to_herdnet_csv import hasty_to_herdnet_csv


SOURCE_ROOT = Path("/home/christian/data/training_data/2026_05_08_data_scaling")
TRAIN_NFULL = SOURCE_ROOT / "train_Nfull_s42" / "Default"
OLD_VAL = SOURCE_ROOT / "val" / "Default"
OLD_TEST = SOURCE_ROOT / "test" / "Default"

DEST_ROOT = Path("/home/christian/data/training_data/2026_06_01_al_v2")
NEW_TRAIN = DEST_ROOT / "train" / "Default"
NEW_VAL = DEST_ROOT / "val" / "Default"

HASTY_MASTER = Path(
    "/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json"
)


def parse_dataset(tile_name: str) -> str:
    return tile_name.split("___", 1)[0]


def stratified_split(
    tiles: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Within-dataset random sample. Returns (train_part, val_part)."""
    rng = random.Random(seed)
    by_ds: dict[str, list[str]] = defaultdict(list)
    for t in tiles:
        by_ds[parse_dataset(t)].append(t)
    train_part, val_part = [], []
    for ds in sorted(by_ds.keys()):
        names = sorted(by_ds[ds])
        rng.shuffle(names)
        k = max(1, round(len(names) * val_fraction)) if len(names) >= 2 else 0
        val_part.extend(names[:k])
        train_part.extend(names[k:])
    return train_part, val_part


def symlink_all(names: list[str], src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for nm in names:
        src = src_dir / nm
        if not src.is_file():
            logger.warning(f"Source missing, skipping: {src}")
            continue
        tgt = dst_dir / nm
        if tgt.exists() or tgt.is_symlink():
            continue
        tgt.symlink_to(src.resolve())
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val-fraction", type=float, default=0.149,
                   help="fraction of train_Nfull per dataset to pull into new val "
                        "(default ≈ 1028/6908)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not TRAIN_NFULL.is_dir():
        logger.error(f"Missing source: {TRAIN_NFULL}"); return 1
    if not HASTY_MASTER.is_file():
        logger.error(f"Missing master: {HASTY_MASTER}"); return 1

    train_nfull_tiles = sorted([f.name for f in TRAIN_NFULL.iterdir() if f.is_file()])
    old_val_tiles = sorted([f.name for f in OLD_VAL.iterdir() if f.is_file()])
    old_test_tiles = sorted([f.name for f in OLD_TEST.iterdir() if f.is_file()])
    logger.info(f"Source counts: train_Nfull={len(train_nfull_tiles)}, "
                f"old_val={len(old_val_tiles)}, old_test={len(old_test_tiles)}")

    # Stratified carve of new val from train_Nfull
    new_train_from_nfull, new_val_tiles = stratified_split(
        train_nfull_tiles, args.val_fraction, args.seed,
    )
    logger.info(f"Stratified split (val_fraction={args.val_fraction}, seed={args.seed}): "
                f"train_from_nfull={len(new_train_from_nfull)}, new_val={len(new_val_tiles)}")

    # New train = (train_Nfull − new_val) ∪ old_val ∪ old_test
    new_train_set = set(new_train_from_nfull)
    new_train_set.update(old_val_tiles)
    new_train_set.update(old_test_tiles)
    new_train_list = sorted(new_train_set)
    logger.info(f"new_train total: {len(new_train_list)}")

    # Symlink staging
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    n_t1 = symlink_all(new_train_from_nfull, TRAIN_NFULL, NEW_TRAIN)
    n_t2 = symlink_all(old_val_tiles, OLD_VAL, NEW_TRAIN)
    n_t3 = symlink_all(old_test_tiles, OLD_TEST, NEW_TRAIN)
    logger.info(f"symlinked into new_train: from Nfull={n_t1} + from old_val={n_t2} + from old_test={n_t3}")
    n_v = symlink_all(new_val_tiles, TRAIN_NFULL, NEW_VAL)
    logger.info(f"symlinked into new_val: {n_v}")

    # Per-dataset breakdown of new_val (so the user can see coverage)
    by_ds = defaultdict(int)
    for t in new_val_tiles:
        by_ds[parse_dataset(t)] += 1
    logger.info("new_val coverage by dataset (sorted desc):")
    for ds, n in sorted(by_ds.items(), key=lambda x: -x[1]):
        logger.info(f"  {n:>4}  {ds}")

    # Generate herdnet_format.csv for each split from the current master
    train_csv = DEST_ROOT / "train" / "herdnet_format.csv"
    val_csv = DEST_ROOT / "val" / "herdnet_format.csv"
    n_train_rows = hasty_to_herdnet_csv(HASTY_MASTER, NEW_TRAIN, train_csv)
    n_val_rows = hasty_to_herdnet_csv(HASTY_MASTER, NEW_VAL, val_csv)
    logger.info(f"Wrote {train_csv}: {n_train_rows} rows")
    logger.info(f"Wrote {val_csv}:   {n_val_rows} rows")

    # Manifest
    manifest = DEST_ROOT / "manifest.json"
    import json
    manifest.write_text(json.dumps({
        "created_from_master": str(HASTY_MASTER),
        "source_root": str(SOURCE_ROOT),
        "new_train_total": len(new_train_list),
        "new_train_from_nfull": len(new_train_from_nfull),
        "new_train_from_old_val": len(old_val_tiles),
        "new_train_from_old_test": len(old_test_tiles),
        "new_val_total": len(new_val_tiles),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "new_val_by_dataset": dict(by_ds),
    }, indent=2))
    logger.info(f"Wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
