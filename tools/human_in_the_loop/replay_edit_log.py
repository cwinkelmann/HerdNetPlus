"""
Reconstruct an updated master Hasty by replaying a Phase-15 edit log.

The download script (101_HIT_phase15_download.py) had a bug where it
deep-copied hA_master after the tile->master mapping had been built —
so the apply step mutated dead references and the saved file was
pristine. The edit log on disk, however, faithfully records every
intended edit. This helper replays the log to recover the corrected
master without needing to round-trip through CVAT again.

Usage:
    python tools/human_in_the_loop/replay_edit_log.py \
        --master /data/.../2026_05_06_labels.json \
        --edit-log /home/.../data/phase15_edit_log.csv \
        --mapping /home/.../report/phase15_iter0_..._image_to_dataset.json \
        --iteration-id phase15_iter0_val_test \
        --out /data/.../2026_05_06_labels_phase15_iter0_corrected.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import uuid
from pathlib import Path

from loguru import logger

from com.biospheredata.types.HastyAnnotationV2 import (
    AnnotatedImage,
    HastyAnnotationV2,
    ImageLabel,
    Keypoint,
)


CANONICAL_CLASS_NAME = "iguana_point"
HARD_NEG_CLASS_NAME = "not_iguana_but_similar_look"


def _keypoint_xy(label: ImageLabel) -> tuple[float | None, float | None]:
    if label.keypoints:
        kp = label.keypoints[0]
        if kp.x is not None and kp.y is not None:
            return float(kp.x), float(kp.y)
    return None, None


def _remove_keypoint_near(image: AnnotatedImage, xy: tuple[float, float], radius: float) -> bool:
    """Remove the first keypoint-bearing label within radius. NEVER touches boxes."""
    if xy[0] is None:
        return False
    new_labels = []
    removed = False
    for l in image.labels:
        if removed or not l.keypoints:
            new_labels.append(l); continue
        kp = l.keypoints[0]
        if kp.x is None or kp.y is None:
            new_labels.append(l); continue
        if math.hypot(float(kp.x) - xy[0], float(kp.y) - xy[1]) <= radius:
            removed = True
            continue
        new_labels.append(l)
    image.labels = new_labels
    return removed


def _relocate_keypoint_near(image: AnnotatedImage, xy_old, xy_new, radius: float) -> bool:
    if xy_old[0] is None or xy_new[0] is None:
        return False
    for l in image.labels:
        if not l.keypoints:
            continue
        kp = l.keypoints[0]
        if kp.x is None or kp.y is None:
            continue
        if math.hypot(float(kp.x) - xy_old[0], float(kp.y) - xy_old[1]) <= radius:
            kp.x = int(round(xy_new[0]))
            kp.y = int(round(xy_new[1]))
            return True
    return False


def _add_keypoint(image: AnnotatedImage, xy, class_name: str) -> None:
    if xy[0] is None:
        return
    kp = Keypoint(
        id=str(uuid.uuid4()),
        x=int(round(xy[0])),
        y=int(round(xy[1])),
        keypoint_class_id="body",
    )
    image.labels.append(ImageLabel(
        id=str(uuid.uuid4()),
        class_name=class_name,
        keypoints=[kp],
        attributes={"source": "phase15_review"},
    ))


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--edit-log", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True,
                    help="tile_name -> {dataset_name, hasty_image_name} json")
    ap.add_argument("--iteration-id", required=True,
                    help="only rows with iteration_id == this are replayed")
    ap.add_argument("--timestamp-prefix",
                    help="optional ISO-timestamp prefix; only rows whose "
                         "timestamp starts with this string are replayed. "
                         "Use to pick a single download run when the edit "
                         "log accumulated multiple.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    logger.info(f"Loading master: {args.master}")
    hA_master = HastyAnnotationV2.from_file(args.master)
    logger.info(f"  {len(hA_master.images)} images, "
                f"{sum(len(img.labels) for img in hA_master.images)} total labels")

    logger.info(f"Loading mapping: {args.mapping}")
    mapping = json.loads(args.mapping.read_text())

    # Build (dataset_name, hasty_image_name) -> AnnotatedImage lookup on the
    # ACTUAL master, then a tile_name -> AnnotatedImage lookup via mapping.
    master_by_key: dict[tuple[str, str], AnnotatedImage] = {}
    for img in hA_master.images:
        if img.image_name and img.dataset_name:
            master_by_key.setdefault((img.dataset_name, img.image_name), img)
    tile_to_master_image: dict[str, AnnotatedImage] = {}
    unmatched = 0
    for tile_name, route in mapping.items():
        key = (route["dataset_name"], route["hasty_image_name"])
        img = master_by_key.get(key)
        if img is None:
            unmatched += 1
            continue
        tile_to_master_image[tile_name] = img
    if unmatched:
        logger.error(f"{unmatched} mapping entries don't resolve in master — aborting")
        return 1
    logger.info(f"Resolved {len(tile_to_master_image)} tile names to master records.")

    # Replay the edit log.
    n_applied = {"A": 0, "B": 0, "C": 0, "H": 0, "skipped": 0}
    n_seen = 0
    n_skip_other_iter = 0
    with args.edit_log.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("iteration_id") != args.iteration_id:
                n_skip_other_iter += 1
                continue
            if args.timestamp_prefix and not (row.get("timestamp", "") or "").startswith(args.timestamp_prefix):
                continue
            n_seen += 1
            category = row.get("category", "")
            image_name = row.get("image", "")
            if image_name not in tile_to_master_image:
                n_applied["skipped"] += 1
                continue
            master_img = tile_to_master_image[image_name]
            x_old = _to_float(row.get("x_old"))
            y_old = _to_float(row.get("y_old"))
            x_new = _to_float(row.get("x_new"))
            y_new = _to_float(row.get("y_new"))

            if category == "A":
                # Promote kept iguana to master at the (possibly relocated) position.
                pos = (x_new, y_new) if x_new is not None else (x_old, y_old)
                if pos[0] is not None:
                    _add_keypoint(master_img, pos, CANONICAL_CLASS_NAME)
                    n_applied["A"] += 1
            elif category == "B":
                if x_old is not None:
                    if _remove_keypoint_near(master_img, (x_old, y_old), radius=10):
                        n_applied["B"] += 1
            elif category == "C":
                if x_old is not None and x_new is not None:
                    if _relocate_keypoint_near(master_img, (x_old, y_old), (x_new, y_new), radius=10):
                        n_applied["C"] += 1
            elif category == "H":
                # Either deleted pred (x_new is None, position is x_old)
                # or kept-with-non-iguana-class (x_new set, position is x_new).
                if x_new is not None:
                    pos = (x_new, y_new)
                elif x_old is not None:
                    pos = (x_old, y_old)
                else:
                    continue
                _add_keypoint(master_img, pos, HARD_NEG_CLASS_NAME)
                n_applied["H"] += 1

    logger.info(f"Replayed iteration_id={args.iteration_id}: "
                f"{n_seen} rows seen ({n_skip_other_iter} other-iteration rows skipped)")
    logger.info(f"  Applied: A={n_applied['A']} B={n_applied['B']} "
                f"C={n_applied['C']} H={n_applied['H']} "
                f"(skipped {n_applied['skipped']} rows with unknown tile)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hA_master.save(args.out)
    logger.info(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
