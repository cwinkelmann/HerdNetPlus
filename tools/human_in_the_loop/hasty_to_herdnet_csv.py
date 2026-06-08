"""
Convert a HastyAnnotationV2 master into a per-tile herdnet_format.csv.

Phase-13 tiles use ``<dataset>___<image_name>`` filenames while the Hasty
master stores bare ``image_name`` records with ``dataset_name`` as a
separate field. This helper splits each tile name on ``___`` and looks
up the (dataset, image_name) pair in the master, extracting all
iguana-class keypoints. Non-iguana classes (e.g. ``not_iguana_but_similar_look``
hard negatives) are *skipped* — the herdnet evaluator only sees positives.

Output CSV columns match the existing herdnet_format.csv: ``images, x, y,
species, labels``.

Used after a Phase-15 correction round to regenerate the GT CSV that
``100_HIT_phase15_upload.py`` consumes — so the next upload reflects the
reviewer's corrections without re-running the full hasty -> herdnet
export pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


IGUANA_CLASSES = {"iguana_point", "iguana"}


def hasty_to_herdnet_csv(
    hasty_path: Path,
    images_dir: Path,
    output_csv: Path,
    species_value: str = "iguana_point",
    label_value: int = 1,
) -> int:
    """Generate a herdnet_format.csv for every tile in images_dir.

    Returns the number of rows written.
    """
    logger.info(f"Loading master Hasty: {hasty_path}")
    hA = HastyAnnotationV2.from_file(hasty_path)
    by_dataset_and_name: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
    for img in hA.images:
        if not img.image_name or not img.dataset_name:
            continue
        keypoints: list[tuple[float, float, str]] = []
        for label in img.labels:
            if label.class_name not in IGUANA_CLASSES:
                continue
            # Phase 15 explicitly operates on the keypoint annotations only;
            # the master also contains ~20k `iguana` bbox labels that we
            # leave UNTOUCHED in this round (the reviewer is fixing points,
            # not boxes). Skip any label without a keypoints array — that
            # way a box centroid never gets emitted as a point and never
            # ends up as a green marker that could be accidentally deleted
            # via CVAT and remove the box from master.
            if not label.keypoints:
                continue
            for kp in label.keypoints:
                if kp.x is None or kp.y is None:
                    continue
                keypoints.append((float(kp.x), float(kp.y), label.class_name))
        if keypoints:
            by_dataset_and_name[(img.dataset_name, img.image_name)] = keypoints

    rows: list[dict] = []
    n_tiles_with_gt = 0
    n_tiles_no_match = 0
    for tile_path in sorted(images_dir.iterdir()):
        if not tile_path.is_file():
            continue
        tile_name = tile_path.name
        if "___" not in tile_name:
            n_tiles_no_match += 1
            continue
        ds, hasty_name = tile_name.split("___", 1)
        kps = by_dataset_and_name.get((ds, hasty_name))
        if not kps:
            continue
        n_tiles_with_gt += 1
        for x, y, _class in kps:
            rows.append({
                "images": tile_name,
                "x": int(round(x)),
                "y": int(round(y)),
                "species": species_value,
                "labels": label_value,
            })

    df = pd.DataFrame(rows, columns=["images", "x", "y", "species", "labels"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(
        f"Wrote {len(df)} rows ({n_tiles_with_gt} tiles with GT, "
        f"{n_tiles_no_match} tiles couldn't be matched to a (dataset, image) "
        f"pair) -> {output_csv}"
    )
    return len(df)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hasty", required=True, type=Path)
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--species", default="iguana_point")
    args = p.parse_args()
    hasty_to_herdnet_csv(args.hasty, args.images_dir, args.output, args.species)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
