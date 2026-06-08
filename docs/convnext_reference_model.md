# ConvNeXt reference model — training recipe

This is the **al_v8 E7** model: the current production candidate
(R=0.873, P=0.814, F1=0.842, F2=0.860 on the 1,123-image multi-island
val). Trained 2026-06-03 → 2026-06-04 on a single 4090. Used as the
inference model for phase-15 iter-7 review. Also serves as the
performance baseline that the al_v9 sweep is trying to beat.

This doc is a recipe — anyone should be able to reproduce a
near-identical run from it.

---

## 1. Architecture

**`CamouflageHerdNetConvNeXtV2`**, backbone size `tiny`.
Defined in `animaloc/models/herdnet_timm_convnext_camouflaged_v2.py`.
Registered via `@MODELS.register()` and instantiated by name in the
Hydra model config.

| component | value |
|---|---|
| Backbone | `timm/convnext_tiny.fcmae_ft_in22k_in1k` (ConvNeXt-V2 tiny) |
| Feature channels | [96, 192, 384, 768] over 4 stages |
| Feature-map sizes at 512 px input | 128×128 / 64×64 / 32×32 / 16×16 |
| Edge enhancement | ON |
| Multi-resolution head | ON, resolutions [256, 512, 786] |
| Gabor pre-filters | 8 filters (4 orientations × 2 frequencies), 32→96 ch fused into stage 0 |
| Detection head | Multi-scale + Edge enhancement + CBAM |
| Total params | 34,929,123 (all trainable) |
| Backbone params | 27,818,592 (trainable, no freezing) |
| `down_ratio` | 4 (output stride; feature map at 128×128 for 512-px input) |
| `num_classes` | 2 (background + iguana) |

ConvNeXt-V2 was chosen after explicit comparison against DLA34,
EfficientViT, and DINOv2 (see `.claude/skills/training-insights/PHASE_HISTORY.md`).
DINOv2's 14-px patch embedding destroys signal for small iguana
targets; DLA34 plateaus lower on F1. ConvNeXt-V2 wins on every
phase-13 sweep we ran.

---

## 2. Loss

Two-component, summed with equal lambda (`lambda_const=1.0` each):

```yaml
FocalLoss:
  output_idx: 0
  target_idx: 0          # FIDT heatmap target
  kwargs:
    alpha: 2
    beta: 4
    reduction: mean
    normalize: false

CrossEntropyLoss:
  output_idx: 1
  target_idx: 1          # PointsToMask target (binary mask, down_ratio=32)
  background_class_weight: 0.1
  kwargs:
    weight: [0.1, 5.0]   # background, iguana
```

The model emits two heads — a dense FIDT heatmap at down_ratio=4 (for
LMDS keypoint extraction) and a coarse classification map at
down_ratio=32 (for hard sample regularization). The focal loss handles
the dense head; cross-entropy with strong positive weighting handles
the coarse one. Config at `configs/demo/losses/herdnet_fmo03.yaml`.

---

## 3. Data

| dataset | source |
|---|---|
| Master Hasty | `/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json` (post phase-15 iter-5 at the time of al_v8 training: 75,821 iguana_point + 815 not_iguana points) |
| Train images list | `al_v8_train_image_names.txt` — 5,142 distinct image names |
| Val images list | `al_v8_val_image_names.txt` — 1,125 net-new non-Fern images across 14 datasets / 8 islands (never reviewed at the time) |
| Tile size | 512×512 (regular-grid crop of full-resolution drone images, overlap=0) |
| Class filter | `["iguana_point"]` (NOT including not_iguana_but_similar_look — that's al_v9's change) |
| Label mapping | `{"iguana_point": 1}` |

Data prep script: `active-learning/scripts/training_data_preparation/021_hasty_to_tile_point_detection.py`
with `dataset_configs_al_v8_2026_06_03.py`.

Output canonical layout at
`/home/christian/data/training_data/2026_06_03_al_v8_canonical/al_v8_2026_06_03/`:

```
train/
  Default/                          # 5,652 cropped 512×512 tiles
  herdnet_format.csv                # columns: images, x, y, species, labels
val/
  Default/                          # 610 → 1,125 tiles (al_v7 → al_v8)
  herdnet_format.csv
```

---

## 4. Augmentation pipeline

Albumentations 2.x. Defined in `configs/demo/al_v8_balanced.yaml` →
`datasets.train.albu_transforms`. `augmentation_multiplier=10` (each
training image yields 10 augmented samples per epoch).

| stage | transform | params |
|---|---|---|
| 1 | **ObjectAwareRandomCrop** | 512×512, min_edge_distance=0, **empty_probability=0.20**, edge_probability=0.20, edge_zone=120, p=1.0 |
| 2 | HorizontalFlip / VerticalFlip / RandomRotate90 | p=0.5 each |
| 3 | ShiftScaleRotate | shift=0.0625, scale=0.15, rotate=45°, p=0.5 |
| 4 | Perspective | scale=0.05, p=0.2 |
| 5 | OneOf (downscale/blur/sharpen) | p=0.3 |
| 6 | RandomBrightnessContrast | ±0.2, p=0.3 |
| 7 | RandomGamma | [80,120], p=0.2 |
| 8 | SomeOf (PlasmaShadow / RandomShadow) | n=1, p=0.2 |
| 9 | HueSaturationValue | hue±10, sat±20, val±15, p=0.3 |
| 10 | RGBShift | ±10 per ch, p=0.2 |
| 11 | CLAHE | clip=4.0, p=0.2 |
| 12 | SomeOf (GaussNoise / ISONoise / ImageCompression) | n=1, p=0.2 |
| 13 | CoarseDropout | 3–8 holes, 8–25 px, p=0.15 |
| 14 | Normalize | p=1.0 (mean/std of ImageNet) |

`ObjectAwareRandomCrop` is the bespoke crop that anchors on iguana
keypoints with a configurable empty_probability. Implementation in
`animaloc/utils/augmentations.py`.

**End transforms** (after albumentations, before loss):

```yaml
MultiTransformsWrapper:
  FIDT:                 # dense heatmap target for focal head
    num_classes: 2
    down_ratio: 4
    radius: 2
  PointsToMask:         # coarse binary mask for CE head
    radius: 2
    num_classes: 2
    squeeze: true
    down_ratio: 32
```

### Validation augmentation

Val pipeline is **only `Normalize`** + `DownSample(down_ratio=4)` end
transform. No random augmentation — the val tiles are eval'd as-is.

---

## 5. Optimizer + LR schedule

```yaml
optimizer: adamW
lr: 8.0e-05              # global learning rate
backbone_lr: 1.0e-05     # lower LR for pretrained backbone
head_lr: 1.0e-03         # higher LR for fresh detection head
weight_decay: 0.00016

warmup_iters: 500        # linear warmup steps

auto_lr:
  mode: max              # higher F2 = better
  patience: 8            # epochs without improvement before LR cut
  threshold: 0.0001
  threshold_mode: rel
  cooldown: 5
  min_lr: 1.0e-08
  verbose: true
```

The 3-group LR (backbone vs head vs global) is wired up inside
`animaloc/train/trainers.py` so the head trains aggressively while the
backbone is finetuned softly.

`auto_lr` uses PyTorch `ReduceLROnPlateau` keyed on the chosen
selection metric (here `f2_score`).

---

## 6. Validation regime

```yaml
evaluator:
  name: HerdNetEvaluator
  threshold: 100               # Hungarian match radius (pixels)
  select_mode: max
  validate_on: f2_score        # MODEL SELECTION METRIC

  kwargs:
    print_freq: 100
    lmds_kwargs:
      kernel_size: [9, 9]
      adapt_ts: 0.20           # relative threshold (× est_map.max())
      scale_factor: 1
      up: true
      score_threshold: 0.30    # ABSOLUTE floor on detection score

stitcher:
  name: HerdNetStitcher
  kwargs:
    overlap: 120
    down_ratio: 4
    up: false
    reduction: mean
```

**LMDS = Local Maxima Detection and Suppression.** It's the
post-processing that converts the dense heatmap into discrete keypoint
predictions:

1. Threshold the heatmap at `adapt_ts × map.max()` (relative).
2. Also drop anything below `score_threshold` (absolute floor — new
   in iter-4 / al_v5; preserved through al_v8).
3. Find local maxima with a `kernel_size=[9, 9]` max-pool.

Both thresholds matter and they compose. The default
(`adapt_ts=0.20, score_threshold=0.30`) is the "recall-tuned but
precision-protected" pair we converged on after the al_v6 threshold
sweep.

`HerdNetStitcher` is used for inference on full-resolution images that
exceed 512 px — it tiles, runs inference per tile, and merges the
heatmaps with `overlap=120` and `reduction=mean`. For training-time
validation on 512-px crops it's a no-op.

---

## 7. Trainer + training settings

```yaml
trainer: Trainer            # animaloc/train/trainers.py
epochs: 10
valid_freq: 1               # eval every epoch
print_freq: 20              # print every 20 training steps
batch_size: 4               # tight for 4090; docker on 80GB uses 32
num_workers: 8

early_stopping: false
early_stopping_patience: 20
early_stopping_min_delta: 0.0
early_stopping_restore_best_weights: true

vizual_fn: visualize_sample
visualiser:
  name: HeatMapVisualizer
  output_dir: ./visualizations
  down_ratio: 4
```

The trainer auto-saves `best_model.pth` every time `validate_on`
improves, plus `latest_model.pth` at the end of each epoch.

Early stopping was OFF for al_v8 (we wanted the full 10-epoch
trajectory). For al_v9 the same setting is kept — 30 epochs to
completion.

---

## 8. Warm-start anchor

```yaml
model:
  name: CamouflageHerdNetConvNeXtV2
  load_from: /home/christian/hnee/HerdNet/best_models/phase8/b4_seed42/best_model.pth
  resume_from: null
  kwargs:
    pretrained: true              # ImageNet pretrained backbone weights
    down_ratio: 4
    backbone_size: tiny
  freeze: null
```

`pretrained: true` pulls ImageNet weights from timm for the backbone.
**Then** `load_from` overlays the phase-8 b4_seed42 checkpoint — that
checkpoint is the consensus warm-start anchor used across ALL the
phase-15 active-learning runs (al_v3, al_v4, al_v5, al_v6, al_v7,
al_v8 — and al_v9 A/B/C/D too). It's the most-validated starting
point.

If `load_from` is null, the head is randomly initialized and the model
needs ~3× more epochs to converge. We did this once intentionally
(`herdnet-al-v7-base` with ConvNeXt-V2 base backbone, where no warm-start
of compatible shape existed) — see
`docker/Dockerfile.al_v7_base`.

---

## 9. Launch command

Hydra invocation. The config is self-contained (no `defaults:` list
composition), so just point at it directly:

```bash
cd /home/christian/hnee/HerdNet
python tools/train.py \
    --config-path /home/christian/hnee/HerdNet/configs/demo \
    --config-name al_v8_balanced \
    > 20260603_al_v8_b4_train.log 2>&1
```

Hydra output dir (set in the config):
`./output/al_v8_b4_seed42/${now:%Y-%m-%d}/${now:%H-%M-%S}/`.
Concretely:
`output/al_v8_b4_seed42/2026-06-03/23-19-13/` for this run.

W&B tracking:
- project: `hn_active_learning`
- entity: `karisu`
- run: `al_v8_b4_seed42_balanced`
- tags: `[active_learning, v8, b4, balanced, f2_score, emp20, post_iter5, multi_island_val]`

---

## 10. Training trajectory (actual)

10 epochs, ~1 h 47 min/epoch on the 4090 (≈ 17 h 52 min total wall
time). `best_model.pth` overwrite happened at epochs 1, 2, 3, 6, 7:

| Epoch | recall | precision | F1 | F2 (select) | F5 | TP / FN / FP | mae |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.840 | **0.884** | 0.861 | 0.848 | 0.842 | 2418 / 461 / 318 | 0.55 |
| 2 | 0.865 | 0.806 | 0.834 | 0.853 | 0.863 | 2491 / 388 / 601 | 0.71 |
| 3 | 0.869 | 0.810 | 0.838 | 0.857 | 0.867 | 2502 / 377 / 588 | 0.68 |
| 4 | — | — | — | ≤0.857 | — | — | — |
| 5 | — | — | — | ≤0.857 | — | — | — |
| 6 | 0.867 | 0.834 | 0.850 | 0.860 | 0.865 | 2495 / 384 / 497 | 0.62 |
| **7** | **0.873** | 0.814 | 0.842 | **0.860** | 0.870 | 2513 / 366 / 576 | 0.66 |
| 8 | — | — | — | ≤0.860 | — | — | — |
| 9 | — | — | — | ≤0.860 | — | — | — |
| 10 | 0.853 | 0.848 | 0.851 | 0.852 | 0.853 | 2457 / 422 / 441 | 0.60 |

Pattern: E1 starts at high precision (0.88, mostly because the
warm-start was already well-calibrated). E2 jumps recall but tanks
precision (model becomes too aggressive). E3 holds. E4-5 plateau.
E6-7 rebound — precision climbs back into 0.81-0.83 territory while
keeping the recall gains. E8-10 oscillate around the same F2 level.

**E7 is the production snapshot** — preserved at
`output/al_v8_b4_seed42/snapshots/al_v8_e7_best.pth` to protect it
from later best_model.pth overwrites.

---

## 11. How to verify / sanity-check a reproduction

After training a fresh model with the same recipe:

1. Tile counts should match: train has 5,652 tiles / 25,212 keypoints,
   val has 610 (al_v7) or 1,125 (al_v8) tiles / 2,879 keypoints.
2. E1 recall+precision should land at roughly (0.83-0.85, 0.86-0.88).
   If E1 starts below 0.80 in recall, either the data prep or the
   warm-start path is wrong.
3. F2 should hit ≥ 0.85 by E3, ≥ 0.86 by E6-E7.
4. Final E10 should remain around F2 = 0.85.

If any of these is off by > 0.02, suspect:
- master Hasty was promoted out from under you between split and prep
- a different warm-start checkpoint
- batch_size was halved without an LR adjustment
- ObjectAwareRandomCrop's empty_probability was changed

---

## 12. Pointers

```text
config             configs/demo/al_v8_balanced.yaml
model class        animaloc/models/herdnet_timm_convnext_camouflaged_v2.py
loss config        configs/demo/losses/herdnet_fmo03.yaml
augmentation       animaloc/utils/augmentations.py  (ObjectAwareRandomCrop)
trainer            animaloc/train/trainers.py
evaluator          animaloc/eval/evaluators.py     (HerdNetEvaluator)
stitcher           animaloc/eval/stitchers.py      (HerdNetStitcher)
LMDS               animaloc/eval/lmds.py

warm-start         best_models/phase8/b4_seed42/best_model.pth
data prep          /home/christian/hnee/active-learning/scripts/training_data_preparation/
                     021_hasty_to_tile_point_detection.py +
                     dataset_configs_al_v8_2026_06_03.py

production weight  output/al_v8_b4_seed42/snapshots/al_v8_e7_best.pth
training log       20260603_al_v8_b4_train.log (1.9 MB)

This doc           docs/convnext_reference_model.md
```

End of recipe.
