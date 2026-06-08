# Model Improvement Plan for Iguana Point Detection

**Baseline:** CamouflageHerdNetConvNeXt-Tiny, F1=0.9486, 33M params

## Model Comparison Matrix

The pipeline supports a smooth progression from the classic DLA to fully-enhanced architectures.
All variants can be smoke-tested via `bash run_micro_all.sh` (2 images, 1 epoch, WandB).

| # | Config | Model | Backbone | Gabor | Edge | BiFPN | Deform | P2P | Notes |
|---|--------|-------|----------|-------|------|-------|--------|-----|-------|
| 1 | `fmo03_micro_dla34_classic` | HerdNet | DLA-34 (custom) | - | - | - | - | - | Delplanque original |
| 2 | `fmo03_micro_dla34` | HerdNetTimmDLA | DLA-34 (timm) | - | - | - | - | - | Modernized DLA |
| 3 | `fmo03_micro_baseline` | HerdNetConvNeXt | ConvNeXt-Tiny | - | - | - | - | - | Separate plain arch |
| 4 | `fmo03_micro_convnext_plain` | CamouflageHerdNetConvNeXt | ConvNeXt-Tiny | OFF | OFF | - | - | - | Same arch, features disabled |
| 5 | `fmo03_micro_v1` | CamouflageHerdNetConvNeXt | ConvNeXt-Tiny | ON | ON | - | - | - | V1: Gabor + Edge + MultiRes |
| 6 | `fmo03_micro_convnextv2` | CamouflageHerdNetConvNeXt | ConvNeXt-V2 Tiny | ON | ON | - | - | - | FCMAE pretrained backbone |
| 7 | `fmo03_micro_efficientvit` | CamouflageHerdNetConvNeXt | EfficientViT-B1 | ON | ON | - | - | - | Lightweight ViT backbone |
| 8 | `fmo03_micro_v2` | CamouflageHerdNetConvNeXtV2 | ConvNeXt-Tiny | ON | ON | ON | ON | - | V2: + BiFPN + DeformConv |
| 9 | `fmo03_micro_v3` | CamouflageHerdNetConvNeXtV3 | ConvNeXt-V2 Tiny | ON | ON | ON | ON | ON | V3: All tiers combined |

**Backbone swap** is controlled by the `backbone` parameter in `CamouflageHerdNetConvNeXt`:
- `convnext` (default) — ConvNeXt-Tiny, ImageNet-22k pretrained
- `convnextv2` — ConvNeXt-V2 Tiny, FCMAE + ImageNet-22k pretrained
- `efficientvit` — EfficientViT-B1, lightweight CNN-ViT hybrid

**Camouflage features** can be toggled independently:
- `use_gabor: False` — disable Gabor texture filters
- `use_edge_enhancement: False` — disable edge enhancement in detection head
- `use_multi_res: False` — disable multi-resolution processing

This enables fair ablation: compare #4 (plain) vs #5 (camouflage ON) with identical architecture.

## Tier 1: Quick Wins (hours, no/low risk)

- [x] **1A. Fuse Gabor features into backbone** (~1h)
  - Gabor module runs but output is discarded (line 705: "Store for potential use")
  - Add 1x1 projection (32 -> 96 channels) and add to `features[0]` before FPN
  - File: `animaloc/models/herdnet_timm_convnext_camouflaged.py`
  - Why: Gabor filters discriminate iguana texture from rock. Already paying compute cost.

- [x] **1B. Switch to DensityAwareFocalLoss** (~30min, config-only)
  - Create `configs/demo/losses/density_aware_fmo03.yaml` for 2 classes
  - Based on existing `configs/submission/losses/density_aware_herdnet_iguana.yaml`
  - Why: Upweights isolated iguanas that are hardest to detect

- [x] **1C. Test-Time Augmentation (TTA)** (~2h, inference-only)
  - Run each patch 4x (orig + H-flip + V-flip + 180deg), average heatmaps
  - File: `animaloc/eval/stitchers.py` HerdNetStitcher
  - Add `tta: bool = False` parameter
  - Why: No canonical orientation in drone imagery. No retraining needed.

## Tier 2: Medium Effort (1-3 days)

- [x] **2A. Auxiliary deep supervision on FPN P3/P4** (~1 day)
  - Add 1x1 conv heads on `fpn_features[1]` (64x64) and `fpn_features[2]` (32x32)
  - Supervise with downsampled FIDT GT at 0.3x and 0.1x loss weight
  - Need to modify `LossWrapper` for 4 outputs instead of 2
  - Why: Forces intermediate FPN layers to produce useful gradients

- [x] **2B. BiFPN (bidirectional feature pyramid)** (~1 day)
  - Add bottom-up pass after existing top-down: P2 -> P3 -> P4 -> P5
  - File: `animaloc/models/herdnet_timm_convnext_camouflaged_v2.py` CamouflageEnhancedFPN
  - ~2M extra params
  - Why: Fine-grained texture at P2 flows up to inform coarser context

- [x] **2C. Deformable convolutions in detection head** (~2 days)
  - Replace dilated Conv2d (d=1,2,4) with `torchvision.ops.DeformConv2d`
  - Each branch needs an offset prediction conv
  - File: `animaloc/models/herdnet_timm_convnext_camouflaged_v2.py` CamouflageDetectionHead
  - Why: Iguanas have irregular shapes. Rigid dilation grids miss small objects. ~1M extra params.

## Tier 3: Larger Changes (1+ week)

- [x] **3A. Auxiliary P2P loss during training** (~1 week)
  - Add HungarianLoss alongside heatmap focal loss
  - P2P head predicts point coordinates from FPN features
  - Only used during training; inference still uses heatmap + LMDS
  - Files: `animaloc/train/losses/p2p.py` (existing), model (new head)
  - Why: Complementary gradient signal for ambiguous peaks

- [x] **3B. ConvNeXt-V2 or EfficientViT backbone** (~1 week)
  - Replace timm ConvNeXt-Tiny with ConvNeXt-V2 (FCMAE pre-training) or EfficientViT-M4
  - Both available in timm, same interface
  - Why: Better pre-trained features

## Execution Order

**Phase 1** (no retraining):
1. 1C (TTA) -- test on existing best_model.pth

**Phase 2** (single retraining run):
2. 1A (Gabor fusion) + 1B (DensityAwareFocalLoss) combined

**Phase 3** (architecture):
3. 2A (auxiliary supervision) + 2B (BiFPN) combined
4. 2C (deformable conv) separate run

**Phase 4** (if needed):
5. 3A (P2P loss) or 3B (backbone swap)

## Verification

For each change:
1. Train with `fmo03_new_objcrop_augplus` dataset, ConvNeXt-Tiny, 30 epochs
2. Run `tools/inference_hparam_search.py` to find optimal adapt_ts
3. Compare F1, recall, precision, MAE, ME against baseline
4. If F1 > 0.95, save to `best_models/`
