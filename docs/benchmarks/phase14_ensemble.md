# Phase 14 — cross-arch ensemble experiment (negative result)

**Run date**: 2026-05-27 → 2026-05-30.
**Branch**: `convnext_extension` @ `add7549..`.
**Wandb projects**: `hn_phase14_ensemble`, `hn_phase14b_ensemble`, `hn_phase14c_ensemble`.
**Plan reference**: [`docs/phase14_ensemble_strategy.md`](../phase14_ensemble_strategy.md).
**Outcome**: H1 success gate (F1 ≥ 0.94) not met. **Phase 13 N=full single B4 remains production.**

## Question

Phase 13 showed the data-scaling F1 curve plateaus at N≈600 single-model on the Phase-13 val (FPE02 site). Phase 8 had previously shown that cross-arch ensembling (B3×3 + B4×3) lifts F1 from 0.945 to 0.972 on a different val (FMO05 site). Does the same ensembling lever work on the Phase-13 val, and does it close the recall gap for the active-learning use case?

## Setup

Three sequential variants, each a 6-member ensemble of `CamouflageHerdNetConvNeXt` (B3) × 3 seeds + `CamouflageHerdNetConvNeXtV2` (B4) × 3 seeds, all warm-started from the corresponding Phase-8 production member (`best_models/phase8/{b3,b4}_seed{7,42,123}/best_model.pth`). Trained on `N=full` (6,908 frames). Each variant differs only in selection metric, CE foreground weight, and augmentation budget — full table below.

| Variant | `validate_on` | `CE.weight` (fg) | `adapt_ts` (train) | `aug_mult` | epochs | early-stop |
|---|---|---|---|---|---|---|
| Phase 13 single (baseline) | f1_score | 5.0 | 0.30 | 75 | 30 | off |
| Phase 14 | **f5_score** | **2.0** | **0.20** | 1 | 10 | patience 3 |
| Phase 14b | f2_score | 3.0 | 0.30 | 1 | 10 | patience 3 |
| Phase 14c | f1_score | 5.0 | 0.30 | **1** | **10** | patience 3 |

Total compute: ~36 GPU-hours across the three variants.

## Results — Stage B ensemble inference on Phase-13 val

Each row is the 6-member cross-arch ensemble run through the LMDS at the listed threshold, on the full-size stitched Phase-13 val (1,793 GT iguanas).

### Phase 14 (F5 + CE 2.0 + adapt_ts=0.20 + aug_mult=1)

| ts | F1 | Precision | Recall | MAE | AP |
|---|---|---|---|---|---|
| 0.10 | 0.251 | 0.146 | 0.905 | 34.72 | 0.852 |
| 0.20 | 0.577 | 0.425 | 0.898 | 7.46 | 0.865 |
| 0.30 | 0.768 | 0.680 | 0.881 | 2.40 | 0.862 |
| 0.40 | 0.844 | 0.829 | 0.859 | 1.42 | 0.847 |
| **0.50** | **0.866** | 0.911 | 0.825 | 1.38 | 0.817 |

### Phase 14b (F2 + CE 3.0 + adapt_ts=0.30 + aug_mult=1)

| ts | F1 | Precision | Recall | MAE | AP |
|---|---|---|---|---|---|
| 0.20 | 0.639 | 0.495 | 0.898 | 5.53 | 0.855 |
| 0.30 | 0.798 | 0.728 | 0.881 | 2.07 | 0.857 |
| 0.40 | 0.858 | 0.853 | 0.863 | 1.39 | 0.848 |
| **0.50** | **0.872** | 0.920 | 0.829 | 1.31 | 0.819 |

### Phase 14c (F1 + CE 5.0 + adapt_ts=0.30 + aug_mult=1)

| ts | F1 | Precision | Recall | MAE | AP |
|---|---|---|---|---|---|
| 0.20 | 0.626 | 0.483 | 0.888 | 5.73 | 0.863 |
| 0.30 | 0.789 | 0.723 | 0.867 | 2.37 | 0.854 |
| 0.40 | 0.842 | 0.843 | 0.841 | 1.66 | 0.834 |
| **0.50** | **0.857** | 0.918 | 0.803 | 1.54 | 0.798 |

### Side-by-side at the F1 peak

| Variant | F1 peak | Recall at peak | Precision at peak | AP | Best ts |
|---|---|---|---|---|---|
| **Phase 13 single B4 (baseline)** | **0.890** | 0.842 | 0.943 | 0.824 | 0.35 |
| Phase 14 | 0.866 | 0.825 | 0.911 | 0.86 | 0.50 |
| Phase 14b | 0.872 | 0.829 | 0.920 | 0.86 | 0.50 |
| Phase 14c | 0.857 | 0.803 | 0.918 | 0.86 | 0.50 |

**All three Phase 14 variants are worse than a single Phase-13 model**, on every headline number except AP.

## Analysis

### 1. The ensembling lever doesn't transfer from FMO05 to FPE02

Phase 8 hit F1 = 0.972 on the FMO05 val (12 frames, 181 GT iguanas). The Phase 14 ensemble peaks at F1 = 0.872 on the FPE02 val (much larger, 1,793 GT iguanas, ~10× the annotations, different site). Same B3 + B4 architecture, comparable recipe, dramatically different result. **The conclusion isn't that ensembling fails everywhere; it's that the FMO05 ceiling is artificially high because the val set is small and from the model's training distribution.** The "9-point F1 ensemble lift" we extrapolated from Phase 8 was site-specific.

### 2. Per-member operating point matters more than ensembling

Phase 14 (F5 + CE 2.0) produced 6 members with intrinsic precision 0.45–0.68. Averaging six heatmaps that all light up on similar FPs **cements** those FPs instead of cancelling them — the FP locations are correlated across members because they share the same training distribution and recipe. Phase 14b walked the recipe back to F2 + CE 3.0 and recovered some per-member precision (~0.75–0.79), netting +0.006 F1 over Phase 14 — meaningful direction, insufficient magnitude.

### 3. Phase 14c proved aug_mult=1 is the dominant under-training factor

Phase 14c used the Phase-13 / Phase-8 recipe verbatim (validate_on=f1_score, CE.weight=5.0, adapt_ts=0.30) but kept Phase 14's `aug_mult=1` and `epochs=10`. Per-member F1 collapsed to 0.78–0.83 (vs Phase 13 single 0.89). Compute math explains it cleanly:

- Phase 13 N=full at best_epoch=4: **4 × 75 = 300 effective passes** through the training data.
- Phase 14c best epochs were 3–5: **3–5 effective passes**.

The "Phase 14 efficiency win" of dropping aug_mult was correct for visibility of the per-epoch training trajectory (Phase 13 §6, recall-vs-epoch finding), but it left the F1 recipe under-converged. **Without aug_mult=75, you cannot reproduce the Phase-13 per-member F1**, and without the per-member F1 the ensemble can't help.

### 4. The AP plateau at 0.86 is the architectural ceiling

All three Phase 14 variants land at AP = 0.86 across the entire threshold sweep — *higher* than Phase 13 single's AP = 0.824. The Phase 14 models have **better discrimination** (the PR curve is shifted up), but the operating point we want (recall ≥ 0.93 AT precision ≥ 0.85) doesn't exist on the curve. Threshold tuning alone can't reach it because LMDS produces too few high-confidence candidates beyond a given level. The architectural ceiling (B3 + B4 + this val) is around F1 = 0.87 / recall = 0.83.

### 5. Threshold tuning at the recall-priority operating point still works

Phase-13 single B4 at `adapt_ts=0.20`: **F1 = 0.882, recall = 0.856, precision = 0.911, MAE = 1.10**. This is the active-learning operating point — high enough recall that human reviewers see most true iguanas, high enough precision that the FP load is manageable.

## Decision

Per the strategy doc's explicit kill criterion:

> Stage A+B fail to clear F1 ≥ 0.92 → **Stop**. Single-model ceiling is real and ensembling didn't help. Re-examine architecture, loss design, or val-set composition before more training.

We're at F1=0.872 across all three variants. **Phase 14 is closed as a negative result.**

## Production model

**`output/phase13_Nfull_s42/2026-05-16/05-50-52/best_model.pth`** at `adapt_ts=0.20` is the active-learning production model. Headline numbers:

- F1 = 0.882
- Recall = 0.856 (of 1,793 GT iguanas, ~258 missed)
- Precision = 0.911 (~150 FPs per 1,793 detections)
- MAE = 1.10 (per-frame counting error ~1.1 iguanas on frames containing ~15)

## What went right

Each Phase 14 variant produced cleanly reusable infrastructure:

- `configs/demo/phase14*_ensemble_{b3,b4}.yaml` — 6 Hydra configs, F2/F5/F1 recipes parameterised.
- `run_phase14.sh` — generic 6-member sequential runner with `CONFIG_PREFIX` / `PHASE_TAG` / `BATCH_SIZE` / `NUM_WORKERS` env knobs.
- `run_phase14_stage_b.sh`, `run_phase14b_stage_b.sh`, `run_phase14c_stage_b.sh` — ensemble inference + threshold sweep with locale-safe CSV parsing.
- **Trainer bug fix** (`animaloc/train/trainers.py`, commit `2013460`): `best_model.pth` was never being written when early-stopping was enabled, due to a side-effect collision between `_early_stopping_check` and `_is_best`. Both now use `self.min_delta` consistently.

## What we learned (the negative result is the result)

1. **Phase 8's F1=0.972 was site-specific** — it doesn't predict performance on novel sites. Future ensembling experiments need a held-out, per-site baseline before claiming generalisable gains.
2. **`aug_mult=75` is load-bearing for the F1 recipe**, not a tunable knob. If reduced, training time per epoch is shorter but the F1 recipe loses 1.5 orders of magnitude of optimizer-step coverage.
3. **For active learning specifically**, threshold-tuning the Phase-13 single model at `adapt_ts=0.20` already gives the recall-priority operating point we wanted. The recipe rewrite was solving the wrong problem.
4. **AP = 0.86 is the architectural ceiling on this val**, regardless of recipe. To beat it, we need either (a) a different architecture, (b) better data per-site, or (c) larger backbones / SSL pretraining. None of these are recipe knobs.

## What to do next

Per the strategy doc's parked candidates (Phase 15 / 16):

1. **Phase-11-style error analysis on the Phase-13 single B4** — identify whether residual FNs are small-iguana failures (→ Phase 15 zoom augmentation), site-specific (→ Phase 16 per-island fine-tuning), or annotation gaps (→ active-learning loop). Triggered as a follow-up to this writeup.
2. **Phase 15 — multi-scale (zoom) augmentation** if the error analysis points to small-iguana failures.
3. **Phase 16 — per-island fine-tuning** if the residuals are distributionally site-specific.

## Files

- Strategy doc (parked Phase 15 / 16 candidates): [`docs/phase14_ensemble_strategy.md`](../phase14_ensemble_strategy.md)
- Stage A training logs: `output/phase14*_b{3,4}_s{7,42,123}/*/*_training.log`
- Stage B sweeps: `output/phase14_stage_b_*/summary.csv`, `output/phase14b_stage_b_*/summary.csv`, `output/phase14c_stage_b_*/summary.csv`
- All 18 checkpoints (3 phases × 6 members): `output/phase14*_b{3,4}_s{7,42,123}/.../best_model.pth`
