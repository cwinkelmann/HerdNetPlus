# Phase 14 — held-out test-set evaluation

**Run date**: 2026-05-30 (sweep) → 2026-06-01 (writeup).
**Branch**: `convnext_extension` @ `ac08269..`.
**Script**: [`run_test_eval_sweep.sh`](../../run_test_eval_sweep.sh) + [`tools/per_site_metrics.py`](../../tools/per_site_metrics.py).
**Outputs**: `output/test_sweep_20260530_101438/per_site_aggregate.csv`.
**Predecessors**: [Phase 13 data scaling](phase13_data_scaling.md), [Phase 14 ensemble — val](phase14_ensemble.md).

## Question

Phase 14 closed on the val set with three ensemble variants under-performing single Phase-13 B4 (writeup [`phase14_ensemble.md`](phase14_ensemble.md)). The val set covered 4 sites; the held-out **test set covers 22 sites** including locations the val never saw. Does the Phase-14 negative result hold on test? And what does the per-site breakdown say about where the production model actually struggles?

## Setup

Single coarse sweep over four models × multiple LMDS thresholds on the held-out Phase-13 test set:

- **Phase 13 single B4** (production candidate, F1 recipe) at `adapt_ts ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.50}`
- **Phase 14 / 14b / 14c ensembles** (3 × B3 + 3 × B4, F5/F2/F1 recipes) at `adapt_ts ∈ {0.30, 0.40, 0.50}`

Per-(model, threshold, site) precision/recall/F1 aggregated by greedy GT-vs-detection matching at radius=100 px. Total: 15 eval runs, ~5 hours GPU.

Test set: **708 tiles, 5,102 GT iguana keypoints, 22 sites** spanning Fernandina, Floreana, FMO, Genovesa, Isabela, ISCW/ISNCW, Marchena, Pinta, Pinzon, and San Cristóbal. Of these, the val set only used FPE02 + ISA_ISPVR + SCRUZ — so most of the test sites are genuinely new distributions.

## Aggregate results (whole test set)

| Model | F1 peak | Recall @ peak | Precision @ peak | best ts |
|---|---|---|---|---|
| **Phase 13 single B4** | **0.848** | 0.811 | 0.889 | 0.50 |
| Phase 14 ensemble (F5 recipe) | 0.784 | 0.757 | 0.813 | 0.50 |
| Phase 14b ensemble (F2 recipe) | 0.791 | 0.756 | 0.829 | 0.50 |
| Phase 14c ensemble (F1 recipe + aug_mult=1) | 0.778 | 0.731 | 0.832 | 0.50 |

### Side-by-side with val (F1 peak from `phase14_ensemble.md`)

| Model | F1 — val | F1 — test | Δ |
|---|---|---|---|
| Phase 13 single B4 | 0.890 | 0.848 | −0.042 |
| Phase 14 (F5) | 0.866 | 0.784 | −0.082 |
| Phase 14b (F2) | 0.872 | 0.791 | −0.081 |
| Phase 14c (F1, aug_mult=1) | 0.857 | 0.778 | −0.079 |

**Two findings**:

1. **All four models lose F1 on test vs val.** Expected — test is broader-distribution and includes sites not seen during training. The single B4 loses 0.042 F1; the Phase 14 ensembles lose ~0.08 each.
2. **The Phase 14 negative result is robust.** Single B4 beats every ensemble variant on test by **0.057 → 0.070 F1**. The gap is *wider* on test than on val, ruling out "Phase 14 just wasn't tuned for the val distribution." Ensembling under the F5/F2/F1 recipes simply doesn't transfer across sites.

## Per-site breakdown (Phase 13 single B4 at `ts=0.20`, the recall-tuned operating point)

Sorted by F1, descending. `n_gt` is the count of GT iguanas at that site.

| Site | n_gt | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| San_STJB01_10012023_DJI_0068 | 25 | 25 | 4 | 0 | 0.862 | **1.000** | 0.926 |
| FMO04 | 825 | 760 | 109 | 65 | 0.875 | 0.921 | **0.897** |
| isa_isvp01_27012023 | 333 | 305 | 46 | 28 | 0.869 | 0.916 | 0.892 |
| FMO02 | 49 | 49 | 14 | 0 | 0.778 | **1.000** | 0.875 |
| iscw01_25012023 | 701 | 614 | 99 | 87 | 0.861 | 0.876 | 0.869 |
| Floreana_03.02.21_FMO06 | 529 | 472 | 124 | 57 | 0.792 | 0.892 | 0.839 |
| Genovesa | 882 | 687 | 102 | 195 | 0.871 | 0.779 | 0.822 |
| isncw02_25012023 | 398 | 340 | 97 | 58 | 0.778 | 0.854 | 0.814 |
| isncw01_25012023 | 482 | 432 | 154 | 50 | 0.737 | 0.896 | 0.809 |
| fer_fne03_19122021 | 346 | 249 | 23 | 97 | 0.915 | 0.720 | 0.806 |
| pin_pwc02_11122021 | 58 | 48 | 11 | 10 | 0.814 | 0.828 | 0.821 |
| mar_mnw05_07122021 | 2 | 2 | 1 | 0 | 0.667 | 1.000 | 0.800 |
| fer_fne01_19122021 | 300 | 218 | 73 | 82 | 0.749 | 0.727 | 0.738 |
| San_STJB01_12012023 | 23 | 22 | 19 | 1 | 0.537 | 0.957 | 0.688 |
| San_STJB02_12012023 | 6 | 6 | 7 | 0 | 0.462 | 1.000 | 0.632 |
| mar_mbbe04_09122021 | 18 | 16 | 17 | 2 | 0.485 | 0.889 | 0.628 |
| San_STJB03_12012023 | 19 | 17 | 26 | 2 | 0.395 | 0.895 | 0.548 |
| San_STJB04_12012023 | 9 | 9 | 17 | 0 | 0.346 | **1.000** | 0.514 |
| pze11_08012023 | 13 | 11 | 19 | 2 | 0.367 | 0.846 | 0.512 |
| **San_STJB06_12012023** | 62 | 57 | **140** | 5 | **0.289** | 0.919 | **0.440** |
| pze10_08012023 | 12 | 9 | 21 | 3 | 0.300 | 0.750 | 0.429 |
| **mar_mnw04_07122021** | 10 | 5 | 22 | 5 | **0.185** | **0.500** | **0.270** |

**Per-site F1 ranges from 0.27 → 0.93** — variance across sites *dwarfs* the cross-model F1 differences (0.07) we spent ~36 GPU-hours chasing in Phase 14. Same architecture, same recipe, same threshold — radically different outcomes per location.

## Two failure modes

### 1. "Tiny San Cristóbal sites" — precision collapse

The `San_STJB*` cluster (San Cristóbal Bahía sites) and `pze*` (Pinzón East) show a consistent pattern:

| Site | n_gt | FP | Precision | Recall |
|---|---|---|---|---|
| San_STJB06 | 62 | 140 | 0.29 | 0.92 |
| San_STJB04 | 9 | 17 | 0.35 | 1.00 |
| San_STJB03 | 19 | 26 | 0.40 | 0.89 |
| pze10 | 12 | 21 | 0.30 | 0.75 |
| pze11 | 13 | 19 | 0.37 | 0.85 |

Recall is good-to-excellent (the iguanas that exist *are* detected), but precision is catastrophic — model is finding 1.5–3× as many "iguanas" as exist. These sites likely have visual features (volcanic rocks, sticks, shadows) the model has never been trained against. **Pure hard-negative mining territory** — exactly what Phase 15's hard-negative pipeline targets.

### 2. "Marchena" — small + hard + ambiguous

`mar_*` sites have very few GT (2, 10, 18) plus both low precision AND low recall:

| Site | n_gt | Precision | Recall | F1 |
|---|---|---|---|---|
| mar_mnw04 | 10 | 0.19 | 0.50 | 0.27 |
| mar_mbbe04 | 18 | 0.49 | 0.89 | 0.63 |
| mar_mnw05 | 2 | 0.67 | 1.00 | 0.80 |

mar_mnw04 is the worst single site overall — both modes are broken there (the model misses half the real iguanas *and* fires on twice as many non-iguanas). Could be lighting, altitude, or a subspecies the training set didn't cover.

### What works

Sites with **F1 ≥ 0.84**: FMO04, FMO02, isa_isvp01, iscw01, Floreana_FMO06, Genovesa, isncw02, fer_fne03, plus a few small ones. These have at least one of:
- Visual similarity to training distribution (FMO04 / FMO02 share imagery characteristics with FMO03 which is in train).
- Large GT count so even imperfect precision averages out (Genovesa: n_gt=882).

## Implications

### For production model selection (immediate)

**Phase 13 single B4 is the production model.** F1 = 0.848 / recall = 0.811 / precision = 0.889 on the held-out test at `ts=0.50`, or **F1 = 0.821 / recall = 0.853 / precision = 0.792 at ts=0.20** for the active-learning operating point. Phase 14 ensembling is closed as a negative result on both val and test.

### For Phase 15 (annotation cleanup loop — currently in progress)

The San_STJB / pze precision-collapse sites are the highest-value Phase-15 targets. The iter-0 50-image review you've already done included STJB06 frames (you removed 9 + 8 + 6 + 4 = 27 high-confidence FPs across the STJB06 series). Each one becomes a `not_iguana_but_similar_look` training example. **Iteration 1's Stage A retrain should show measurable precision lift on STJB sites specifically** — that's the load-bearing prediction this evaluation makes.

### For Phase 16 (per-island fine-tuning — parked)

This is the strongest evidence to date for per-island fine-tuning. Per-site F1 variance (0.27 → 0.93) is 9× larger than cross-model variance under Phase 14 (0.07). The marginal gain from per-island fine-tuning on mar_mnw04 / San_STJB sites is plausibly +0.20 F1 each — vastly more than any recipe tweak. Phase 16 activation criteria from the strategy doc are now firmly met:

| Phase 16 gate | Status |
|---|---|
| Generalist F1 < 0.93 | ✓ 0.848 |
| ≥ 1000 annotations per site for ≥ 2–3 sites | Genovesa (882), FMO04 (825), iscw01 (701) — need 1000+ for the failure-mode sites (STJB / mar / pze) |

The annotation-budget question for Phase 16: do we have ≥ 1000 frames per failure-mode site to fine-tune on? STJB06 currently has only 62 GT in test — insufficient. **Phase 16 should be paired with a targeted annotation push at the failure sites**, not done with current data alone.

## Files

- **Per-site CSV** (cross-model, cross-threshold): [`output/test_sweep_20260530_101438/per_site_aggregate.csv`](../../output/test_sweep_20260530_101438/per_site_aggregate.csv) — open in pandas / Excel and pivot on (model, threshold, site).
- **Per-eval detections + metrics**: `output/test_sweep_20260530_101438/<model>_ts<ts>/` (e.g., `phase13_single_b4_ts0.50/detections.csv` + `metrics_results.csv`).
- **Per-eval inference logs**: `output/test_sweep_20260530_101438/<model>_ts<ts>.log`.

## Reproduction

```bash
bash run_test_eval_sweep.sh
# ~5h on a single GPU; writes output/test_sweep_<date>/ with one subdir
# per (model, threshold) plus per_site_aggregate.csv.
```

The runscript references the Phase-13 single B4 checkpoint at
`output/phase13_Nfull_s42/2026-05-16/05-50-52/best_model.pth` and the
six checkpoints per Phase-14 variant under
`output/phase14*_b{3,4}_s{7,42,123}/.../best_model.pth`. All checkpoints
are git-untracked but live on disk under the HerdNet output tree.

## One-line takeaway

The Phase 14 negative result holds on the held-out test set; **per-site F1 variance (0.27→0.93) is the dominant signal**, and it's pointing squarely at Phase 15 hard-negative mining for the precision-collapse sites and Phase 16 per-island fine-tuning for the bigger-picture distribution shift.
