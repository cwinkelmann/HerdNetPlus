# Phase 14 — beyond data scaling: ensemble + F5-weighted training

**Status**: draft strategy, 2026-05-26.
**Branch**: `convnext_extension`.
**Predecessors**: [Phase 13 data-scaling](benchmarks/phase13_data_scaling.md), [Phase 8 cross-arch ensemble](benchmarks/phase8_*.md), [Phase 11 error analysis](benchmarks/phase11_error_analysis_v2.md).
**Prerequisite reading**: Phase 13 §7 (recall peaks at epoch 1) and §8 (threshold tuning can't recover lost recall).

## Why this phase exists

Phase 13 made it conclusive that **bulk data scaling is no longer the lever**:

| Observation | Evidence |
|---|---|
| Single-model F1 saturates at ~0.88 on the Phase-13 val | N=608 → N=full (×11 data) buys only +0.017 F1 |
| Single-model recall ceiling is ~0.89 *regardless of threshold* | Sweep at adapt_ts=0.05 reaches 0.887 recall but at precision=0.655 |
| Training past epoch 1–2 is actively **costing** us recall | N=full: recall 0.889 (ep 1) → 0.808 (ep 30), while precision climbs +12.4 pts |
| The current loss + selection metric is wrong for the downstream use case | `validate_on=f1_score` + `CE.weight=[0.1, 5.0]` jointly produce a confidence-suppressing model |
| Phase 8 already proved ensembling closes most of the gap | B3×3+B4×3 cross-arch ensemble: **F1=0.972, recall=0.961** on Phase-8 val |

The gap between single-model (0.88) and ensemble (0.97) is ~9 F1 points — roughly **100× more leverage** than another doubling of training data. This phase pursues that leverage on the Phase-13 split.

## Goal

Establish the production model and operating point for active-learning iguana detection on the Phase-13 fixed val/test split. Specifically:

1. **Reach F1 ≥ 0.95 / recall ≥ 0.93** on the Phase-13 val set — i.e. close most of the gap to Phase-8's ensemble numbers.
2. **Provide a recall-tuned operating point** suitable for human-in-the-loop annotation review: recall ≥ 0.95 at precision ≥ 0.80.
3. **Catalogue the remaining failure modes** (Phase-11-style error analysis) on the new ensemble, so the next round of annotation effort is targeted, not bulk.

## Hypotheses

Each is testable, with success criteria.

**H1. Cross-arch multi-seed ensemble closes the single-model gap.**
*Test*: 3 seeds × {B3 (ConvNeXt-T), B4 (ConvNeXt-V2-T + BiFPN + DeformConv)} = 6 members, mean-prediction ensemble. *Success*: F1 ≥ 0.94 on Phase-13 val (Phase 8 hit 0.972 on a different split — anything in the 0.94+ band confirms ensembling is the dominant lever).

**H2. F5-weighted training recipe expands the recall envelope of every ensemble member without collapsing precision.**
*Test*: Train each ensemble member with `validate_on=f5_score`, `early_stopping_patience=3`, `CE.weight=[0.1, 2.0]` (foreground 5.0 → 2.0), stop at the F5-best checkpoint. *Success*: per-member recall at default threshold ≥ 0.86 AND per-member precision ≥ 0.85 (vs 0.94 today — explicit trade), AND ensemble recall ≥ 0.95.

*Why F5, not raw recall*: pure recall is degenerate — a model that flags every pixel as iguana scores recall=1.0 at precision≈0. F5 keeps recall as the dominant term (`F5 = (1 + 25) · P · R / (25 · P + R)` — recall weighted 5× over precision) while remaining bounded by *both* metrics, so the optimum is at the operating point that maximises recall *given some precision floor*. F2 is the safer alternative if F5 over-corrects.

**H3. Hard-negative-mined frames buy more than equivalent bulk frames.**
*Test*: From the Phase-13 ensemble's FPs, manually verify 100 hard-negative regions. Add them as labelled background. Retrain one member, compare member precision at fixed recall to a member trained on +100 bulk frames. *Success*: hard-negative member has ≥0.02 higher precision at recall=0.85 than the bulk-frame member. (This experiment is deferred until H1+H2 are confirmed — no point hard-negative mining the wrong model.)

## Experimental protocol

### Stage A — multi-seed ensemble (H1 + H2 jointly)

**6 trainings, each independent, parallelisable across remote GPUs.**

| Variable | Value |
|---|---|
| Architecture | {B3: `CamouflageHerdNetConvNeXt` (tiny), B4: `HerdNetTimmConvNext_Camouflaged_V2` (tiny)} |
| Seeds | {7, 42, 123} per architecture |
| Training pool | `N=full` (6,908 frames) — picking the asymptote because the curve is flat past N=608; extra data is free if we're already paying for the run |
| Warm-start | Phase-8 production member for that arch (`best_models/phase8/{b3,b4}_seed{7,42,123}/best_model.pth`) |
| Loss | `losses=herdnet_fmo03` with `CE.kwargs.weight=[0.1, 2.0]` (default override: 5.0 → 2.0) |
| Epochs | 10 max — Phase 13 §6 shows late epochs only hurt recall |
| `validate_on` | `f5_score` (`evaluator.select_mode=max`) — bounded recall-priority; see H2 |
| Early stopping | `early_stopping=True`, `early_stopping_patience=3`, `early_stopping_min_delta=0.005` |
| `adapt_ts` (training-time eval) | 0.20 — lower than current 0.30 to align training-time validation with the intended inference threshold |
| Augmentation | augplus pipeline, **`augmentation_multiplier=1`** (see note below) |
| Batch / workers | `BATCH_SIZE=8`, `NUM_WORKERS=16` if remote GPU has the headroom |
| Wandb | Project `hn_phase14_ensemble`, tag per arch+seed |
| Wall-clock per run | ~5 h on a single GPU |
| Total wall-clock | ~30 GPU-h sequential, ~5 h if all 6 run in parallel |
| Container | `dockerkartok/herdnet:phase13-nfull-latest` (already validated to reproduce local results to ΔF1=0.0002) |

**Note on `augmentation_multiplier=1`** (originally 75). The multiplier was introduced for *small-N* datasets — at N=19 or N=38, seeing each frame only once per epoch gives such a sparse gradient signal that "real" epochs are meaningless. Multiplying by 75 was a way to inflate epoch length so the LR scheduler and validator had enough optimizer steps to do useful work. With Phase 14 running on N=full (6,908 frames), that crutch is no longer needed:

- One real epoch is already 6,908 frames — plenty of optimizer steps for the scheduler.
- With multiplier=75, each "epoch" was ~518k samples — the per-epoch validation curve became a coarse summary over what was effectively 75 mini-epochs, hiding the actual learning trajectory.
- Phase 13's recall-collapse-after-epoch-1 finding was only visible because the per-step trajectory was still readable; if we want to *see* training dynamics clearly (and have early stopping fire on the right signal), we want one frame = one sample seen.
- Wall-clock benefit: training time scales linearly with multiplier; dropping 75→1 cuts a ~14-day asymptotic run to ~5 hours.

The augplus *pipeline* itself (HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, etc.) still applies — every sample is still augmented randomly. We're just no longer drawing 75 augmented variants of every frame per epoch.

### Stage B — ensemble inference + operating-point sweep

Run `tools/ensemble_infer.py` over all 6 members. Sweep `adapt_ts ∈ {0.10, 0.15, 0.20, 0.25, 0.30}` on the Phase-13 val and produce:

- PR curve (one point per threshold)
- Per-frame precision/recall/MAE (Phase-12-style plots)
- Two recommended operating points: **F1-optimal** and **recall-optimal-at-precision≥0.80**

### Stage C — Phase-11-style error analysis on the ensemble

Same script as Phase 11: greedy 1:1 GT-vs-detection matching within radius=100 px, dump all FN/FP crops sorted by confidence, write a markdown writeup. Categorise the residual error modes so Stage D (hard-negative mining) is targeted.

### Stage D — hard-negative mining (H3, conditional)

Only run if Stage A+B hit ≥ 0.94 F1 / ≥ 0.93 recall. Otherwise the model is too far from production-ready for FP-focused annotation to be the right next move.

Procedure:
1. From Stage B's high-recall operating point, dump all detections with score ∈ [0.20, 0.50] (the "ambiguous" band).
2. Human verifies — typically ~200 detections, ~4 hours of labelling.
3. Verified FPs added as labelled background regions in the training pool.
4. Retrain one ensemble member (B4, seed=42) with the augmented pool, compare member-level precision at fixed recall to the same member trained without the additions.

## Compute budget

| Stage | GPU-h | Wall-clock | Labour |
|---|---|---|---|
| A — train 6 members (parallel) | 30 | 5 h | none (containerised) |
| A — train 6 members (sequential) | 30 | 30 h | none |
| B — ensemble inference + threshold sweep | 2 | 2 h | none |
| C — error analysis | 0.5 | 30 min | ~1 h reviewing crops |
| D — hard-negative mining (conditional) | 5 | 6 h | ~4 h human labelling |
| **Total (parallel A, with D)** | **~38** | **~14 h GPU + 5 h human** | |

## Success / kill criteria

| Outcome | Action |
|---|---|
| Stage A+B reach F1 ≥ 0.95 AND recall ≥ 0.93 | Declare production model, proceed to Stage C+D, archive as Phase 14 |
| Stage A+B reach F1 ∈ [0.92, 0.95] but recall < 0.93 | F5 isn't pulling far enough into recall — try `validate_on=recall` with a precision-floor early-stop guard (`stop if precision < 0.75`) and rerun A |
| Stage A+B reach F1 ∈ [0.92, 0.95] but precision < 0.80 | F5 pulled too far — fall back to `validate_on=f2_score` (recall weighted 2× over precision) and rerun A |
| Stage A+B fail to clear F1 ≥ 0.92 | **Stop**. Single-model ceiling is real and ensembling didn't help. Re-examine architecture, loss design, or val-set composition before more training |
| Stage D fails H3 (hard negatives no better than bulk) | Stop annotation effort entirely. Lever is architectural, not data |

## Risks and mitigations

1. **NaN losses at large N.** Phase 13 saw `ce_loss=nan` mid-training on both 2432 and local N=full runs. **Mitigated** by `d24334d` (skip-batch on non-finite loss, abort only after 50 consecutive). Verified on the existing skip-step code path.
2. **EMA + early-stopping interaction.** Trainer's `restore_best_weights=True` is EMA-aware (CLAUDE.md), but resumed runs reset the patience counter. Don't resume Stage A runs mid-flight; let them run cleanly.
3. **CE foreground weight change affects gradient scale.** Going 5.0 → 2.0 reduces foreground loss magnitude. The auto_lr scheduler may not adapt fast enough at lr=8e-5. *Mitigation*: monitor first epoch closely; if loss plateaus immediately, bump head_lr 0.001 → 0.0015.
4. **Wandb storage.** 6 model artifacts at ~400 MB each = 2.4 GB on the wandb account. Acceptable; set retention to "keep latest only" per arch if needed.
5. **Phase-13 val set is small (12 frames, 181 GT iguanas).** Statistical resolution is poor at the high-precision tail — a 0.01 F1 difference between ensemble configs may not be significant. *Mitigation*: report per-frame breakdowns (Phase 12 style), not just headline numbers.

## What we expect to learn

Each of these is a useful learning regardless of outcome:

- **Confirmed**: ensembling on the Phase-13 val pushes F1 to the same 0.97 band as Phase-8 → publish the recipe, freeze production stack
- **Surprising**: ensembling helps less on this val split than on Phase-8 → val split is harder, OR the F5 recipe over-corrects → A/B test the recipe
- **Confirmed**: F5 recipe lifts per-member recall by ≥0.01 → adopt as the default training recipe going forward
- **Surprising**: F5 recipe leaves per-member F1 unchanged from a normal F1-trained member → the issue isn't the training objective, it's the data; reopen hard-negative mining as the primary lever
- **Confirmed**: hard-negative mining outperforms bulk frames at fixed cost → re-prioritise annotation budget toward FP regions identified by the ensemble
- **Surprising**: no improvement from hard negatives → architectural change needed (larger backbone, self-supervised pretraining on unlabelled drone imagery)

## Out of scope (defer until Phase 14 lands)

- Self-supervised pretraining on unlabelled drone imagery (Tier-3 lever, only worth it if H1/H2/H3 all under-perform)
- Multi-GPU training (Phase 14 already fits on a single GPU; multi-GPU is a refactor concern, not a model-quality concern)
- New architectures beyond B3/B4 (EfficientViT, EVA-02 — keep as fallback if the existing ensemble can't break 0.92)
- Inference-time TTA — Phase 6 showed +0.01-0.02 F1; cheap to add later, but not the bottleneck

## Implementation kickoff

```bash
# Generate the 6 training configs (one per arch × seed)
for ARCH in b3 b4; do
  for SEED in 7 42 123; do
    WANDB_PROJECT=hn_phase14_ensemble \
    WARM_START=best_models/phase8/${ARCH}_seed${SEED}/best_model.pth \
    SEED=${SEED} \
    AUG_MULT=1 BATCH_SIZE=8 NUM_WORKERS=16 \
    bash docker/build.sh   # build & tag
    # push to dockerkartok/herdnet:phase14-${ARCH}-s${SEED}, run remote
  done
done
```

Concrete next file edits (not done yet):
- New `configs/demo/phase14_ensemble_b3.yaml`, `phase14_ensemble_b4.yaml` — clone of `data_scaling_b4.yaml` with `CE.kwargs.weight=[0.1, 2.0]`, `epochs=10`, `evaluator.validate_on=f5_score`, `early_stopping*` set
- New `run_phase14.sh` — wrapper around 6 trainings (sequential local fallback) + Stage B ensemble inference
- Extend `tools/ensemble_infer.py` if needed to handle mixed-arch / per-checkpoint config reads (already supports cross-arch per ensemble_infer.py docstring)
- Stage C error-analysis script: copy `/tmp/error_analysis_phase8.py` from Phase 11 and re-point at the new detections CSV

## TODO checklist (for the eventual implementor)

- [ ] Create `configs/demo/phase14_ensemble_{b3,b4}.yaml` with the F5 recipe
- [ ] Verify `dockerkartok/herdnet:phase13-nfull-latest` accepts `CE.weight` and `validate_on=f5_score` overrides — add `LOSS_WEIGHT` and `VALIDATE_ON` env vars to `docker/train_entrypoint.sh` if not (similar to the `BATCH_SIZE` / `NUM_WORKERS` pattern)
- [ ] Write `run_phase14.sh` (6-member train + ensemble inference + sweep)
- [ ] Stage A — train 6 members (locally sequential or remote parallel)
- [ ] Stage B — ensemble inference + adapt_ts sweep on Phase-13 val
- [ ] Stage C — Phase-11-style error analysis on the ensemble's residual FNs/FPs
- [ ] **Decision gate**: kill criteria from above
- [ ] Stage D (conditional) — hard-negative mining round
- [ ] Phase 14 writeup as `docs/benchmarks/phase14_ensemble.md` (mirror Phase 13's structure)
