# HerdNet Phase History — read this BEFORE starting a new experiment

Each phase has a benchmark doc under `docs/benchmarks/`. Before designing
a new run, check whether the question has already been answered.

## Phase Timeline

| Phase | Date | Question tested | Headline result | Doc |
|---|---|---|---|---|
| **1** | 2026-05-04 | Pre-cropped tiles vs ObjectAwareRandomCrop, basic vs augplus | ObjCrop + augplus wins; per-epoch cropped val is misleading; full-size stitched eval is the only metric that matters | `docs/benchmarks/phase1_augmentation_crop.md` |
| **2 / 2b** | — | Loss ablation, L2 threshold sweep | FocalLoss + CE settled; canonical params locked | `docs/benchmarks/phase2_loss_ablation.md`, `phase2b_l2_threshold_sweep.md` |
| **10** | — | Stitcher overlap sweep | Production overlap=120 chosen | `docs/benchmarks/phase10_overlap_sweep.md` |
| **11** | — | Error analysis v2 (which images fail and why) | Failures cluster on rocks/shadows, dense colonies | `docs/benchmarks/phase11_error_analysis_v2.md` |
| **12** | — | Metric plots / convergence | Convergence by epoch 10, plateau by 20+ | `docs/benchmarks/phase12_metrics_plots.md` |
| **13** | 2026-05-08 → 21 | Data-scaling curve N ∈ {19, 38, 76, 152, 304, 608, 1216, 2432, full=6908} | Log scaling: F1 0.709 → 0.882; knee at N≈304; production model = phase13_Nfull_s42 | `docs/benchmarks/phase13_data_scaling.md`, `phase13_error_analysis.md` |
| **14** | 2026-05-27 → 30 | 6-member cross-arch ensemble (B3×3 + B4×3) | **NEGATIVE** — all variants worse than single B4. Ensembling cements correlated FPs at this data scale | `docs/benchmarks/phase14_ensemble.md`, `phase14_test_set_eval.md` |
| **15** | 2026-05-30 → ongoing | Active-learning annotation cleanup via CVAT | iter-0 (50 images) + iter-1 (184 images) merged → +117 iguana_point, +112 not_iguana_but_similar_look in master. ~225 of 707 phase-13 val/test images human-reviewed | `docs/phase15_annotation_cleanup_loop.md` |
| **al_v2 / v3** | 2026-06-01 → 02 | First model trained on corrected master | r3 epoch 1 = F1 0.7138 on uncorrected new val (3000 random images). Uses loss_evaluation disabled to avoid full-size shape mismatch | see project memory `project_active_learning_v2`, `20260602_al_v3_b4_train_r3.log` |

## Locked production decisions — DO NOT re-run

Before proposing a "let's try X" experiment, check if it's already locked:

| Decision | Locked value | Source phase |
|---|---|---|
| Backbone | B4 ConvNeXt-Tiny via `HerdNetTimmConvNext_Camouflaged_V2` | Phase 14 9-model comparison |
| Loss | `herdnet_fmo03` (FocalLoss + weighted CE) | Phase 2 |
| Dataset recipe | full-size `Default/` + `ObjectAwareRandomCrop` at train time | Phase 1 |
| Stitcher overlap | 120px | Phase 10 |
| Training hparams | aug_mult=75 (default), lr=8e-5, batch_size=4, epochs=30, warmup=500, patience=8, validate_on=`f1_score` | Phase 1, Phase 13 |
| Production threshold | `adapt_ts=0.50` for precision-tuned, `0.20` for recall-tuned active-learning | Phase 13 |
| Ensembling | **don't** — proven worse than single B4 at this scale | Phase 14 |
| **CRITICAL — albumentations v2 fix** | `ObjectAwareRandomCrop.get_params_dependent_on_data(self, params, data)` (was `get_params_dependent_on_targets`) | training-insights SKILL.md "Bug Fixes Log" — without this, every crop is the top-left 512px and labels disappear |

## Active questions

| Question | Status | Owner doc |
|---|---|---|
| Does annotation cleanup boost F1 from 0.882 → ≥0.93? | Phase 15 in progress (iter-2 model trained 2026-06-02) | `docs/phase15_annotation_cleanup_loop.md` |
| Per-island fine-tuning vs mixed-site model? | Pending Phase 15 clean GT | — |
| Cross-island transfer (FMO03 → FMO05) | Phase 16 candidate | — |
| Seed robustness (only seed=42 used) | Pending; noted variance ~0.02 F1 | — |
| **Vegetation-FP problem**: model emits many high-confidence FPs on vegetation; ObjectAwareRandomCrop's `empty_probability=0.05` insufficient | Open as of 2026-06-02 iter-2 review. Candidate fixes: (a) raise `empty_probability` to 0.15-0.25; (b) empty-only pretraining stage; (c) push vegetation hard-negatives via iter-2 CVAT deletions | `memory/project_vegetation_fp_problem.md` |

## Reference card

When a future session asks "have we tried X?", check in this order:

1. **`tools/best_runs.py --top 20`** — current F1 leaderboard from `output/`.
2. **`docs/benchmarks/phase*.md`** — phase docs above.
3. **`.claude/skills/training-insights/SKILL.md`** — detailed technical notes (config snippets, hparams, bug fixes).
4. **`.claude/skills/herdnet-training/SKILL.md`** — lifecycle routing (data prep → training → inference).
5. **`.claude/ARCHITECTURE.md`** — package structure, Hydra composition.
6. **`configs/demo/*.yaml`** — canonical configs by name (`data_scaling_b4`, `phase14_ensemble_b4`, `fmo03_new_objcrop_augplus`, etc).
7. **`output/<phase>_<exp>/<date>/<time>/best_model.pth`** — every run keeps a checkpoint with `metrics` dict + `config` dict embedded. Inspect with `torch.load(p, weights_only=False)['metrics']`.
8. **`20260*_*.log`** files at repo root — chronological training logs (recent: `al_v3`, `al_v2`, phase14, phase13).

## When NOT to start a new experiment

If a session is about to propose any of these, push back first:
- "Let's try DLA-34 again" — phase 14 ranked it 7th, -2.5 F1 vs B4 ConvNeXt
- "Let's ensemble" — phase 14 ruled this out at our scale
- "Let's use plain ConvNeXt without camouflage features" — Phase-13 production already does (ConvNeXt-V2 BiFPN+DeformConv added complexity for no F1 gain in 9-model comparison)
- "Let's drop ObjectAwareRandomCrop" — Phase 1 proved -10 F1 from pre-cropping alone
- "Let's reduce aug_mult below 75 for production" — Phase 14 confirmed under-convergence
- "Let's evaluate on cropped 512 patches as the final F1" — Phase 1 explicit: cropped val is misleading; trust only full-size + stitcher

Exceptions: diagnostic runs (aug_mult=1 for clean signal) and active-learning iterations (smaller per-iter epoch counts to fit overnight) are explicitly permitted.
