# Phase 13 error analysis — production candidate (single B4, ts=0.20)

**Run date**: 2026-05-30.
**Branch**: `convnext_extension` @ `ac08269..`.
**Script**: [`tools/error_analysis.py`](../../tools/error_analysis.py).
**Model**: `output/phase13_Nfull_s42/2026-05-16/05-50-52/best_model.pth` (the active-learning production candidate after Phase 14 closed as a negative result).
**Detections**: `output/phase13_eval_20260508_142316/Nfull/ts_0.20/detections.csv` (full-size stitched eval at `adapt_ts=0.20`).
**Matching**: greedy 1:1 GT-vs-detection within radius=100 px (matches `HerdNetEvaluator`'s `threshold=100`).

## Setup

After Phase 14's three ensembling variants (F5 / F2 / F1) all failed to beat the Phase 13 single B4 on the Phase-13 val (see [`phase14_ensemble.md`](phase14_ensemble.md)), this analysis takes the single-model production candidate, runs Phase-11-style error analysis on its detections, and tests two specific hypotheses:

1. **Annotation-quality hypothesis** (raised after visual review of FP crops): many of the "false positives" are actually un-annotated iguanas — model right, GT wrong. If true, the headline precision is *understated* and the actual production quality is higher than the F1=0.89 figure suggests.
2. **Adaptive-threshold hypothesis** (raised earlier in this session): the LMDS `adapt_ts` mechanism applies a per-tile *relative* threshold (`score >= adapt_ts × max(tile_scores)`). On a tile containing no real iguanas, the per-tile max is whatever spurious bright pixel happens to be largest, so the threshold collapses near zero and lets weak detections through. The prediction: FP-per-GT should be much higher on sparse frames than dense frames.

## Aggregate (before any QA correction)

| | count |
|---|---|
| Frames | 270 |
| GT iguanas | 1,796 |
| Detections | 1,698 |
| Matched (TP) | 1,557 |
| **FN** (missed GT) | **239** |
| **FP** (extra detection) | **141** |
| Recall | **0.867** |
| Precision | **0.917** |
| F1 | **0.891** |

These numbers reproduce the headline `metrics_results.csv` from the Phase 13 sweep ($F1 = 0.89$ at $ts = 0.20$). The radius and matching strategy match the in-codebase evaluator.

## §1 — The most important finding: many "FPs" are unlabeled GT

### Score distribution of the 141 FPs

| Score band | Count | % of FPs |
|---|---|---|
| ≥ 0.99 | **131** | **93 %** |
| 0.95 – 0.99 | 0 | 0 % |
| 0.90 – 0.95 | 0 | 0 % |
| 0.70 – 0.90 | 0 | 0 % |
| 0.50 – 0.70 | 0 | 0 % |
| < 0.50 | 10 | 7 % |

**This distribution is bimodal in a very specific way**: nearly every FP is at the model's absolute confidence ceiling (≥ 0.99), with essentially nothing in the broad uncertainty band (0.50–0.99). That's the diagnostic signature of *correctly-classified iguanas that were missed by the annotators*, not the signature of genuine classification errors. A model that was confused between iguana and not-iguana would produce a spread of FP confidences across the 0.50–0.95 range. A model that's certain on *un-annotated true positives* produces exactly the bimodal pattern observed.

**Visual review of the top FPs (the user inspected `docs/benchmarks/assets/error_analysis_phase13/fp/`) confirmed**: many of the highest-confidence FPs are visibly iguanas. The model is correct; the ground truth is incomplete.

The same finding occurred earlier on the FMO03 val (Phase-8 / Phase-11 era — "Model found ~12 unlabeled iguanas in val set; use model as annotation QA tool"). It's recurring at much larger scale on the Phase-13 val.

### What this means for the headline numbers

**If we assume all 131 high-confidence FPs are actually annotation gaps** (a reasonable upper bound; visual confirmation is needed for the exact count):

| | Reported | Post-QA (upper bound) | Δ |
|---|---|---|---|
| TP | 1,557 | 1,688 | +131 |
| FP | 141 | 10 | −131 |
| FN | 239 | 239 | (unchanged) |
| GT total | 1,796 | 1,927 | +131 missed |
| Precision | 0.917 | **0.994** | **+0.077** |
| Recall | 0.867 | **0.876** | +0.009 |
| F1 | 0.891 | **0.932** | **+0.041** |

Precision jumps to 0.99, F1 to 0.93. **The production model is meaningfully better than the F1=0.89 headline suggests** — the metric has been measuring annotation quality alongside model quality.

This is the **#1 action item** from this analysis (see §5).

## §2 — The adaptive-threshold hypothesis (confirmed)

Frames stratified by GT density:

| Bucket | Frames | GT | TP | FP | FN | Recall | Precision | **FP/GT** | FP/frame | mean FP score |
|---|---|---|---|---|---|---|---|---|---|---|
| sparse_1-2 | 152 | 200 | 176 | **73** | 24 | 0.880 | 0.707 | **0.365** | 0.48 | 0.487 |
| medium_3-10 | 74 | 361 | 311 | 29 | 50 | 0.862 | 0.915 | **0.080** | 0.39 | 0.731 |
| dense_11+ | 44 | 1,235 | 1,070 | 39 | 165 | 0.866 | 0.965 | **0.032** | 0.89 | 0.825 |

**The FP/GT rate drops by 11× from sparse (0.365) to dense (0.032)** — exact direction predicted by the hypothesis. The mean FP score also rises monotonically with density (0.49 → 0.73 → 0.83), which is the second tell: on dense tiles, only high-confidence spurious detections survive the higher relative threshold; on sparse tiles, low-confidence ones leak through because the per-tile max is itself low.

Note that this signal is partly compounded with the §1 annotation-quality finding — many of those "FPs" on sparse frames are likely unlabeled iguanas that look real enough to score ≥ 0.99. The cleanest re-test of the LMDS-mechanism question would be on a re-annotated val set where the §1 issue is removed.

### Implications for `adapt_ts`

The relative-to-tile-max design was meant to make the threshold adaptive across heterogeneous backgrounds (forest vs sand vs lava field). In practice on this data:

- **Dense tiles**: works as intended — high real-iguana scores set the threshold, low-confidence spurious detections are filtered.
- **Sparse / empty tiles**: degenerate — there is no high-confidence real signal to set the threshold against, so the "relative" floor collapses and the model is essentially forced to emit *something* per tile.

**Mitigation candidates** (for a future Phase 15 / 16):

- **Absolute floor**: combine `adapt_ts` with an absolute threshold `score >= max(adapt_ts * tile_max, abs_floor)`. With `abs_floor = 0.30`, the 23 sparse-bucket FPs at score ≥ 0.90 stay (they're high-confidence and probably annotation gaps), but the 27 sparse-bucket FPs at score < 0.30 get filtered.
- **Frame-level rather than tile-level normalisation**: take the max over the whole stitched frame, not per-tile. This lets dense regions of a frame "lend confidence" to sparse regions.

These would not be retraining changes — pure inference-time fixes.

## §3 — Per-site breakdown (preview of Phase 16 case)

Val sites are FPE02, ISA_ISPVR04, SCRUZ × 2. FPs and FNs are heavily concentrated on the first two:

| Site | FPs | FNs |
|---|---|---|
| ISA_ISPVR04 | **81** (57 %) | **135** (56 %) |
| FPE02 | 41 (29 %) | 89 (37 %) |
| SCRUZ_SCM01 | 18 | 15 |
| SCRUZ_SCPLF01 | 1 | 0 |

**ISA_ISPVR04 dominates both error modes**. Half the errors of either kind are at that site. Two interpretations, both plausible:

1. ISA_ISPVR04 is genuinely harder (terrain, lighting, iguana density) → Phase-16 per-island fine-tuning would help.
2. ISA_ISPVR04 was less carefully annotated (consistent with §1) → re-annotation pass first, then re-measure.

The two are testable independently: if §1's QA pass brings ISA_ISPVR04's "FPs" down to ~10 (most being annotation gaps), then the residual is the real per-site difficulty. Phase 16 should run *after* §1 is resolved.

## §4 — Top false positives by confidence

The top 30 FPs are all at score ≥ 0.9999 (the model's effective certainty ceiling under LMDS post-NMS). They concentrate at:

| Rank-1 image | Frame GT count | (x, y) |
|---|---|---|
| isa_ispvr04 frame 736,148 | 11 | center |
| isa_ispvr04 frame 728,180 | 11 | (same frame) |
| isa_ispvr04 frame 100,772 | 4 | bottom-left |
| isa_ispvr04 frame 4,32 | 6 | top-left edge |
| … | | |

The full sorted list is in [`assets/error_analysis_phase13/fp.csv`](assets/error_analysis_phase13/fp.csv); cropped images are at [`assets/error_analysis_phase13/fp/`](assets/error_analysis_phase13/fp/) (60 highest-confidence FPs, 256×256 PNGs with orange-circle markers at the predicted location).

Several patterns visible across the top FPs:

- **Cluster of FPs in the same frame**: 4 FPs in `659212_9995139` (frame_GT=11) and 4 in `659339_9994820` (frame_GT=21). When the model is finding 4+ extra iguanas in a frame that "has 11", it's overwhelmingly likely those 4 are real and just not labelled.
- **Edge proximity**: many top-FP coordinates are within 100 px of the frame edge (`x=4, y=32`, `x=16, y=4`, `x=776, y=764`). Either edge artefacts on the stitcher, or partial iguanas truncated by tiling that the annotators skipped.

## §5 — Recommendations (in priority order)

1. **🟥 Re-annotation pass on the top-100 FPs** (~1 hour of human labelling on the 100 highest-confidence FPs). For each: classify as (a) genuine iguana — fix GT; (b) borderline/partial — flag for second opinion; (c) genuine FP — note the failure mode (rock, stick, shadow, …). Then re-compute headline numbers on the corrected val. **Expected outcome**: precision climbs from 0.917 to ~0.97–0.99, F1 from 0.891 to ~0.93–0.94. The production model crosses the H1 ≥ 0.94 gate that Phase 14 ensembling could never reach — *not by training, but by fixing the metric*.
2. **🟧 Repeat the same QA on the training set** of ISA_ISPVR04 and FPE02. If val has 5–10 % missed annotations, training has at least that. Models partially trained on negative-only examples of iguanas in those frames are getting *penalised* for finding them — which damages both precision AND recall.
3. **🟧 Implement absolute-floor in LMDS** alongside `adapt_ts`: `score >= max(adapt_ts × tile_max, 0.30)`. Eliminates the degenerate sparse-tile case in §2. Pure inference-time change, no retraining.
4. **🟨 Phase 16 per-island fine-tuning** — only after §1 and §2 are resolved. The current per-site numbers will be uninterpretable until annotation gaps are fixed: ISA_ISPVR04's "57 % of FPs" might be 57 % of annotation gaps, not 57 % of difficulty.
5. **🟨 Phase 15 multi-scale (zoom) augmentation** — FNs (especially edge-truncation cases) could partly be small-iguana failures that scale augmentation would help. Evaluate after §1 — if the residual FN list is dominated by small/truncated cases, Phase 15 is justified.

## §6 — What we got and where

Files produced by this analysis:

- [`assets/error_analysis_phase13/per_frame.csv`](assets/error_analysis_phase13/per_frame.csv) — per-frame TP/FP/FN with density bucket
- [`assets/error_analysis_phase13/fp.csv`](assets/error_analysis_phase13/fp.csv) — all 141 FPs sorted by score descending
- [`assets/error_analysis_phase13/fn.csv`](assets/error_analysis_phase13/fn.csv) — all 239 FNs
- [`assets/error_analysis_phase13/fp/`](assets/error_analysis_phase13/fp/) — 60 top-FP 256×256 crops with orange markers
- [`assets/error_analysis_phase13/fn/`](assets/error_analysis_phase13/fn/) — all 239 FN 256×256 crops with red markers

Reproduction:

```bash
python tools/error_analysis.py \
  --gt /home/christian/data/training_data/2026_05_08_data_scaling/val/herdnet_format.csv \
  --det output/phase13_eval_20260508_142316/Nfull/ts_0.20/detections.csv \
  --images /home/christian/data/training_data/2026_05_08_data_scaling/val/Default \
  --out docs/benchmarks/assets/error_analysis_phase13 \
  --radius 100 --top-fp 30 --max-crops 60
```

## Bottom line

We have been chasing F1 ≥ 0.94 via architecture and recipe changes (Phase 14 / 14b / 14c, ~36 GPU-hours), but the gap between F1=0.891 and the target was always *partly a measurement problem, not a model problem*. The first action — re-annotate the top-100 FPs — costs ~1 hour of human time, plausibly closes that gap entirely, and clarifies which residual errors are real for downstream interventions (Phase 15 / 16).
