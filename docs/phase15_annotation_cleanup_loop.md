# Phase 15 — model-assisted annotation cleanup loop

**Status**: planning, 2026-05-30. Replaces the abstract "structured annotation queue" entry in [`phase14_ensemble_strategy.md` § Phase 15 candidates](phase14_ensemble_strategy.md).
**Branch**: `convnext_extension` (continues on the same branch as Phase 13/14).
**Predecessors**: [Phase 13 data scaling](benchmarks/phase13_data_scaling.md), [Phase 13 error analysis](benchmarks/phase13_error_analysis.md), [Phase 14 ensemble (negative result)](benchmarks/phase14_ensemble.md).
**Prerequisite reading**: Phase 13 error analysis §1 — annotation quality is the bottleneck.

## Why this phase exists

Phase 13's error analysis showed that **93 % of "false positives" on val are model-correct, annotator-missed iguanas** (131 of 141 FPs score ≥ 0.99, with nothing in the 0.50–0.99 uncertainty band — the diagnostic signature of un-annotated true positives, not classification errors). If those 131 FPs are corrected, headline numbers shift from F1=0.891 / precision=0.917 to **F1≈0.93 / precision≈0.99** — closing the H1 ≥ 0.94 gate that ~36 GPU-hours of Phase 14 ensembling could never reach.

The training set is much larger (~7,200 frames vs 270 on val) and almost certainly has *at least* the same 5–10 % annotation-gap rate. Models trained on noisy GT are being *penalised* for finding iguanas the annotators missed — depressing both precision (extra "FPs") and recall (the model learns to suppress weak signals). Fixing the data fixes both.

## Goal

Iteratively use the current production model to surface annotation errors in the training pool, have a human verify and fix them, and re-train. Continue until every frame has been reviewed once by a model that did *not* train on it, and the rate of newly-found errors per iteration has converged.

Concrete success: report **final headline numbers on a frozen golden test set** (Phase-13 `test/`, 708 frames, 10+ sites, 5,102 annotations). The golden test is never edited; it is the unbiased measurement instrument. The expected outcome is F1 ≥ 0.93 on golden test after the loop, but the loop's *value* is the cleaner dataset, not a particular metric jump.

## Hypotheses

**H1. The training pool has ≥ 5 % annotation-gap rate.**
*Test*: After iteration 1 (current production model surveys top-100 candidates per holdout), classify each candidate as A/B/C/D (see § Edit categories). *Success*: ≥ 25 % of high-confidence FP candidates are category A (genuine missed iguanas).
*If false*: Phase 13's val finding was site-specific to FPE02 and the cleanup loop has lower expected value than projected. Reassess.

**H2. Model-assisted QA converges within ~10 iterations.**
*Test*: Track edits-per-iteration. *Success*: by iteration 5, edit count drops below 25 % of iteration 1's count; by iteration 10, below 5 %.
*If false*: Either we're catching genuine model blind spots and the loop is mining real signal indefinitely (good problem) or we're stuck in a self-reinforcing bias loop (bad problem). The two are distinguishable by comparing edits across model-architecture variants in a later sanity check.

**H3. Cleaner GT improves model recall, not just precision.**
*Test*: After each iteration's retrain, evaluate on the frozen golden test. *Success*: recall on golden test rises monotonically (or near-monotonically) across iterations.
*If false*: cleanup is recovering precision (the model is no longer penalised for finding genuine iguanas), but recall is bounded by something else — architecture, scale, or per-island distribution shift. That re-prioritises Phase 16 (per-island fine-tuning) or Phase 15-alt (zoom augmentation).

## Protocol

### Setup (one-time, before iteration 1)

1. **Lock the golden test set**. Phase-13 `test/` (708 frames, 5102 annotations, 10+ sites). Never edit. All headline metrics across all iterations are measured against this same set.
2. **Define the editable pool**. All training frames (~6,908) + val frames (270) = ~7,178 frames. The val set is editable because it's only used for hyperparameter tuning and per-iteration model selection; the golden test is the only frozen reference.
3. **Write the annotation rubric** (a separate doc, ~1 page). Define what counts as an iguana (size threshold, partial visibility, ambiguous shadow cases). All reviewers must follow it. Update as edge cases emerge but version the rubric in git.
4. **Stratify a holdout schedule**. Split the editable pool into 10 disjoint holdouts of ~720 frames each, stratified by site so every iteration's holdout has a similar site mix. Each frame appears in exactly one holdout. This is the deterministic coverage gate.
5. **Init the edit log**. CSV with columns: `iteration, holdout_id, image, x_old, y_old, x_new, y_new, score, category, site, reviewer, notes, timestamp`. Append-only. One row per edit.

### Iteration N (N = 1, 2, ..., 10)

| Step | Action | Cost |
|---|---|---|
| 1 | **Fine-tune** the previous iteration's model on the 90 % training pool (i.e. all editable frames *except* holdout N). 5–10 epochs from warm-start, F1 recipe (aug_mult=75, validate_on=f1, CE.weight=[0.1, 5.0]). Iteration 0's starting point is `output/phase13_Nfull_s42/.../best_model.pth`. | ~1.5 h GPU |
| 2 | Run stitched inference on holdout N at `adapt_ts=0.15` (lower than production 0.20 to catch more candidate gaps; we want recall here, not precision). | ~5 min |
| 3 | Compute candidate set: (a) all "FPs" with score ≥ 0.95 — likely missed iguanas; (b) "FNs" where the heatmap peak score ≥ 0.40 within 100 px of GT — likely off-by-location errors; (c) GT keypoints with no detection within 100 px AND no heatmap signal — possibly mis-annotated (false GT) or extremely hard. Cap each list at 50 per holdout. Use `tools/error_analysis.py` (already exists) to dump 256×256 crops with markers. | ~5 min |
| 4 | **Human reviews** the candidate crops (target: 50–150 candidates per iteration, ~1–2 h human time). For each: classify A/B/C/D (see § Edit categories), apply the fix to the underlying `herdnet_format.csv`, append a row to the edit log. | 1–2 h |
| 5 | Re-validate the holdout numbers post-edit (recompute precision/recall using the now-corrected GT). Append summary row to per-iteration metrics CSV. | ~5 min |
| 6 | **Evaluate the iteration-N model on the golden test set** at `adapt_ts=0.20` (production setting). Append summary row to per-iteration test metrics CSV. | ~10 min |
| 7 | Stop if convergence gate met (see § Stop conditions). Otherwise advance to iteration N+1. | — |

**Total per iteration**: ~1.5 h GPU + ~2 h human ≈ 3.5 h elapsed. 10 iterations ≈ 35 h elapsed. Spread over the work week, that's manageable.

### Stop conditions

The loop ends when **any** of these is met (whichever comes first):

1. **Coverage gate** (the primary stop): every frame in the editable pool has appeared in exactly one holdout = N reached 10.
2. **Convergence gate**: two consecutive iterations produce < 5 % of iteration 1's edit count. Indicates the model and humans have converged on the same view of the data.
3. **Plateau gate**: golden test F1 has not improved by ≥ 0.005 over the last 3 iterations. Indicates further data cleanup isn't shifting metrics.

Document which gate fired in the writeup.

## Edit categories

Every reviewed candidate gets exactly one label:

| Category | Meaning | Action |
|---|---|---|
| **A** — missed iguana | Model is correct; GT is incomplete | Add new keypoint to GT CSV |
| **B** — false GT | Annotator labelled a non-iguana | Remove keypoint from GT CSV |
| **C** — relocation | GT keypoint exists but ≥ 100 px from actual iguana centre | Update (x, y) in GT CSV |
| **D** — borderline | Ambiguous (shadow, partial, edge, occluded) | Flag — second reviewer required; no edit yet |
| **E** — confirmed FP | Model wrong, GT correct (genuine model failure) | No edit; record failure mode in notes (rock, stick, shadow, etc.) — feeds Phase 16/15-alt prioritisation |

Track A:B:C:D:E ratio per iteration. The trend over iterations is the diagnostic signal: A dominant in early iterations (model finding gaps), shifting toward E in late iterations (model finding its own real blind spots).

## Risks and mitigations

1. **Self-reinforcing bias**. A model trained on biased GT (e.g. annotators systematically miss rock-crevice iguanas) will keep missing them as a reviewer. *Mitigation*: every frame is reviewed by a model that did NOT train on it (the holdout structure guarantees this). *Sanity check*: at iteration 5, run an architecture-diverse model (e.g. a DLA-34 from `best_models/`) on a sample of already-reviewed frames. If DLA-34 finds candidates the ConvNeXt missed, both have the same blind spot and we have a real problem.
2. **Reviewer confirmation bias**. Humans accept "score=0.99 → iguana" without looking. *Mitigation*: review crops without scores visible (`tools/error_analysis.py` can be modified to optionally hide score in filename). Verify a random 10 % of category-A edits with a second reviewer.
3. **Annotation rubric drift**. Rules emerge implicitly across iterations. *Mitigation*: rubric is a versioned doc; every edge-case decision goes into it; reviewers refer to it before each session.
4. **Compute drift from fine-tune cumulation**. After 10 iterations of fine-tuning a fine-tuned model, the warm-start chain may have drifted from optimal. *Mitigation*: every 5 iterations, also train one from-scratch model on the current cleaned data and compare on golden test. If the from-scratch model beats the fine-tune chain by > 0.01 F1, switch.
5. **Per-iteration retrain failure** (NaN, OOM, hardware). *Mitigation*: the trainer's NaN-skip fix (`d24334d`) handles the in-batch case. For hard failures, the previous iteration's model is the fallback for inference on the next holdout — degrades gracefully.

## What we expect to learn

Each of these is a useful finding regardless of outcome:

- **Confirmed**: the editable pool has 5–10 % annotation gaps and cleaning them lifts golden-test F1 to ≥ 0.93 → ship the cleaned dataset, update the production model, freeze.
- **Confirmed but smaller**: < 5 % gap rate, F1 lift < 0.02 → annotation quality wasn't the dominant bottleneck after all, re-prioritise toward Phase 16 (per-island) or Phase 15-alt (zoom-aug).
- **Surprising**: gap rate is high (≥ 10 %) but recall on golden test plateaus → annotation gaps and per-site distribution shift are *both* contributing. Phase 16 still needed.
- **Surprising**: category-E rate spikes mid-loop → model is finding genuine failure modes worth cataloguing (Phase-11-style). Pivots toward Phase 15-alt and Phase 16.

## Out of scope

- **Architectural changes**. Phase 15 is data, not modelling. The model used per iteration is the same Phase-13 / Phase-14c architecture, fine-tuned. New architectures wait for Phase 17+.
- **Multi-annotator inter-rater agreement studies**. Useful but a separate effort; this phase assumes one or two reviewers following the same rubric.
- **Active learning on the * golden test*** (i.e. using model to find errors in test). That would destroy the measurement instrument. Test set is frozen for the duration of this phase.
- **Per-island specialisation**. Phase 16 territory; do not entangle.

## Implementation kickoff

### Files to create

- `docs/phase15_annotation_rubric.md` — the canonical edge-case rules. Started before iteration 1.
- `tools/cleanup_loop.py` — orchestrator wrapping `tools/error_analysis.py` + an interactive review UI (or just dump crops and have reviewer fill in a CSV by hand for iteration 1; UI for iteration 2+).
- `output/phase15_iter_<N>/` — per-iteration outputs (fine-tuned model, holdout detections, edit log row, golden-test metrics row).
- `data/phase15_edit_log.csv` — append-only edit log (the single source of truth for what changed and why).
- `data/phase15_holdout_schedule.csv` — the deterministic 10-way frame-to-holdout assignment (stratified by site).

### Concrete first steps (iteration 0 — the seed)

Before launching iteration 1, do the val-only QA pass that's already prepped:

1. Review the 60 cropped FPs at `docs/benchmarks/assets/error_analysis_phase13/fp/` (already on disk from the error analysis).
2. Fill out the edit-log columns for each (category A/B/C/D/E, action taken).
3. Apply the edits to `/home/christian/data/training_data/2026_05_08_data_scaling/val/herdnet_format.csv`.
4. Re-run `tools/per_site_metrics.py` and `tools/error_analysis.py` with the corrected val to record the post-QA val metrics.

This iteration-0 step is high-leverage and zero new compute. It establishes the baseline rubric, gives a calibrated estimate of the gap rate (which feeds H1), and produces the *post-QA* val number we'll use to validate the loop is working as intended in subsequent iterations.

### TODO checklist

- [ ] **Iteration 0** — review 60 val FP crops; produce post-QA val metrics
- [ ] Write `docs/phase15_annotation_rubric.md` (after iteration 0 surfaces real edge cases)
- [ ] Stratify the editable pool into 10 holdouts (`data/phase15_holdout_schedule.csv`)
- [ ] Iteration 1 fine-tune + holdout-1 inference + review pass
- [ ] Architecture sanity check at iteration 5 (DLA-34 reviewer)
- [ ] From-scratch retrain at iteration 5 — compare to fine-tune chain
- [ ] Iterations 2-10 (one per week is plenty)
- [ ] Final writeup as `docs/benchmarks/phase15_results.md` (mirror Phase 13's structure)
- [ ] Decision gate on Phase 16 (per-island) based on H3 outcome

## Bottom line

We spent ~36 GPU-hours of Phase 14 chasing F1 ≥ 0.94 via ensembling — and didn't move the needle. Phase 15 makes a 1-hour iteration-0 val-QA pass that plausibly *already* takes F1 to 0.93 on the existing model, then runs a multi-week iterative loop that costs ~35 h elapsed across human + GPU to clean the entire 7,200-frame editable pool. The data path is the higher-leverage one; this phase formalises it.
