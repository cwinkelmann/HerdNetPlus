# al_v9 sweep findings (post phase-15 iter-7)

Multi-experiment training on the al_v9 split — a 2×2 factorial design
to isolate two production-relevant choices. Launched on a multi-GPU
big machine 2026-06-05 → 2026-06-08.

## 1. Design

| Axis | level A | level B |
|---|---|---|
| Dataloader | **old** (no H-anchor, `hard_negative_probability=0`, `empty_probability=0.20`) | **H-aware** (`hard_negative_probability=0.25`, `empty_probability=0.10`) |
| Fer in train | **out** (no Fer anywhere) | **in** (~162 phase-15-reviewed Fer + FPM01_24012023) |

Same val for all 4 runs: 1,123 imgs / 14 untouched non-Fern datasets
across 8 islands. Same warm-start (`phase8/b4_seed42/best_model.pth`).
Same hyperparameters: 30 epochs, batch=32, lr=6.4e-4, validate_on=f2_score,
adapt_ts=0.20, score_threshold=0.30, seed=42. Container:
`dockerkartok/herdnet:al_v9_sweep-latest`.

## 2. Headline results (best F2 per run)

| Run | Dataloader | Fer | State | **Best F2** | Best epoch | Recall | Precision | F1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| **A** | old | out | finished E30 | 0.847 | E16 | 0.859 | 0.800 | 0.829 |
| **B** | old | **in** | finished E30 | **0.853** ★ | E11 | 0.861 | 0.819 | 0.840 |
| **C** | H-aware | out | finished E30 | 0.844 | E21 | 0.837 | **0.873** | **0.855** |
| **D** | H-aware | **in** | failed × 2 (E12, E20) | 0.846 (partial) | E17 | 0.863 | 0.777 | 0.818 |

★ = best F2 across the sweep.

Multiple D runs were attempted (`20260605_al_v9_D_new_ferin_1dzxoa2y`
failed at E12; `20260606_al_v9_D_new_ferin_11t9jg8f` failed at E20).
Other failures: 2× A (`pdynccjm`, `93dnb8h6`) failed at start, 2× B
(`kuo5yfnv`, `6d3uo8z0`) failed mid-training. Only the three finished
runs above (and one of B's two completions) were analyzed.

## 3. Attribution (clean 2×2)

Dataloader effect on F2 (within Fer mode):
- fer_out: C − A = 0.844 − 0.847 = **−0.003**  (H-aware worse than old)
- fer_in:  D − B = 0.846 − 0.853 = **−0.007**  (H-aware worse than old)

Fer-in-train effect on F2 (within dataloader):
- old loader:    B − A = 0.853 − 0.847 = **+0.006**  (Fer helps)
- H-aware loader: D − C = 0.846 − 0.844 = **+0.002**  (Fer helps slightly)

**Reading:**
- Fer-in-train **helps by ~0.006 F2**. The phase-15-corrected Fer
  images add real training signal even on a non-Fer val.
- H-aware loader **does NOT improve F2** in this setup (−0.003 to
  −0.007). Suggests the 815 hard-negative anchors (post iter-5
  master state) weren't enough density to move the needle on a
  predominantly-iguana val, OR the 0.25/0.10 split over-allocated to
  vegetation-anchored crops at the cost of iguana-coverage.
- C **has the highest precision** (0.873) and highest F1 (0.855) —
  the H-aware loader DOES suppress vegetation FPs as intended, it
  just doesn't trade off favorably on F2 (which weights recall 4× more
  than precision).

## 4. Convergence shape

| Run | Best epoch | Wall epochs run | Notes |
|---|---:|---:|---|
| A | E16 | 30 | F2 peaks E16, slowly oscillates |
| B | **E11** | 30 | **early convergence**; F2 then DROPS 0.853 → 0.842 by E30 |
| C | E21 | 30 | slower convergence, slightly higher F1 ceiling |
| D | E17 | crashed | F2 trajectory unstable |

**Implication:** Training the al_v8/al_v9 recipe past ~15 epochs is
counterproductive — F2 actively regresses. The auto_lr `ReduceLROnPlateau`
isn't preventing this; early stopping on F2 with patience=3-5 would
have caught B's degradation cleanly.

## 5. D's instability

Both D runs (H-aware + Fer-in) failed mid-training without producing
a definite NaN signal in the summary. Hypotheses we did not eliminate:

- The Fer-in train set has high label density, so when an H-anchor
  fires on a Fer tile, the crop captures many other iguana points
  AND the H. The mixed signal might be destabilizing if the focal
  loss weighting is sensitive to multiple-keypoint crops.
- The H-aware code path passes labels via packed keypoint tuples
  `(x, y, label_id)`; if albumentations strips one of them under a
  rare augmentation path, the train loop sees malformed targets.
- Pure infrastructure: multiple containers running on the same host
  may have hit OOM, oncall reboot, or NCCL collective stalls.

**Action item:** if al_v10 (also H-aware + Fer-in) hits a similar
failure, capture the last 200 lines of the training log + the
albumentations stack trace and reproduce locally to isolate.

## 6. Comparison vs al_v8 baseline (local 4090)

al_v8 final = E7 best on a 1,123-img val with F2=0.860 / R=0.873 / P=0.814.
The al_v9 sweep ran on the same val + same warm-start with the same
recipe except scaled batch + lr. Result: al_v9 B peaked at F2=0.853
— slightly **below** al_v8's E7.

Possible reasons:
- batch=32 + lr=6.4e-4 (al_v9) vs batch=4 + lr=8e-5 (al_v8). The
  linear-scaled lr might be too aggressive for the post-phase-15
  master state. Worth testing batch=16 + lr=3.2e-4.
- al_v8 was trained on post-iter-5 master (75,821 iguana_point);
  al_v9 on the same. So data difference is zero.

In other words: **al_v8 E7 may still be the production-best model**.
The al_v9 sweep tells us Fer-in is good and H-aware doesn't yet win
on F2, but it didn't surpass the baseline.

## 7. Decisions taken from this sweep

1. **Production model candidate**: al_v8 E7 stays the leader. Use it
   for census inference until a clean al_v10 result is in.
2. **Always include Fer in train** going forward (al_v10 onwards).
3. **Keep iterating on the H-aware loader** — but tune the
   probability split: try `hard_negative_probability=0.15`,
   `empty_probability=0.15` (less aggressive H-anchoring). Or add
   more H pool first by reviewing more vegetation-FP-heavy datasets.
4. **Add early stopping on F2 with patience=5** to all future runs.
   Saves wall time + prevents the post-convergence degradation seen
   in B's E11 → E30 trajectory.

## 8. Pointers

- Sweep entrypoint: `docker/al_v9_sweep_entrypoint.sh`
- Per-experiment best_model.pth files: in each container's
  `/app/output/al_v9_<NAME>/<date>/<time>/best_model.pth`
- wandb runs: `karisu/hn_active_learning`, filter by tag
  `sweep_2x2`. Raw query script: `tools/check_al_v9_wandb.py`.
- This summary was generated 2026-06-08 from the wandb fetch.

End of findings.
