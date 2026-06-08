# LMDS thresholds: `adapt_ts` vs the new `score_threshold`

**Status:** Methodology note for HerdNet model evaluation. Written 2026-06-02
during al_v4 training tuning, after observing very low precision
(~0.14) under `adapt_ts=0.15`.

## TL;DR

- `adapt_ts` is a **per-image RELATIVE** threshold (cutoff =
  `adapt_ts × image_peak_score`). It behaves differently on every image
  and couples to the model's calibration.
- `score_threshold` is a new **absolute** threshold applied on top of
  `adapt_ts`. Predictable, image-independent. Default 0.0 (off) so
  pre-existing configs are unaffected.
- Recommendation: keep `adapt_ts` for its useful role (local-maxima
  sharpening) and use `score_threshold` for the actual precision/recall
  trade-off.

## What `adapt_ts` actually does

Code path: `animaloc/eval/lmds.py::LMDS._lmds`

```python
est_map_max = torch.max(est_map).item()           # peak score in THIS image
est_map = self._local_max(est_map…)                # find local maxima
est_map[est_map < self.adapt_ts * est_map_max] = 0  # threshold
```

The cutoff is **`adapt_ts × image_peak`**, not `adapt_ts` alone. Example
with `adapt_ts=0.15`:

| Image type | Peak score | Effective cutoff | Behaviour |
|---|---|---|---|
| Colony-dense, model confident | 0.99 | 0.15 | Strict — drops candidates < 0.15 |
| Sparse, one mediocre detection | 0.20 | 0.03 | Permissive — keeps everything ≥ 0.03 |
| Truly empty | 0.005 | 0.00075 | Lets through nearly all noise |

A `neg_ts=0.1` safety valve zeros entire images whose peak is below 0.1,
but anything with peak ≥ 0.1 escapes the floor and gets the per-image
treatment above.

## Why this complicates tuning

1. **Image-dependent behaviour** — same `adapt_ts` produces different
   effective thresholds per image. Tuning on the busy images doesn't
   predict behaviour on the sparse ones.
2. **Counterintuitive on empties** — lowering `adapt_ts` becomes MORE
   permissive on already-empty images, *not* less. The opposite of what
   you want when chasing precision on background.
3. **Couples to model calibration** — as training improves, peak scores
   rise, so the effective threshold tightens *without you touching it*.
   A working background-suppression fix (e.g.,
   higher `empty_probability`) can paradoxically over-tighten next
   epoch and look like a regression.
4. **Two-knob coupling** — `adapt_ts` and `neg_ts` interact
   non-monotonically near `est_map_max ≈ neg_ts`.

## The new `score_threshold`

A previously-dead `score_threshold` field already existed on
`LMDS.__init__` but was never enforced. As of 2026-06-02 it now applies
an absolute floor after `adapt_ts`:

```python
est_map[est_map < self.adapt_ts * est_map_max] = 0   # existing relative gate
if self.score_threshold > 0:
    est_map[est_map < self.score_threshold] = 0      # new absolute floor
```

**Backwards compatible:** default is `0.0` = no-op. Configs that don't
set `score_threshold` behave exactly as before.

## How to use

Via Hydra override on a training/inference call:

```bash
python tools/train.py … \
  training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=0.30 \
  training_settings.evaluator.kwargs.lmds_kwargs.score_threshold=0.50
```

Or via inference:

```bash
python tools/infer.py … --overrides \
  training_settings.evaluator.kwargs.lmds_kwargs.score_threshold=0.50
```

## Suggested tuning ladder

For active-learning where you want recall-then-precision review batches:

| Phase | adapt_ts | score_threshold | Intuition |
|---|---|---|---|
| Training validation (track convergence) | 0.30 | 0.0 | Historic baseline, image-dependent |
| Final production inference | 0.30 | **0.50** | Predictable precision floor |
| AL review surface (find missed iguanas) | 0.20 | **0.30** | Recall-biased but bounded |
| Recall-extreme exploration | 0.15 | **0.20** | Even more permissive, still grounded |

Sweep `score_threshold` ∈ {0.0, 0.3, 0.5, 0.7, 0.9} on a held-out val
to find the precision/recall trade-off curve for your deployment.

## Open follow-ups

- Compare `adapt_ts=0.50, score_threshold=0` vs
  `adapt_ts=0.30, score_threshold=0.50` head-to-head — is the absolute
  variant strictly better, or are there scenes where the relative
  gate still helps?
- Should `adapt_ts` default lower (e.g., 0.05) once `score_threshold`
  takes over the precision job? The `adapt_ts` value would then only
  do local-maxima refinement, not thresholding.
