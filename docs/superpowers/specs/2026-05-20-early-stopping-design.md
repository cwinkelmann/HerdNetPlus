# Early Stopping — Design

**Branch:** `feat/early-stopping-finish`
**Date:** 2026-05-20
**Related:** `doc/features.md` ("early stopping and model checkpointing")

## 1. Motivation

The user observed in run [`hn_phase13_data_scaling/j855ekjq`](https://wandb.ai/karisu/hn_phase13_data_scaling/runs/j855ekjq) that validation F1 oscillated around 0.86–0.88 for many epochs after the first, wasting GPU hours. `doc/features.md` proposes implementing early stopping by polling the wandb API for F1 history and aborting when no improvement happens for 3 epochs.

## 2. Audit: what already exists

Searching the codebase reveals early stopping is **already implemented in `Trainer`**:

| Aspect | Location | Status |
|---|---|---|
| Constructor flags | `animaloc/train/trainers.py:69-72` | `early_stopping`, `patience`, `min_delta`, `restore_best_weights` |
| State fields | `animaloc/train/trainers.py:227-235` | `wait`, `stopped_epoch`, `best_weights` |
| State reset on `start()` | `animaloc/train/trainers.py:309-312` | clears `wait`, `stopped_epoch`, `best_weights` |
| Check site (post-validation) | `animaloc/train/trainers.py:467-472` | breaks epoch loop when triggered |
| Check method | `animaloc/train/trainers.py:599-632` | min/max modes; `wait += 1` on no improvement; EMA-aware weight snapshot |
| Restore best weights | `animaloc/train/trainers.py:586-589` | reloads `best_weights` after the loop |
| wandb summary | `animaloc/train/trainers.py:591-594` | logs `best_validation` and `stopped_epoch` |
| Hydra → Trainer wiring | `animaloc/utils/train.py:389-392` | passes `early_stopping` and `early_stopping_patience` only |
| Example YAML usage | `configs/reference_data/delplanque2022_opt/dla102x_ft.yaml:201-202` | `early_stopping: False`, `early_stopping_patience: 20` |
| Metric source | `evaluator.evaluate(returns=validate_on)` → default `f1_score` (per `configs/demo/training_settings/evaluator.yaml`) | already the F1 score the user wants |

So the wandb-API polling design proposed in `features.md` is **not needed** — in-process F1 is already available at every validation epoch.

## 3. Gaps the audit found

1. **Fallback-default bug** (`animaloc/utils/train.py:391`): when `early_stopping_patience` is absent from the config, the fallback is the boolean `False` instead of an integer. Inside `_early_stopping_check`, `self.wait >= self.patience` with `patience=False` evaluates as `self.wait >= 0` → fires after the first non-improving epoch. Latent any time someone enables `early_stopping=True` without also setting `early_stopping_patience`.
2. **`min_delta` not wired**: the Trainer accepts it but `animaloc/utils/train.py` never forwards `cfg.training_settings.early_stopping_min_delta`. It is always 0.0.
3. **`restore_best_weights` not wired**: same — always defaults to `True`. Users cannot disable the snapshot/restore from config.
4. **No defaults in `configs/demo/training_settings/evaluator.yaml`**: the demo training_settings file (the base most experiment configs inherit from in spirit) doesn't declare these fields. Combined with gap #1 this is a footgun.
5. **No tests**: nothing in `tests/test_train.py` covers an early-stopping run.
6. **Undocumented**: neither `CLAUDE.md` nor `README` mentions the feature, which is exactly why `features.md` proposes building it.

## 4. Approaches considered

### A. Minimal finishing pass — **recommended**
Fix the four config-plumbing gaps, fix the fallback bug, add one regression test, document the feature. Keeps the existing in-process architecture untouched. ~5 small file edits, < 80 LoC net.

### B. Refactor into an `EarlyStopper` callback class
Extract `_early_stopping_check` + tracking state into `animaloc/train/callbacks/early_stopping.py` for cleaner isolation and unit-testability. Cleaner long-term, but invasive in a class (`Trainer`) that already has subclasses (`P2PNetTrainer`) and resume logic that touches the same state. Premature given the user's stated need.

### C. wandb-API poller (literal reading of `features.md`)
Rejected. Adds a network dependency, a wandb credentials requirement, polling lag, and duplicates state we already compute in-process. The only honest reason to consider it would be cross-process monitoring (a separate sidecar killing a training job from outside) — not a goal here.

**Decision: A.**

## 5. Architecture (Approach A)

```
configs/demo/training_settings/evaluator.yaml
    │   early_stopping: False                       ← new defaults
    │   early_stopping_patience: 20                 ← new
    │   early_stopping_min_delta: 0.0               ← new
    │   early_stopping_restore_best_weights: True   ← new
    ▼
animaloc/utils/train.py (main)
    │   - reads all four cfg.training_settings fields
    │   - patience fallback fixed from `False` → `10`
    │   - forwards min_delta and restore_best_weights to Trainer()
    ▼
animaloc/train/trainers.py (Trainer)
    │   constructor unchanged — already accepts all four kwargs
    ▼
Trainer.start()  →  per-epoch validation  →  _early_stopping_check(val_output, select, epoch)
                                                 │  on improvement: wait=0, snapshot best_weights (EMA-aware)
                                                 │  on stale: wait += 1
                                                 │  return wait >= patience
                                                 ▼
                                              break, log `stopped_epoch`, wandb summary,
                                              restore_best_weights → load state_dict
```

The metric monitored is whatever `evaluator.validate_on` selects (default `f1_score` for `HerdNetEvaluator`); `select` is `'max'` for that evaluator, satisfying the user's F1-maximization intent.

## 6. Components & changes (Approach A)

### 6.1 `animaloc/utils/train.py` — fix fallback bug and wire two more fields

**Current** (`animaloc/utils/train.py:389-392`):
```python
early_stopping=cfg.training_settings.early_stopping if hasattr(cfg.training_settings,
                                                               "early_stopping") else False,
patience=cfg.training_settings.early_stopping_patience if hasattr(cfg.training_settings,
                                                                  "early_stopping_patience") else False,
```

**Target**: pull all four fields, with integer/float fallbacks matching Trainer defaults:
```python
early_stopping=getattr(cfg.training_settings, "early_stopping", False),
patience=getattr(cfg.training_settings, "early_stopping_patience", 10),
min_delta=getattr(cfg.training_settings, "early_stopping_min_delta", 0.0),
restore_best_weights=getattr(cfg.training_settings, "early_stopping_restore_best_weights", True),
```
(Using `getattr` cleans up the `hasattr ... if ... else` repetition.)

### 6.2 `configs/demo/training_settings/evaluator.yaml` — add documented defaults

Append after the `warmup_iters` block:
```yaml
# Early stopping (in-process, on the evaluator's validate_on metric).
# Disabled by default. When enabled, training stops if validate_on does not
# improve by more than min_delta for `patience` consecutive validations.
early_stopping: False
early_stopping_patience: 20
early_stopping_min_delta: 0.0
early_stopping_restore_best_weights: True
```

### 6.3 `tests/test_train.py` — one regression test

Add a test that enables early stopping with a very small patience and a very small `min_delta`, runs 4–6 epochs of the existing demo pipeline, and asserts that `trainer.stopped_epoch > 0` (training stopped early) **or** that all epochs completed. The point is: it must not raise and must not hang. The test uses `_common_overrides(...)` plus:
```python
"training_settings.epochs=4",
"training_settings.early_stopping=True",
"training_settings.early_stopping_patience=1",
"training_settings.early_stopping_min_delta=10.0",  # impossibly large → forces no improvement
```

We need `main()` to return enough state to assert on. If `main()` does not currently return the trainer, the test can check the file system for `best_model.pth` and parse `[METRICS]` lines from the loguru log file — but the cleaner option is to extend the test to call `main()` and inspect `metrics["stopped_epoch"]` if `main` already plumbs that, or add a single line to surface it. **The implementation plan must verify what `main()` returns and choose the lightest assertion path.** No new return surface should be added just for the test.

### 6.4 Documentation

- Append a short "Early stopping" subsection to `CLAUDE.md` under **Config conventions** (or **Best Practices**), naming the four fields, the default behavior (off), and the fact that the monitored metric is `evaluator.validate_on`.
- Remove the early-stopping bullet from `doc/features.md` (or mark it done) so it is not re-proposed.

## 7. Data flow & error handling

- **Metric source**: in-process. `val_output` is whatever `evaluator.evaluate(returns=self.validate_on)` returns at line 446. No external call. No new failure modes.
- **EMA interaction**: already handled — `_early_stopping_check` snapshots `self.ema.module` when EMA is enabled. No change.
- **Resume**: `Trainer.resume()` does not currently reset early-stopping state. **Decision**: out of scope for this design. Document the limitation in the docs subsection; behavior is identical to before this work.
- **`select='min'` vs `'max'`**: `_early_stopping_check` already branches on `mode`. The same `select` value is passed from `start()`, derived from `evaluator.select_mode` in the config (e.g. `'max'` for F1). No change.

## 8. Testing strategy

- One new pytest case (Section 6.3) using the existing demo data and `_common_overrides` pattern; runs CPU-fast (epochs=4, batch_size=2, num_workers=0).
- Existing tests (`test_train_dla34`, `test_train_timm_dla34`, `test_train_convnext_camouflaged`) must continue to pass — they default to `early_stopping=False` so they should not be affected.
- No mocking of wandb, since `wandb_flag=False` for tests.

## 9. Out of scope

- Refactor to a callback class (Approach B).
- Polling wandb API (Approach C).
- Early-stopping behavior under `Trainer.resume()`.
- Reporting `stopped_epoch` to wandb at run-summary level beyond what already exists.
- The `Runpod remote training` section of `features.md`.

## 10. Acceptance criteria

1. `getattr` fallback in `animaloc/utils/train.py` returns `10` (int) for missing `early_stopping_patience` and `True` for missing `early_stopping_restore_best_weights`.
2. Setting `training_settings.early_stopping=True` and `training_settings.early_stopping_patience=1` in a config triggers stopping after one non-improving validation, in-process, without touching wandb.
3. The new pytest case passes on CPU.
4. All existing `tests/` tests still pass.
5. `CLAUDE.md` mentions the four config fields and that early stopping monitors `evaluator.validate_on`.
