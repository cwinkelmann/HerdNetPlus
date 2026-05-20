# Early Stopping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the partially-implemented early stopping feature in `Trainer` by fixing a fallback bug, wiring two unused config fields, adding demo defaults, adding a regression test, and documenting it.

**Architecture:** Approach A from the design spec (`docs/superpowers/specs/2026-05-20-early-stopping-design.md`). In-process check on `evaluator.validate_on` runs after every validation epoch; on no improvement for `patience` consecutive validations, the epoch loop breaks. EMA-aware best-weight snapshot and restore are already wired in `animaloc/train/trainers.py`. This plan adds nothing new to the Trainer — it fixes config plumbing only.

**Tech Stack:** Python 3.11+, PyTorch, Hydra/OmegaConf, pytest, loguru.

**Reference files (read-only context for the implementer):**
- `animaloc/train/trainers.py:215-235` — Trainer state init for early stopping
- `animaloc/train/trainers.py:295-296` — reset on `start()`
- `animaloc/train/trainers.py:430-440` — check site in epoch loop
- `animaloc/train/trainers.py:540-545` — wandb summary write
- `animaloc/train/trainers.py:435` — `stopped_epoch` assignment
- `tests/test_train.py:1-80` — existing test scaffold and `_common_overrides`
- `configs/demo/training_settings/evaluator.yaml` — demo training settings

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `tests/test_train.py` | Modify (append) | Add two regression tests for early stopping behavior |
| `animaloc/utils/train.py` | Modify (lines 389-392) | Fix `else False` fallback bug; forward `min_delta` and `restore_best_weights` from config |
| `configs/demo/training_settings/evaluator.yaml` | Modify (append) | Add four `early_stopping_*` default fields |
| `CLAUDE.md` | Modify (append to **Config conventions** section) | Document the four config fields and that `evaluator.validate_on` is the monitored metric |
| `doc/features.md` | Modify (mark bullet done) | Remove obsolete early-stopping bullet |

No new files are created. No new functions are added to the production code — the work is entirely about wiring and tests.

---

## Task 1: Add failing regression tests for early-stopping behavior

**Files:**
- Modify: `tests/test_train.py` (append two new test functions after `test_train_convnext_camouflaged`)

**Background for the engineer:**
- Tests use the existing demo HF data via the `training_data` fixture (defined in `tests/conftest.py`) and the `load_config` fixture.
- `main(cfg)` from `animaloc.utils.train` returns the Hydra working directory as a `Path`. Loguru writes a `YYYYMMDD_training.log` file into that directory. We assert on the presence/absence of the literal log line `"Early stopping triggered at epoch"`.
- Hydra overrides for fields not present in the YAML must use the `++` prefix (force-add); fields already in the YAML use plain `key=value`. At this point the four `early_stopping_*` fields are NOT in `configs/demo/training_settings/evaluator.yaml`, so we use `++` here. After Task 3 the `++` could be dropped, but leaving it keeps the test independent of the YAML defaults.
- The `dla34_delplanque` config is used because the existing `test_train_dla34` already exercises it on this dataset, so the test environment is known good.

- [ ] **Step 1: Write the two failing tests**

Append to `tests/test_train.py`:

```python
def _read_training_log(work_dir):
    """Return concatenated contents of all *_training.log files in work_dir."""
    logs = list(Path(work_dir).glob("*_training.log"))
    assert logs, f"No *_training.log found in {work_dir}"
    return "\n".join(p.read_text() for p in logs)


@pytest.mark.slow
def test_early_stopping_default_patience_does_not_trigger(
    load_config, training_data, tmp_output_dir
):
    """Enabling early_stopping with no explicit patience must fall back to a
    sane integer default (10), not the boolean `False`. With only 2 epochs,
    training must run to completion — no early-stop log line should appear.

    Pre-fix behavior: patience defaults to `False` (== 0), so `wait >= patience`
    is True from the first validation and training stops at epoch 1.
    """
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
        "training_settings.epochs=2",
        "++training_settings.early_stopping=True",
        # Deliberately omit patience — exercises the fallback path.
    ]
    cfg = load_config("dla34_delplanque", overrides=overrides)
    work_dir = main(cfg)
    log_text = _read_training_log(work_dir)
    assert "Early stopping triggered at epoch" not in log_text, (
        "Default fallback should prevent early stopping from firing in 2 epochs; "
        "if this fires, the patience fallback is wrong."
    )


@pytest.mark.slow
def test_early_stopping_triggers_when_metric_does_not_improve(
    load_config, training_data, tmp_output_dir
):
    """With patience=1 and an unreachable min_delta (F1 is in [0,1]),
    no validation can register as an improvement, so training must stop
    at the first validation epoch.
    """
    overrides = _common_overrides(training_data, tmp_output_dir) + [
        "datasets.num_classes=7",
        "++datasets.class_def={1: buffalo, 2: elephant, 3: kob, 4: topi, 5: warthog, 6: waterbuck}",
        "losses.CrossEntropyLoss.kwargs.weight=[0.1,1.0,2.0,1.0,6.0,12.0,1.0]",
        "training_settings.epochs=3",
        "++training_settings.early_stopping=True",
        "++training_settings.early_stopping_patience=1",
        "++training_settings.early_stopping_min_delta=10.0",
    ]
    cfg = load_config("dla34_delplanque", overrides=overrides)
    work_dir = main(cfg)
    log_text = _read_training_log(work_dir)
    assert "Early stopping triggered at epoch" in log_text, (
        "Patience=1 with unreachable min_delta should stop at epoch 1."
    )
```

- [ ] **Step 2: Run the new tests and confirm the failure mode**

Run:
```bash
pytest tests/test_train.py::test_early_stopping_default_patience_does_not_trigger \
       tests/test_train.py::test_early_stopping_triggers_when_metric_does_not_improve \
       -v -m slow
```

Expected:
- `test_early_stopping_default_patience_does_not_trigger` **FAILS** with `AssertionError` mentioning "patience fallback is wrong" — because `animaloc/utils/train.py:391` falls back to `False`, `wait >= False` is `wait >= 0`, and training stops at epoch 1.
- `test_early_stopping_triggers_when_metric_does_not_improve` **PASSES** — the existing Trainer code already supports the trigger path correctly when patience is set explicitly.

If both pass on the first run, stop and re-check: the bug should still exist before Task 2. The likely cause is that `min_delta` and `restore_best_weights` overrides are being silently ignored (they are, until Task 2) — that doesn't affect this assertion, so it's fine. But if test #1 passes, the fallback may have already been fixed; verify against `animaloc/utils/train.py:391` before proceeding.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_train.py
git commit -m "$(cat <<'EOF'
test: add early stopping regression tests

test_early_stopping_default_patience_does_not_trigger currently FAILS due to
the False-fallback bug at animaloc/utils/train.py:391 (patience=False makes
wait>=patience always true). Will be fixed in next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Fix the config plumbing in `animaloc/utils/train.py`

**Files:**
- Modify: `animaloc/utils/train.py:389-392`

**Background for the engineer:**
- The bug: `else False` makes `patience` a boolean when the field is absent. Python treats `False == 0` in comparisons, so `self.wait >= self.patience` (line ~437 in `animaloc/train/trainers.py`) becomes `wait >= 0` → always True → stops on the first validation.
- The Trainer constructor (`animaloc/train/trainers.py:69-72`) already accepts `min_delta` and `restore_best_weights`. They're just not being forwarded.
- We switch to `getattr(...)` for clarity. Default values must match the Trainer's own defaults: `early_stopping=False`, `patience=10`, `min_delta=0.0`, `restore_best_weights=True`.

- [ ] **Step 1: Apply the edit**

Replace lines 389-392 in `animaloc/utils/train.py` (the four lines passing `early_stopping=` and `patience=` to the Trainer constructor) with:

```python
        early_stopping=getattr(cfg.training_settings, "early_stopping", False),
        patience=getattr(cfg.training_settings, "early_stopping_patience", 10),
        min_delta=getattr(cfg.training_settings, "early_stopping_min_delta", 0.0),
        restore_best_weights=getattr(cfg.training_settings, "early_stopping_restore_best_weights", True),
```

The surrounding lines (the `early_stopping=` kwarg through to `ema_decay=ema_decay,` and the closing `)`) stay as they are; only these four assignments change.

- [ ] **Step 2: Re-run the two early-stopping tests**

Run:
```bash
pytest tests/test_train.py::test_early_stopping_default_patience_does_not_trigger \
       tests/test_train.py::test_early_stopping_triggers_when_metric_does_not_improve \
       -v -m slow
```

Expected: **both PASS**.

- `test_early_stopping_default_patience_does_not_trigger` now passes because the patience fallback is `10`, and 2 epochs cannot exhaust a patience of 10.
- `test_early_stopping_triggers_when_metric_does_not_improve` continues to pass.

- [ ] **Step 3: Commit**

```bash
git add animaloc/utils/train.py
git commit -m "$(cat <<'EOF'
fix(train): correct early_stopping config fallbacks and wire min_delta/restore_best_weights

The previous fallback for missing early_stopping_patience was the literal
False, which Python evaluates as 0 in `wait >= patience`, causing early
stopping to fire on the first validation whenever the field was absent.
Replaced the four hasattr-or-False expressions with getattr() calls using
integer/float defaults that match the Trainer constructor's own defaults.
Also forwards min_delta and restore_best_weights, which the Trainer already
accepts but the previous code never passed through.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add early-stopping defaults to the demo training_settings config

**Files:**
- Modify: `configs/demo/training_settings/evaluator.yaml` (append after the `warmup_iters` line)

**Background for the engineer:**
- The demo training_settings YAML is the canonical reference for what fields a training config should declare. Today it omits early-stopping fields, which is why some experiment configs forget to set them and silently inherit the (previously buggy) fallbacks.
- After this task, downstream configs can override these fields with plain `training_settings.early_stopping=True` (no `++` needed), although our tests keep using `++` to remain decoupled.

- [ ] **Step 1: Append the four defaults to the YAML**

Open `configs/demo/training_settings/evaluator.yaml`. Locate the line `warmup_iters: 1500`. Immediately after it, add a blank line and then:

```yaml

# Early stopping (in-process, on the evaluator's validate_on metric).
# Disabled by default. When enabled, training stops if validate_on does not
# improve by more than min_delta for `patience` consecutive validations.
early_stopping: False
early_stopping_patience: 20
early_stopping_min_delta: 0.0
early_stopping_restore_best_weights: True
```

Do not move existing fields. Do not touch the `auto_lr` block (its own `patience` is for `ReduceLROnPlateau`, unrelated).

- [ ] **Step 2: Re-run both early-stopping tests and the existing demo training tests to confirm no regression**

Run:
```bash
pytest tests/test_train.py -v -m slow
```

Expected:
- `test_train_dla34`, `test_train_timm_dla34`, `test_train_convnext_camouflaged` — all PASS (they don't set early_stopping, so default `False` keeps the feature off).
- `test_early_stopping_default_patience_does_not_trigger` — PASS.
- `test_early_stopping_triggers_when_metric_does_not_improve` — PASS.

- [ ] **Step 3: Commit**

```bash
git add configs/demo/training_settings/evaluator.yaml
git commit -m "$(cat <<'EOF'
config: declare early_stopping defaults in demo training_settings

Adds the four early_stopping_* fields (disabled by default, patience=20,
min_delta=0.0, restore_best_weights=True) to the canonical demo training
settings. Downstream configs that inherit from this file no longer need
the ++ prefix to enable early stopping.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Document early stopping and close out the features.md bullet

**Files:**
- Modify: `CLAUDE.md` (append a subsection under **Best Practices → Config conventions**)
- Modify: `doc/features.md` (mark the early-stopping bullet done or delete it)

**Background for the engineer:**
- The reason `doc/features.md` requested this feature is that nobody documented its existence. The doc fix is part of the deliverable.
- `CLAUDE.md` has a **Config conventions** subsection. Add the new content immediately after it (before the next `###` heading).

- [ ] **Step 1: Edit `CLAUDE.md`** — append after the existing **Config conventions** bullet list:

```markdown
### Early stopping

Training supports in-process early stopping on the evaluator's `validate_on`
metric (F1 score by default for `HerdNetEvaluator`). Configure via
`training_settings`:

| Field | Default | Meaning |
| --- | --- | --- |
| `early_stopping` | `False` | Enable/disable the check entirely. |
| `early_stopping_patience` | `10` (Trainer) / `20` (demo YAML) | Consecutive validations without improvement before stopping. |
| `early_stopping_min_delta` | `0.0` | Minimum delta that counts as an improvement. |
| `early_stopping_restore_best_weights` | `True` | After stopping, reload the best snapshot (EMA-aware) into `self.model`. |

Triggering writes `[Early stopping triggered at epoch N]` to the training
log and sets `wandb.run.summary['stopped_epoch']`. The feature does not
currently survive `Trainer.resume()` — a resumed run starts with a fresh
patience counter.
```

- [ ] **Step 2: Edit `doc/features.md`** — replace the existing early-stopping section (lines 14-23) with:

```markdown
### early stopping and model checkpointing ✅ done (2026-05-20)

Implemented in-process. See `CLAUDE.md` → **Early stopping** and the design
spec at `docs/superpowers/specs/2026-05-20-early-stopping-design.md`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md doc/features.md
git commit -m "$(cat <<'EOF'
docs: document early-stopping configuration in CLAUDE.md

Adds a CLAUDE.md subsection describing the four early_stopping_* fields,
the default behavior, the monitored metric (evaluator.validate_on), and
the resume() limitation. Marks the corresponding bullet in features.md
as done with a pointer to the design spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final verification

**Files:** none

**Background for the engineer:**
- Run the full `tests/` suite (slow and non-slow) to confirm no regression in unrelated tests.
- Two tests changed Trainer-invocation arguments only via Hydra config, so unrelated tests should be unaffected.

- [ ] **Step 1: Run the fast test suite**

Run:
```bash
pytest tests/ -v -m "not slow"
```

Expected: PASS for the non-slow tests (`test_install`, `test_inference`, and any other quick ones). Document any pre-existing failures so the user sees this work didn't cause them.

- [ ] **Step 2: Run the slow training tests**

Run:
```bash
pytest tests/test_train.py -v -m slow
```

Expected: all five tests PASS — three pre-existing (`test_train_dla34`, `test_train_timm_dla34`, `test_train_convnext_camouflaged`) plus the two new ones.

- [ ] **Step 3: Sanity-check git state**

Run:
```bash
git log --oneline main..HEAD
```

Expected: 5 commits (1 spec [already on branch], 4 implementation/test/docs). The spec commit is `687e9a4`; the four added by this plan should be: test commit, fix commit, config commit, docs commit. No commits should touch untracked files like `best_models/` or `.idea/`.

- [ ] **Step 4: Final report to the user**

Print a summary covering: the five commits, the two new tests, the bug-fix description, and a recommendation that the user run a real training experiment with `training_settings.early_stopping=True training_settings.early_stopping_patience=3` to confirm wall-clock behavior on a real GPU run.

---

## Self-Review

**Spec coverage (cross-reference against `docs/superpowers/specs/2026-05-20-early-stopping-design.md`):**

- §6.1 — fix `else False` and wire `min_delta` + `restore_best_weights` → **Task 2** ✓
- §6.2 — add four defaults to demo `evaluator.yaml` → **Task 3** ✓
- §6.3 — one regression test → **Task 1** (two tests, since the bug-fix test and the trigger test exercise different code paths) ✓
- §6.4 — `CLAUDE.md` subsection + close `doc/features.md` bullet → **Task 4** ✓
- §10 acceptance criteria 1 (getattr fallback) → covered by Task 2 + verified by Task 1 test #1 ✓
- §10 acceptance criteria 2 (trigger on patience=1) → covered by Task 1 test #2 ✓
- §10 acceptance criteria 3 (new test passes on CPU) → Task 5 ✓
- §10 acceptance criteria 4 (no existing-test regression) → Task 5 ✓
- §10 acceptance criteria 5 (CLAUDE.md doc) → Task 4 ✓

**Placeholder scan:** no TBD / TODO / "implement later" remains. Every step shows complete code or a complete command.

**Type / signature consistency:**
- `getattr(cfg.training_settings, "...", DEFAULT)` defaults: `False` (bool), `10` (int), `0.0` (float), `True` (bool) — match `Trainer.__init__` defaults at `animaloc/train/trainers.py:69-72`.
- `_read_training_log(work_dir)` test helper accepts `Path` (what `main()` returns); calls `.glob("*_training.log")` which works on `Path` objects.
- Hydra override prefix `++` used consistently in Task 1 because the YAML fields don't exist yet. After Task 3 it would still work — `++` force-adds even when the field is present.

No issues found.
