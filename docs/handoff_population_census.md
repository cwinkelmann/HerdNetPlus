# Marine Iguana population-census handoff

Snapshot 2026-06-05. Authored to bridge into a fresh session that picks
up the project at the point where the al_v9 sweep is running and the
next big question is: **how do we turn the trained model into a full
Galapagos-wide iguana population census?**

This document is self-contained — a new collaborator should be able to
read just this file (plus `CLAUDE.md` and `.claude/ARCHITECTURE.md`) and
know where to look for everything.

---

## 1. Project context

HerdNet is a PyTorch point-detection framework
(`animaloc` package). The fork lives at
`/home/christian/hnee/HerdNet`. It started as a generic
animal-localization framework (Delplanque et al.) and has been adapted
over many phases for **Marine Iguana detection in Galapagos drone
imagery** — the iguanas are tiny, often camouflaged on volcanic rock,
and the dataset is heavily annotator-curated.

The architecture in production is `CamouflageHerdNetConvNeXtV2` —
ConvNeXt-V2 tiny backbone (28M params) + camouflage-aware head, in
`animaloc/models/herdnet_timm_convnext_camouflaged_v2.py`. ConvNeXt
wins out over DLA34, EfficientViT, DINOv2 — the latter's coarse patch
embedding destroys signal for the small iguana targets (see
memory `feedback_dinov2_small_objects`).

### Repos involved

| repo | role |
|---|---|
| `/home/christian/hnee/HerdNet` | training/inference framework |
| `/home/christian/hnee/active-learning` | data prep + CVAT pipeline |
| `/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06` | master Hasty annotations + image tiles (SMB share) |

The active-learning repo is what turns Hasty annotations into HerdNet
training tiles (`021_hasty_to_tile_point_detection.py`).
Use `dataset_configs_*.py` per split.

### Pre-existing project memory

Always check, at the start of any session:

- `/home/christian/.claude/projects/-home-christian-hnee-HerdNet/memory/MEMORY.md` — index
- `.claude/skills/training-insights/PHASE_HISTORY.md` — locked decisions + known-failed ideas
- `CLAUDE.md` (repo root) — project-level instructions and conventions

---

## 2. Master annotation state (as of 2026-06-05 post phase-15 iter-7)

Master file: `/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/2026_05_06_labels.json`
(~89 MB, compact JSON).

| field | count | meaning |
|---|---:|---|
| images | 17,151 | annotated drone tiles |
| boxes (`iguana`) | 20,692 | original iguana bboxes — **never touched** by phase-15 (sacred rule) |
| iguana_point | 75,879 | keypoint annotations — the point-detection target |
| not_iguana_but_similar_look | 1,184 | confirmed hard-negative (vegetation / rock that looks iguana-like) |
| status COMPLETED | 16,232 | reviewed images |

Cumulative phase-15 deltas (pre_iter0 → post_iter7):
**+3,319 iguana_point, +1,176 not_iguana, 0 box changes.**

### Backup chain (rollback-safe)

In the master's directory:

- `2026_05_06_labels.pre_phase15_iter0.bak.json` … `…iter7.bak.json` — one per iteration
- `2026_05_06_labels.pre_iterv2_merge.bak.json` — the snapshot just before
  the external Hasty re-export was merged in

Always backup before any in-place edit. Pattern:

```bash
cd /data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06
cp 2026_05_06_labels.json 2026_05_06_labels.pre_<descriptor>.bak.json
```

### Other-island label-text canonical-ness

Datasets are mixed-case (`Fer_FCD01...` vs `fer_fne01...`). Code that
treats Fernandina specially uses a normalized check — see
`is_fernandina(ds)` in `tools/build_al_v[7-9]_split.py`. Always reuse
that helper rather than re-coding the prefix matching.

---

## 3. Phase-15 active-learning loop — what it is, what it has done

The phase-15 loop is the human-in-the-loop pipeline that turns model
predictions on a held-out val into corrections promoted back to master,
then re-trains. It's documented at `docs/phase15_annotation_cleanup_loop.md`.

### Per-iteration history

| iter | val source | model used | images reviewed | A (real ig.) | B (false GT) | C (relocate) | H (hard-neg) | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | phase-13 val+test | (none — first pass) | 50 | 357 | 0 | — | 0 | hand-rolled CVAT, no H rule yet |
| 1 | same | (none) | 184 | 117 | — | — | 112 | hasty_id round-trip enabled |
| 2 | al_v3 split | al_v3 | 200 | 630 | — | — | 17 | dense colonies, A/(A+H)=97% |
| 3 | al_v4 split | al_v4 | 100 | 55 | — | — | 66 | A/(A+H)=45% |
| 4 | al_v5 split | al_v5 (F5=0.957) | 62 | 6 | 1 | 19 | 63 | A/(A+H)=8.7% |
| 5 | al_v7 split (Esp+Flo) | al_v7 E2 | 319 | 67 | — | 162 | 443 | A/(A+H)=13.1% |
| 6 | al_v8 split | al_v8 E3 | 333 | 43 | 4 | 160 | 321 | A/(A+H)=11.8% |
| 7 | al_v8 split | al_v8 E7 + efficiency knobs | **77** | 19 | — | 43 | 48 | **A/(A+H)=28.4%** ← efficiency knobs work |

### Efficiency knobs landed in iter-7

`tools/human_in_the_loop/100_HIT_phase15_upload.py` now supports:

- `pred_score_threshold=0.50` (was 0.30) — drops low-confidence FPs.
- `ranking_mode="score_sum"` — rank images by sum of pred-only
  confidence, not raw count. High-confidence FPs surface first.
- `min_gt_count=3` — drop sparse images, focus review on dense colonies
  where annotation gaps cluster.

Combined effect: A/(A+H) ratio jumped from ~12 % to **28 %** in iter-7.

Also new in iter-7: **`untouched_pred_is_hard_neg=True`** in the
download flow (`101_HIT_phase15_download.py`). When the reviewer skips
explicit deletions to save time and only re-classes confirmed iguanas
to `iguana_point_gt`, untouched pred_only points get promoted to H
automatically. Iter-7 was the first iteration using this mode (see
`117_HIT_phase15_download_iter7_al_v8_v2.py`).

### Hard-negative-aware data augmentation (option B, landed 2026-06-04)

The 1,184 not_iguana_but_similar_look points in master are now wired
through training:

- `021_hasty_to_tile_point_detection.py` now respects
  `dataset_configs.class_filter` (used to hard-code `["iguana_point"]`);
  if you include `"not_iguana_but_similar_look"` in the filter, those
  points reach the train CSV with `labels=2`.
- `ObjectAwareRandomCrop` gained `hard_negative_probability: float = 0.0`
  and `hard_negative_label: int = 2` parameters. With
  `hard_negative_probability=0.25`, **25 % of crops are anchored on a
  confirmed H point**, forcing the model to look at vegetation it has
  been wrong about.
- `CSVDataset` strips `labels==2` keypoints from the post-augmentation
  output, so the loss target only ever contains iguana_point — H crops
  become "informative empty" tiles.
- The label travels through albumentations v2 as the third element of
  packed keypoint tuples (`(x, y, label_id)`) — NOT as a top-level
  kwarg. (We learned this the hard way; see "B: data-loader bugs" in
  section 7.)
- Tests: `tests/test_hard_negative_aware_crop.py` (5 passing).

### Phase-15 audit trail

Every edit lands in `data/phase15_edit_log.csv` (append-only). Columns:
timestamp, iteration_id, image, site, label_id, pre_class_name,
post_class_name, category (A/B/C/D/E/H/kept_unchanged), fate, x_old,
y_old, x_new, y_new, score, notes.

To audit any past iteration:

```python
import pandas as pd
df = pd.read_csv("/home/christian/hnee/HerdNet/data/phase15_edit_log.csv")
df[df["iteration_id"] == "phase15_iter7_al_v8_v2"]["category"].value_counts()
```

---

## 4. Models trained

All live under `output/al_v*_b4_seed42/...`. Best snapshots:

| model | best epoch | val | recall | precision | F1 | F2 | notes |
|---|---:|---|---:|---:|---:|---:|---|
| al_v3 r3 | 1 | 3,009 (uncorrected) | — | — | 0.714 | — | first post-cleanup |
| al_v4 | 4 | 511 | — | — | — | — | empty_probability=0.20 first run |
| al_v5 | 5 | 192 | 0.97 | 0.78 | — | — | F5=**0.957** (headline number, but small val) |
| al_v6 | 1 (E1 best) | 974 | 0.82 | 0.43 | 0.57 | — | recall regression confirmed at E3, E1 chosen |
| al_v7 | 4 | 610 (Esp+Flo, iter-v2 fresh) | 0.85 | 0.86 | 0.86 | 0.85 | F2 selection, balanced; precision recovered after E2 dip |
| al_v8 | 7 | 1,123 (8 islands, never reviewed) | 0.87 | 0.81 | 0.84 | **0.86** | local 4090, 10 epochs (~17h52m). best model used for iter-7 review |

The best_model.pth for al_v8 E7 is preserved separately at
`output/al_v8_b4_seed42/snapshots/al_v8_e7_best.pth` (the
`best_model.pth` symlink/file in the run dir gets overwritten by each
new-best epoch — once preserved, future runs can't clobber it).

Warm-start anchor: `best_models/phase8/b4_seed42/best_model.pth`.
Every al_v* run uses this. Don't change unless you have a clean reason —
it's the most-validated starting point.

---

## 5. The al_v9 sweep (running NOW on the big machine)

Goal: cleanly answer two design questions at once.

| Axis | levels |
|---|---|
| Dataloader | **old** (no H-anchor) vs **H-aware** (hard_negative_probability=0.25, empty_probability=0.10) |
| Fer in train | **out** (no Fernandina anywhere in train) vs **in** (~162 phase-15-reviewed Fer + FPM01_24012023) |

2×2 = 4 experiments, same val (1,123 imgs / 14 untouched non-Fern
datasets / 8 islands), same warm-start (phase8 b4_seed42), same
hyperparameters (30 epochs, lr=6.4e-4, batch=32, validate_on=f2_score,
adapt_ts=0.20, score_threshold=0.30, seed=42).

| Run | NAME | fer_mode | hnp | ep | label |
|---|---|---|---:|---:|---|
| A | A_old_ferout | fer_out | 0.00 | 0.20 | old_loader |
| B | B_old_ferin  | fer_in  | 0.00 | 0.20 | old_loader |
| C | C_new_ferout | fer_out | 0.25 | 0.10 | hnp25_ep10 |
| D | D_new_ferin  | fer_in  | 0.25 | 0.10 | hnp25_ep10 |

Container image: `dockerkartok/herdnet:al_v9_sweep-latest`
(current digest `sha256:eb2a15eac3716d4727672b5561b500a505d4299891b984db7634d2913a015de1`,
21.5 GB). Built from `docker/Dockerfile.al_v9_sweep` +
`docker/al_v9_sweep_entrypoint.sh`.

Parallel launch on 4 separate GPUs (the canonical run, fired by user
2026-06-05):

```bash
for i in 0 1 2 3; do
  EXP=(A B C D); docker run -d \
    --gpus "\"device=$i\"" --shm-size=16g \
    -e WANDB_API_KEY=$WANDB_API_KEY -e EXPERIMENT=${EXP[$i]} \
    -v $(pwd)/al_v9_output:/app/output \
    --name al_v9_${EXP[$i]} \
    dockerkartok/herdnet:al_v9_sweep-latest
done
```

Each writes its own `al_v9_sweep_summary_<X>.tsv`, and one wandb run
per experiment (project `hn_active_learning`, tags include
`fer_in`/`fer_out` + `old_loader`/`hnp25_ep10` + `sweep_2x2`).
Expected wall-clock: **~15 h** parallel, vs ~60 h sequential.

### Outputs expected (when done)

For each experiment:

- `output/al_v9_<NAME>/<date>/<time>/best_model.pth` — F2-selected snapshot
- `output/al_v9_<NAME>/<date>/<time>/20260605_training.log` — full training log
- `output/al_v9_sweep_summary_<X>.tsv` — one-line per experiment summary
- wandb run (artifact upload at the end via `upload_model.py`)

### How to read the results

Once all 4 land:

```python
import pandas as pd
parts = [pd.read_csv(f"al_v9_output/al_v9_sweep_summary_{x}.tsv", sep="\t")
         for x in "ABCD"]
sweep = pd.concat(parts).reset_index(drop=True)
print(sweep)
```

Clean 2×2 attribution:

- **Dataloader effect** = (C−A + D−B) / 2 on F2
- **Fer effect** = (B−A + D−C) / 2 on F2

If both are positive, the winner is D (H-aware + Fer in). If H-aware
helps but Fer hurts (e.g., because Fer datasets have residual annotation
noise), best is C. The relative magnitudes tell us which lever to push
harder in al_v10.

---

## 6. Tools & infrastructure (everything you might need)

### Data prep + splits

| script | what |
|---|---|
| `tools/build_al_v8_split.py` | most recent active-learning split (al_v8 val was used for iter-6 and iter-7) |
| `tools/build_al_v9_split.py` | a/v9 split: `--fern-mode in/out` |
| `active-learning/scripts/training_data_preparation/021_hasty_to_tile_point_detection.py` | Hasty → tiled training CSV |
| `dataset_configs_al_v9_fer_{in,out}_2026_06_04.py` | class_filter incl. not_iguana_but_similar_look (label_id=2) |

### Phase-15 pipeline

| script | what |
|---|---|
| `tools/human_in_the_loop/100_HIT_phase15_upload.py` | generic upload (Hungarian merge → CVAT) |
| `tools/human_in_the_loop/101_HIT_phase15_download.py` | generic download + master update |
| `tools/human_in_the_loop/hit_phase15_merge.py` | classification rules (A/B/C/D/E/H) |
| `tools/human_in_the_loop/ensemble_detections.py` | NEW: two-model consensus filter (option 4) |
| `tools/human_in_the_loop/1{0[2-9],1[0-7]}_HIT_phase15_*.py` | per-iteration launchers (iter-0 through iter-7) |

### Training

| script | what |
|---|---|
| `tools/train.py` | Hydra entrypoint (`python tools/train.py --config-path … --config-name …`) |
| `configs/demo/al_v8_balanced.yaml` | last-trained recipe (B4 ConvNeXtV2 tiny, validate_on=f2_score) |
| `docker/Dockerfile.al_v9_sweep` + `al_v9_sweep_entrypoint.sh` | 2×2 sweep packaged |
| `docker/build_al_v9_sweep.sh` | builds the sweep image |
| `tools/best_runs.py` | rank all training runs by F-score |

### Inference

```bash
python tools/infer.py \
  --config-dir /home/christian/hnee/HerdNet/configs/demo \
  --config-name <config> \
  --images <dir-of-tiles> \
  --model <path/to/best_model.pth> \
  --output <output-dir> \
  --evaluate \
  --overrides datasets.test.csv_file=... datasets.test.root_dir=...
```

For tiled large-image inference use `animaloc.eval.HerdNetStitcher`
(documented in `.claude/ARCHITECTURE.md`).

### Docker images (private registry `dockerkartok/herdnet`)

| tag | what | digest |
|---|---|---|
| `al_v7-latest` | tiny backbone, al_v7 val, single training | 2f63a98352026914... |
| `al_v7_base-latest` | ConvNeXtV2-base backbone, ImageNet warm-start | 5952ec70f2784c5e... |
| `al_v4_emp_sweep-latest` | empty_probability sweep (legacy) | (older) |
| `al_v9_sweep-latest` | 2×2 sweep, EXPERIMENT={A,B,C,D,all} | eb2a15eac3716d47... |

Push pattern (when retrying intermittent Docker Hub 502s):

```bash
docker push dockerkartok/herdnet:<tag>  # just re-run; cached layers skip
docker manifest inspect dockerkartok/herdnet:<tag>  # verify
```

---

## 7. Known traps and lessons learned (don't repeat these)

### A. Master integrity rules

- **Boxes are sacred.** Phase-15 corrects POINTS only. The 20,692
  iguana bboxes have never changed across 7 iterations and must not.
  Verify in any new merge/promotion step.
- **`is_fernandina(ds)`** has multiple prefix patterns and a hand-typed
  set of exceptions. Use the helper, never re-implement.
- Master is on a SMB share — **symlinks fail with I/O error**. Use
  rsync or copy.

### B. Data-loader bugs we already hit

- `ObjectAwareRandomCrop` initially listed `'labels'` in
  `targets_as_params`. Albumentations v2 packs `label_fields` INTO the
  keypoint tuples and strips them from kwargs — listing `'labels'` causes
  `ValueError: ObjectAwareRandomCrop requires ['image', 'keypoints',
  'labels'] missing keys: {'labels'}`. The fix: read labels from
  `keypoint[2]` (extended tuple). Keep `targets_as_params = ['image',
  'keypoints']`.
- The container ALWAYS runs the sweep entrypoint by default. To use
  `docker run --rm <image> python -c ...` you must add
  `--entrypoint python`. Otherwise the entrypoint hijacks the command.

### C. Eval threshold quirks

- The training-time `score_threshold=0.30` aggressively trims the
  detection tail. At threshold=0.00 the same model can have **+10 %
  recall** but much lower precision. For active-learning review
  (efficiency knobs above), 0.50 is the new default.
- The `[TEST] [N/N]` line in inference logs reports **last-image**
  metrics, NOT aggregates. Pull the `species precision recall …`
  block instead — it's the cumulative summary.

### D. Push reliability

Docker Hub's registry occasionally serves a 502 on a specific blob.
Push-retry usually works on the second attempt. If a push reports
`exit 0` but the manifest isn't queryable, the push failed silently —
retry until `docker manifest inspect` returns proper JSON.

### E. Validation re-prep

When the master changes (every iteration), val GT is stale unless you
re-prep. Both train and val should always be re-prepped from the
LATEST master — see al_v9's split-then-data-prep flow.

---

## 8. The actual goal: a Galapagos-wide population census

This is what the next session is meant to scope and build. We
deliberately did NOT do it yet — we needed first to be confident in the
detector. As of post-iter-7 with al_v8 E7 (R=0.87, P=0.81, F2=0.86 on
the 1,123-img multi-island val), we now have a model that is
**usable but not yet production-grade**. The al_v9 sweep should decide
whether al_v9 D (or whichever wins) is the production model or whether
we need another loop.

### What "full census" means concretely

For each island / sub-region (Espanola, Floreana, Fernandina, Isabela,
Santa Cruz, San Cristobal, Pinta, Marchena, Genovesa, Wolf, etc.):

1. Take the FULL set of drone images (not just the curated training/val
   subset) — these live under
   `/data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/unzipped_images_rsync/`
   (313 dataset dirs, ~11.8 GB just for the iter-v2 additions).
2. Run the best model over every image, tiled with `HerdNetStitcher`
   (overlap=120 px, down_ratio=4 — `animaloc/eval/stitcher.py`).
3. **Dedupe predictions** across overlapping tiles. The current
   stitcher does this within one image; cross-image dedupe (e.g.
   neighboring overhead flights of the same site) is NOT in scope yet.
4. Apply `score_threshold` calibration: there's a precision-recall
   trade-off knob — for census-grade counting we probably want
   precision ≥ 0.9 (i.e. lean toward under-counting rather than
   over-counting, since population estimates are typically reported
   with confidence intervals).
5. Aggregate counts per site / per island / per date. Cross-check with
   transect-survey ground truth where available.

### Concrete open questions for the next session

These are the things the new session should decide and build:

1. **Model choice.** When the al_v9 sweep ends, pick the winning model
   variant. If hard-negative-aware loader (C or D) wins by ≥0.005 on
   F2, use it. Else fall back to al_v8 E7 which is the safest known
   production-grade snapshot.
2. **Calibrated `score_threshold` for census.** Sweep [0.30, 0.40,
   0.50, 0.60, 0.70] on the al_v9 val, plot the precision-recall
   trade-off, pick the threshold that yields P ≥ 0.9 with maximum R.
3. **Inference at scale.** Write a script that walks
   `unzipped_images_rsync/`, runs tiled inference on every image,
   writes per-image detection JSONs. Estimate: ~17,000 images × ~0.5 s/img
   on 80 GB GPU ≈ 2.5 h. Cheap.
4. **Sanity-check on ground-truth subsets.** The 1,123 val images are
   our only fully-clean-label ground truth. Run the chosen model at
   the chosen threshold on them, report the implied population
   undercount/overcount factor. That number IS the census calibration
   constant.
5. **Cross-image dedup** (harder). When the same physical iguana is
   visible in two overlapping flight photos, the model double-counts.
   Options: image-mosaic-first then detect; or GPS-stitch the detections
   and merge within an "iguana_radius" in world coordinates. Mosaicing
   is in `metashape-mosaicing` env (separate workflow); the GPS-stitch
   option is lighter but needs flight metadata.
6. **Reporting layer.** TSV / GeoJSON / fiftyone dataset of detections
   per island per date, ready to hand off to the biology team.

### Suggested first 90 minutes of the census session

```text
Step 1 — verify al_v9 sweep done. Read summary tsvs.
Step 2 — pick winning model. If H-aware wins, copy that best_model.pth
         to /home/christian/hnee/HerdNet/best_models/census_v1/best_model.pth.
Step 3 — threshold sweep (script doesn't exist yet — needs writing).
         Pick the P≥0.9 threshold.
Step 4 — write inference-at-scale script that targets
         unzipped_images_rsync/. Single-GPU, batched, output:
         per-image JSON of (x, y, score) detections.
Step 5 — aggregate to per-site / per-island counts. CSV out.
Step 6 — calibrate: run on the al_v9 val set, compute correction factor
         = #GT / #detections-passing-threshold. Apply across all sites.
```

The output of step 5+6 IS the population census, modulo cross-image
dedup (step 5b, if needed for the user's research framing).

---

## 9. People + access

- User: Christian Winkelmann (`christian.winkelmann@gmail.com`).
- CVAT instance: `https://app.cvat.ai`, org `IguanasFromAbove`, project
  `Hasty_Corr`. Credentials in `/home/christian/hnee/active-learning/.env`.
- Docker registry: `dockerkartok/herdnet` is **private**. Authenticated
  pulls only.
- W&B project: `karisu/hn_active_learning`.

---

## 10. File pointers (everything important in one place)

```text
Master + backups       /data/mnt/storage/Iguanas_From_Above/training_data/2026_05_06/
                       2026_05_06_labels.json (current)
                       2026_05_06_labels.pre_phase15_iter*.bak.json

Training data          /home/christian/data/training_data/
                       2026_06_04_al_v9_canonical/al_v9_fer_{in,out}_2026_06_04/
                         {train,val}/{Default,herdnet_format.csv}

Best model (production candidate, al_v8 E7)
                       /home/christian/hnee/HerdNet/output/al_v8_b4_seed42/
                         snapshots/al_v8_e7_best.pth

Phase-15 audit log     /home/christian/hnee/HerdNet/data/phase15_edit_log.csv

Phase-15 correction stats report
                       /data/mnt/storage/Iguanas_From_Above/database/
                         correction_factor/phase15_correction_loop_stats.md

al_v9 sweep image      dockerkartok/herdnet:al_v9_sweep-latest

Config (last trained)  /home/christian/hnee/HerdNet/configs/demo/al_v8_balanced.yaml

This doc               /home/christian/hnee/HerdNet/docs/handoff_population_census.md
```

End of handoff.
