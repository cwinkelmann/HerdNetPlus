# Phase-13 training container

A self-contained Docker image that bundles **code + warm-start checkpoint + one training subset + the fixed val/test sets**, runs a single Phase-13 fine-tune, and uploads the resulting best model as a wandb artifact.

Intended for remote GPU machines (RunPod, AWS EC2 g4/g5, etc.) where you push the image, run one container, and walk away.

## What's inside

| Path inside image | Source | Approx. size |
|---|---|---|
| `/app/animaloc/`, `/app/configs/`, `/app/tools/` | Repo | ~30 MB |
| `/app/best_models/phase8/b4_seed42/best_model.pth` | The Phase-8 production warm-start | ~400 MB |
| `/app/data/train/` | One `train_N<N>_s<seed>/` subset (chosen at build time) | varies (10 MB → 4 GB) |
| `/app/data/val/` | Fixed validation set (Phase 13) | ~125 MB |
| `/app/data/test/` | Fixed test set (Phase 13) | ~1.1 GB |
| `/app/docker/train_entrypoint.sh` | The entry script | ~3 KB |
| Base image (`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`) | DockerHub | ~6 GB |

Total image size for `TRAIN_N=152`: roughly **8 GB**. For `TRAIN_N=full` (~6900 training frames): roughly **20 GB**.

If you'd rather not bake the data into the image, mount it instead — see "Alternative: mount data" below.

## Build

```bash
# bundle N=152 (smallest practical size, ~8 GB image)
bash docker/build.sh

# bundle a bigger N
TRAIN_N=608 bash docker/build.sh

# custom image tag
TRAIN_N=304 IMAGE_TAG=ifa-phase13:rp1 bash docker/build.sh
```

The build script stages everything into a temp directory so the `docker build` context doesn't include the entire repo (with its 100+ GB `output/`). It cleans up the tempdir after.

## Push to Docker Hub (`dockerkartok/herdnet`)

Images go to the `dockerkartok/herdnet` repository on Docker Hub, tagged by the training subset size so multiple bundles can coexist.

```bash
# 1. One-time login (use a personal access token, not your password)
docker login -u dockerkartok      # paste token when prompted

# 2. Build with the canonical local tag (e.g. herdnet-phase13-nfull:latest)
TRAIN_N=full bash docker/build.sh

# 3. Re-tag for Docker Hub. Use both a dated tag and `latest` for the most recent.
DATE=$(date +%Y-%m-%d)
docker tag herdnet-phase13-nfull:latest dockerkartok/herdnet:phase13-nfull-${DATE}
docker tag herdnet-phase13-nfull:latest dockerkartok/herdnet:phase13-nfull-latest

# 4. Push both tags
docker push dockerkartok/herdnet:phase13-nfull-${DATE}
docker push dockerkartok/herdnet:phase13-nfull-latest
```

Tag scheme:
- `phase13-n<N>-<YYYY-MM-DD>` — immutable, dated snapshot. Cite this in experiment notes.
- `phase13-n<N>-latest` — moving pointer to the newest build for that N.

Push size: ~10 GB compressed for `TRAIN_N=full`. Docker Hub free accounts have unlimited public storage but rate-limited pulls — fine for occasional remote-GPU work.

## Pull and run on a remote host

```bash
# Pull (uses Docker Hub credentials; `docker login` once if it's a private repo)
docker pull dockerkartok/herdnet:phase13-nfull-latest

# Run a full training, streaming per-epoch metrics + final model artifact to wandb
docker run --rm --gpus all --shm-size=8g \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e WANDB_PROJECT=hn_phase13_data_scaling \
  -e TRAIN_N=full \
  -e AUG_MULT=1 \
  -e BATCH_SIZE=8 \
  -e NUM_WORKERS=16 \
  -v $(pwd)/output:/app/output \
  dockerkartok/herdnet:phase13-nfull-latest
```

## Run

Single training, with wandb integration:

```bash
docker run --rm --gpus all --shm-size=8g \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e WANDB_PROJECT=hn_phase13_data_scaling \
  -e TRAIN_N=152 \
  herdnet-phase13-N152:latest
```

To also pull the trained model out as a file on the host:

```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/app/output \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e TRAIN_N=152 \
  herdnet-phase13-N152:latest
```

Best model lands at `output/phase13_N152_s42_docker/<date>/<time>/best_model.pth` on the host.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `WANDB_API_KEY` | unset | **Required** for wandb integration (per-epoch metrics + end-of-training model upload). If unset, wandb runs in offline mode and the artifact upload is skipped |
| `WANDB_PROJECT` | `hn_phase13_data_scaling` | wandb project for per-epoch metrics and the final model artifact |
| `WANDB_FLAG` | `True` | Set `False` to disable wandb entirely (no per-epoch curves, no artifact upload) |
| `TRAIN_N` | `152` (build-time default; used for naming only) | Training-set tag; data is already baked in |
| `SEED` | `42` | Random seed |
| `AUG_MULT` | unset (config default 75) | Override `augmentation_multiplier`; set to `1` for big-N runs to save compute |
| `BATCH_SIZE` | unset (config default 4) | Override `training_settings.batch_size`; raise on big-VRAM GPUs (e.g. A100/H100) to speed up training |
| `NUM_WORKERS` | unset (config default 8) | Override `training_settings.num_workers` (DataLoader workers); raise on many-core hosts, lower if you hit memory pressure |
| `UPLOAD_MODEL` | `1` | Set `0` to skip the post-training wandb artifact upload |
| `ARTIFACT_NAME` | `phase13_best_model` | wandb artifact name |

## Alternative: mount data instead of baking it in

If you want a leaner image (~2 GB plus PyTorch base), build with a stub training subset (e.g. `TRAIN_N=19`) and then mount the real data at runtime:

```bash
# build once with the smallest subset just so the image structure is valid
TRAIN_N=19 bash docker/build.sh

# then run with the real data mounted, overriding /app/data
docker run --rm --gpus all \
  -v /path/to/2026_05_08_data_scaling/train_N608_s42:/app/data/train \
  -v /path/to/2026_05_08_data_scaling/val:/app/data/val \
  -v /path/to/2026_05_08_data_scaling/test:/app/data/test \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e TRAIN_N=608 \
  -e AUG_MULT=1 \
  herdnet-phase13-N19:latest
```

The volume mounts mask `/app/data/{train,val,test}` from the baked image, so the running container uses the host data instead. Same image, any N.

## 80 GB GPU tuning (A100 / H100)

The phase-13 production recipe was tuned on a 16 GB GPU. On an 80 GB
device (A100-80GB, H100-80GB) you can run **~8× the throughput**
without touching the code — just bigger batches + more workers.

### Single-GPU 80 GB recipe

```bash
docker run --rm --gpus all --shm-size=16g \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e WANDB_PROJECT=hn_phase13_data_scaling \
  -e TRAIN_N=full \
  -e BATCH_SIZE=32 \
  -e NUM_WORKERS=24 \
  -e AUG_MULT=75 \
  dockerkartok/herdnet:phase13-nfull-latest
```

Knob walkthrough:

| Knob | 16 GB default | 80 GB tuned | Why |
|---|---|---|---|
| `BATCH_SIZE` | 4 | **32** (try 64 if free VRAM remains) | B4 ConvNeXt at 512×512 + FIDT mask uses ~10 GB at batch=4; scales roughly linearly. 32 fits comfortably with headroom for the validate stitcher pass |
| `NUM_WORKERS` | 8 | **24** | 8× batch needs more dataloader throughput. Cap at `min(24, num_cpu_cores)` |
| `--shm-size` | `8g` | **`16g`** | Bigger batches × more workers = more shared-memory traffic |
| Learning rate | 8e-5 (config) | **6.4e-4** if you keep epochs constant | Linear scaling rule for 8× batch. Or keep `lr=8e-5` and double `warmup_iters` to 1000 for safety |
| `AUG_MULT` | 75 | **75** (unchanged) | Phase 14 ruled out lower values for production convergence |

Set `lr` via a hydra override appended to the entrypoint command if you want to override the config default — but most users get away with the config default + bigger batch, since AdamW handles modest LR mismatches.

Expected speedup vs 16 GB:
- Training step ~6-8× faster (limited by data-loader throughput, not compute)
- Validation pass ~8× faster (batch=32 vs 4 through the stitcher)
- Full N=full run: phase-13 took ~3 days on 16 GB; expect **~6-10 h** on a single 80 GB GPU

### Multi-GPU

Not supported in this image yet. The trainer at
`animaloc/train/trainers.py:170` does a single `model.to(self.device)`
with no DDP wrapping. Adding it is **~30 lines of
DistributedDataParallel boilerplate** per `docs/refactor.md` Tier 2 —
deliberately not Lightning. Cheaper interim if you're stuck on a
multi-GPU box: keep `BATCH_SIZE=32` but raise effective batch via
gradient accumulation (~5 extra lines in `trainers.py`).

When the dataset grows past ~1000 frames *and* a single 80 GB GPU
saturates wall-clock budget, then it's time to do the DDP patch. Until
then, single-GPU 80 GB is the sweet spot.

## What gets uploaded to wandb

**Per-epoch metric curves stream live during training (same observability you have locally). The model checkpoint is the only thing held for the end — uploaded once, to save wandb storage.**

Step by step:

1. Training pushes per-epoch metrics to wandb as usual: losses, validation F1 / MAE / precision / recall, learning rate, etc. The run is named `phase13_N<N>_s<seed>_docker` in `WANDB_PROJECT`, tagged `[phase13, data_scaling, docker, N<N>]`.
2. After training finishes, `upload_model.py` reads the wandb run ID from `<OUT_DIR>/wandb/run-*` and **resumes the same run** (`wandb.init(id=..., resume="must")`) so the artifact lands on the row with the per-epoch curves — one row per training, not two.
3. Attaches `best_model.pth` as a wandb `Artifact` named `phase13_best_model` with metadata `{train_n, seed, best_f1, mae, best_val, epochs, model_file_size_mb}` parsed from the final `SUMMARY` log line.
4. Refreshes `run.summary` so the headline metrics show in the run header.
5. Calls `run.finish()`.

**Storage cost**: one ~400 MB artifact per training. Per-epoch scalar metrics are tiny.

**Fallback**: if the wandb run ID can't be recovered from the training cache (e.g. training was forced offline), the artifact uploads to a sidecar run named `<run>_model` instead. You'll see two rows in that case — flagged with the `model_artifact_sidecar` tag.

Set `UPLOAD_MODEL=0` to skip the artifact upload (training metrics still stream as normal).

## Troubleshooting

**`RuntimeError: DataLoader worker (pid N) ... unable to write to /dev/shm` / "out of shared memory"**

Docker's default `/dev/shm` is 64 MB, but PyTorch's multi-worker DataLoader uses shared memory to pass tensors from worker processes to the main process. With `NUM_WORKERS>0` and reasonably-sized batches this fills up instantly.

Pick one:
- `--shm-size=8g` (or higher) — per-container, isolated. Recommended.
- `--ipc=host` — container shares `/dev/shm` with the host. Effectively unlimited but less isolated.
- `NUM_WORKERS=0` — single-process loading, no shared memory needed but much slower.

The example commands in this README already include `--shm-size=8g`; if you copy-paste your own command, don't drop it.

## Caveats

- **Image size**: the data layer is the biggest contributor. `TRAIN_N=152` → ~8 GB; `TRAIN_N=full` → ~20 GB. Push to a registry that handles large images well (DockerHub Pro, AWS ECR, etc.) or use the volume-mount route.
- **PyTorch base image** is `~6 GB` by itself. If you need it smaller, switch to the slim variant in `Dockerfile.train` — but you'll need to install PyTorch + CUDA bits yourself, which is fiddly.
- **GPU required**: `--gpus all` is mandatory. The base image expects an NVIDIA GPU and CUDA driver on the host.
- **wandb storage**: each artifact is ~400 MB; if you run many of these, mind the storage quota on your wandb account.
- **Reproducibility**: image bundles `pyproject.toml` but the exact CUDA / cuDNN versions come from the PyTorch base tag. Pin to a specific tag if you want bit-exact reproducibility months later.
