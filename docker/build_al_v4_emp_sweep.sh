#!/bin/bash
# Build the al_v4 empty_probability sweep container.
#
# Stages al_v4 train + val + warm-start into the build context,
# then docker build with Dockerfile.al_v4_emp_sweep.
#
# Usage:
#   bash docker/build_al_v4_emp_sweep.sh
#   AL_V4_ROOT=/path/to/al_v4 bash docker/build_al_v4_emp_sweep.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

AL_V4_ROOT="${AL_V4_ROOT:-/home/christian/data/training_data/2026_06_02_al_v4_canonical/al_v4_2026_06_02}"
IMAGE_TAG="${IMAGE_TAG:-herdnet-al-v4-emp-sweep:latest}"

TRAIN_DIR="${AL_V4_ROOT}/train"
VAL_DIR="${AL_V4_ROOT}/val"
WARM_START="${REPO_DIR}/best_models/phase8/b4_seed42/best_model.pth"

for p in "$TRAIN_DIR" "$VAL_DIR"; do
    [ -d "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done
[ -f "$TRAIN_DIR/herdnet_format.csv" ] || { echo "ERROR: missing $TRAIN_DIR/herdnet_format.csv"; exit 1; }
[ -f "$VAL_DIR/herdnet_format.csv"   ] || { echo "ERROR: missing $VAL_DIR/herdnet_format.csv"; exit 1; }
[ -e "$WARM_START" ] || { echo "ERROR: missing warm-start $WARM_START"; exit 1; }
WARM_START_REAL=$(readlink -f "$WARM_START")
[ -f "$WARM_START_REAL" ] || { echo "ERROR: warm-start not a real file"; exit 1; }

TRAIN_TILES=$(find "$TRAIN_DIR/Default" -maxdepth 1 -type f -o -type l | wc -l)
VAL_TILES=$(find   "$VAL_DIR/Default"   -maxdepth 1 -type f -o -type l | wc -l)

echo "============================================================"
echo "  Building al_v4 empty_probability sweep image"
echo "  Train:      $TRAIN_TILES tiles"
echo "  Val:        $VAL_TILES tiles"
echo "  Image tag:  $IMAGE_TAG"
echo "============================================================"

STAGE=$(mktemp -d -t al_v4_emp_ctx_XXXXXX)
trap "echo 'cleaning up $STAGE'; rm -rf $STAGE" EXIT
echo "Staging build context at $STAGE ..."

mkdir -p "$STAGE"
cp pyproject.toml "$STAGE/"
[ -f README.md ] && cp README.md "$STAGE/"
cp -r animaloc "$STAGE/"
cp -r configs  "$STAGE/"
rsync -a \
  --exclude='outputs' --exclude='wandb' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='*.log' --exclude='*.pth' \
  --exclude='detections.csv' --exclude='metrics_results.csv' --exclude='plots' \
  tools/ "$STAGE/tools/"

find "$STAGE" -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ipynb_checkpoints" \) \
  -exec rm -rf {} + 2>/dev/null || true

mkdir -p "$STAGE/best_models/phase8/b4_seed42"
cp "$WARM_START_REAL" "$STAGE/best_models/phase8/b4_seed42/best_model.pth"

mkdir -p "$STAGE/data"
echo "  copying al_v4 train ($TRAIN_TILES tiles) ..."
cp -rL "$TRAIN_DIR" "$STAGE/data/train"
echo "  copying al_v4 val   ($VAL_TILES tiles) ..."
cp -rL "$VAL_DIR"   "$STAGE/data/val"

mkdir -p "$STAGE/docker"
cp docker/Dockerfile.al_v4_emp_sweep      "$STAGE/docker/"
cp docker/al_v4_emp_sweep_entrypoint.sh   "$STAGE/docker/"
cp docker/upload_model.py                 "$STAGE/docker/"

echo ""
echo "Build context size: $(du -sh "$STAGE" | cut -f1)"
echo ""
docker build \
  --tag "$IMAGE_TAG" \
  --file "$STAGE/docker/Dockerfile.al_v4_emp_sweep" \
  "$STAGE"

echo ""
echo "============================================================"
echo "  Built: $IMAGE_TAG"
echo "============================================================"
echo ""
echo "Run example (80 GB GPU, sweeps 5 empty_probability values sequentially):"
echo ""
echo "  docker run --rm --gpus all --shm-size=16g \\"
echo "    -e WANDB_API_KEY=\$WANDB_API_KEY \\"
echo "    -e EMP_VALUES=\"0.05,0.10,0.20,0.30,0.50\" \\"
echo "    -v \$(pwd)/output:/app/output \\"
echo "    $IMAGE_TAG"
echo ""
