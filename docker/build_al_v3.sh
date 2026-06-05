#!/bin/bash
# Build the active-learning iter-3 training container.
#
# Stages code + warm-start + al_v3 train + al_v3 val into a build
# context, then runs docker build with Dockerfile.al_v3.
#
# Usage:
#   bash docker/build_al_v3.sh
#   IMAGE_TAG=my-tag bash docker/build_al_v3.sh
#
# Environment variables:
#   AL_V3_ROOT  : path to the al_v3 dataset root
#                 (default: /home/christian/data/training_data/2026_06_01_al_v3_canonical/al_v3_2026_06_01)
#   IMAGE_TAG   : docker tag (default herdnet-al-v3:latest)

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

AL_V3_ROOT="${AL_V3_ROOT:-/home/christian/data/training_data/2026_06_01_al_v3_canonical/al_v3_2026_06_01}"
IMAGE_TAG="${IMAGE_TAG:-herdnet-al-v3:latest}"

TRAIN_DIR="${AL_V3_ROOT}/train"
VAL_DIR="${AL_V3_ROOT}/val"
WARM_START="${REPO_DIR}/best_models/phase8/b4_seed42/best_model.pth"

for p in "$TRAIN_DIR" "$VAL_DIR"; do
  [ -d "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done
[ -f "$TRAIN_DIR/herdnet_format.csv" ] || { echo "ERROR: missing $TRAIN_DIR/herdnet_format.csv"; exit 1; }
[ -f "$VAL_DIR/herdnet_format.csv"   ] || { echo "ERROR: missing $VAL_DIR/herdnet_format.csv"; exit 1; }
[ -e "$WARM_START" ] || { echo "ERROR: missing warm-start $WARM_START"; exit 1; }
WARM_START_REAL=$(readlink -f "$WARM_START")
[ -f "$WARM_START_REAL" ] || { echo "ERROR: $WARM_START is not a real file (resolved: $WARM_START_REAL)"; exit 1; }

TRAIN_TILES=$(find "$TRAIN_DIR/Default" -maxdepth 1 -type f -o -type l | wc -l)
VAL_TILES=$(find   "$VAL_DIR/Default"   -maxdepth 1 -type f -o -type l | wc -l)

echo "============================================================"
echo "  Building al_v3 training image"
echo "  AL_V3_ROOT: $AL_V3_ROOT"
echo "  Train:      $TRAIN_TILES tiles in $TRAIN_DIR/Default"
echo "  Val:        $VAL_TILES tiles in $VAL_DIR/Default"
echo "  Warm-start: $WARM_START_REAL ($(du -h "$WARM_START_REAL" | cut -f1))"
echo "  Image tag:  $IMAGE_TAG"
echo "============================================================"

STAGE=$(mktemp -d -t al_v3_docker_ctx_XXXXXX)
trap "echo 'cleaning up $STAGE'; rm -rf $STAGE" EXIT
echo "Staging build context at $STAGE ..."

mkdir -p "$STAGE"
cp pyproject.toml "$STAGE/"
[ -f README.md ] && cp README.md "$STAGE/"
cp -r animaloc "$STAGE/"
cp -r configs  "$STAGE/"
rsync -a \
  --exclude='outputs'        \
  --exclude='wandb'          \
  --exclude='__pycache__'    \
  --exclude='*.pyc'          \
  --exclude='*.log'          \
  --exclude='*.pth'          \
  --exclude='detections.csv' \
  --exclude='metrics_results.csv' \
  --exclude='plots'          \
  tools/ "$STAGE/tools/"

find "$STAGE" -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ipynb_checkpoints" \) \
  -exec rm -rf {} + 2>/dev/null || true

mkdir -p "$STAGE/best_models/phase8/b4_seed42"
cp "$WARM_START_REAL" "$STAGE/best_models/phase8/b4_seed42/best_model.pth"

# Stage al_v3 train + val. Dereference any symlinks (Default/ is full of
# symlinks back to unzipped_images_rsync/).
mkdir -p "$STAGE/data"
echo "  copying al_v3 train ($TRAIN_TILES tiles)..."
cp -rL "$TRAIN_DIR" "$STAGE/data/train"
echo "  copying al_v3 val   ($VAL_TILES tiles)..."
cp -rL "$VAL_DIR"   "$STAGE/data/val"

mkdir -p "$STAGE/docker"
cp docker/Dockerfile.al_v3      "$STAGE/docker/"
cp docker/al_v3_entrypoint.sh   "$STAGE/docker/"
cp docker/upload_model.py       "$STAGE/docker/"

echo ""
echo "Build context size: $(du -sh "$STAGE" | cut -f1)"
echo ""
echo "Running docker build ..."
docker build \
  --tag "$IMAGE_TAG" \
  --file "$STAGE/docker/Dockerfile.al_v3" \
  "$STAGE"

echo ""
echo "============================================================"
echo "  Built: $IMAGE_TAG"
echo "============================================================"
echo ""
echo "Run example (single 80 GB GPU):"
echo ""
echo "  docker run --rm --gpus all --shm-size=16g \\"
echo "    -e WANDB_API_KEY=\$WANDB_API_KEY \\"
echo "    $IMAGE_TAG"
echo ""
