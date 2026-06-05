#!/bin/bash
# Build the al_v7 BASE-backbone training container (post iter-v2 merge).
#
# Stages al_v7 train + val into the build context (no warm-start file —
# base uses ImageNet pretrained backbone only), then docker build with
# Dockerfile.al_v7_base.
#
# Usage:
#   bash docker/build_al_v7_base.sh
#   IMAGE_TAG=my-registry/herdnet-al-v7-base:tag bash docker/build_al_v7_base.sh

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

AL_V7_ROOT="${AL_V7_ROOT:-/home/christian/data/training_data/2026_06_03_al_v7_canonical/al_v7_2026_06_03}"
IMAGE_TAG="${IMAGE_TAG:-herdnet-al-v7-base:latest}"

TRAIN_DIR="${AL_V7_ROOT}/train"
VAL_DIR="${AL_V7_ROOT}/val"

for p in "$TRAIN_DIR" "$VAL_DIR"; do
    [ -d "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done
[ -f "$TRAIN_DIR/herdnet_format.csv" ] || { echo "ERROR: missing $TRAIN_DIR/herdnet_format.csv"; exit 1; }
[ -f "$VAL_DIR/herdnet_format.csv"   ] || { echo "ERROR: missing $VAL_DIR/herdnet_format.csv"; exit 1; }

TRAIN_TILES=$(find "$TRAIN_DIR/Default" -maxdepth 1 -type f -o -type l | wc -l)
VAL_TILES=$(find   "$VAL_DIR/Default"   -maxdepth 1 -type f -o -type l | wc -l)

echo "============================================================"
echo "  Building al_v7 BASE training image"
echo "  Train:      $TRAIN_TILES tiles"
echo "  Val:        $VAL_TILES tiles"
echo "  Image tag:  $IMAGE_TAG"
echo "============================================================"

STAGE=$(mktemp -d -t al_v7_base_ctx_XXXXXX)
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

mkdir -p "$STAGE/data"
echo "  copying al_v7 train ($TRAIN_TILES tiles) ..."
cp -rL "$TRAIN_DIR" "$STAGE/data/train"
echo "  copying al_v7 val   ($VAL_TILES tiles) ..."
cp -rL "$VAL_DIR"   "$STAGE/data/val"

mkdir -p "$STAGE/docker"
cp docker/Dockerfile.al_v7_base         "$STAGE/docker/"
cp docker/al_v7_base_entrypoint.sh      "$STAGE/docker/"
cp docker/upload_model.py               "$STAGE/docker/"

echo ""
echo "Build context size: $(du -sh "$STAGE" | cut -f1)"
echo ""
docker build \
  --tag "$IMAGE_TAG" \
  --file "$STAGE/docker/Dockerfile.al_v7_base" \
  "$STAGE"

echo ""
echo "============================================================"
echo "  Built: $IMAGE_TAG"
echo "============================================================"
echo ""
echo "Run example (80 GB GPU, base backbone at batch 16):"
echo ""
echo "  docker run --rm --gpus all --shm-size=16g \\"
echo "    -e WANDB_API_KEY=\$WANDB_API_KEY \\"
echo "    -v \$(pwd)/output:/app/output \\"
echo "    $IMAGE_TAG"
echo ""
echo "Push to registry:"
echo ""
echo "  docker tag $IMAGE_TAG dockerkartok/herdnet:al_v7_base-latest"
echo "  docker push dockerkartok/herdnet:al_v7_base-latest"
echo ""
