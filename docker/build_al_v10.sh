#!/bin/bash
# Build the al_v10 training image (post phase-15 iter-7).

set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

AL_V10_ROOT="${AL_V10_ROOT:-/home/christian/data/training_data/2026_06_08_al_v10_canonical/al_v10_2026_06_08}"
IMAGE_TAG="${IMAGE_TAG:-herdnet-al-v10:latest}"

TRAIN_DIR="${AL_V10_ROOT}/train"
VAL_DIR="${AL_V10_ROOT}/val"
WARM_START="${REPO_DIR}/best_models/phase8/b4_seed42/best_model.pth"

for p in "$TRAIN_DIR" "$VAL_DIR"; do
    [ -d "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done
[ -f "$TRAIN_DIR/herdnet_format.csv" ] || { echo "ERROR: missing train csv"; exit 1; }
[ -f "$VAL_DIR/herdnet_format.csv"   ] || { echo "ERROR: missing val csv"; exit 1; }
[ -e "$WARM_START" ] || { echo "ERROR: missing warm-start $WARM_START"; exit 1; }
WARM_START_REAL=$(readlink -f "$WARM_START")

TRAIN_TILES=$(find "$TRAIN_DIR/Default" -maxdepth 1 \( -type f -o -type l \) | wc -l)
VAL_TILES=$(find   "$VAL_DIR/Default"   -maxdepth 1 \( -type f -o -type l \) | wc -l)

echo "============================================================"
echo "  Building al_v10 image"
echo "  Train: $TRAIN_TILES tiles"
echo "  Val:   $VAL_TILES tiles"
echo "  Image: $IMAGE_TAG"
echo "============================================================"

STAGE=$(mktemp -d -t al_v10_ctx_XXXXXX)
trap "echo 'cleaning up $STAGE'; rm -rf $STAGE" EXIT
echo "Staging at $STAGE ..."

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
echo "  copying train ($TRAIN_TILES tiles)..."
cp -rL "$TRAIN_DIR" "$STAGE/data/train"
echo "  copying val ($VAL_TILES tiles)..."
cp -rL "$VAL_DIR"   "$STAGE/data/val"

mkdir -p "$STAGE/docker"
cp docker/Dockerfile.al_v10        "$STAGE/docker/"
cp docker/al_v10_entrypoint.sh     "$STAGE/docker/"
cp docker/upload_model.py          "$STAGE/docker/"

echo ""
echo "Context: $(du -sh "$STAGE" | cut -f1)"
docker build --tag "$IMAGE_TAG" --file "$STAGE/docker/Dockerfile.al_v10" "$STAGE"

echo ""
echo "============================================================"
echo "  Built: $IMAGE_TAG"
echo "============================================================"
echo "  docker tag $IMAGE_TAG dockerkartok/herdnet:al_v10-latest"
echo "  docker push dockerkartok/herdnet:al_v10-latest"
