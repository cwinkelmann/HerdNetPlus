#!/bin/bash
# Build the al_v9 sweep container.
# Stages both train_fer_{in,out} + shared val into /app/data/{fer_in,fer_out}/.

set -e
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

AL_V9_ROOT="${AL_V9_ROOT:-/home/christian/data/training_data/2026_06_04_al_v9_canonical}"
IMAGE_TAG="${IMAGE_TAG:-herdnet-al-v9-sweep:latest}"

FER_IN_DIR="${AL_V9_ROOT}/al_v9_fer_in_2026_06_04"
FER_OUT_DIR="${AL_V9_ROOT}/al_v9_fer_out_2026_06_04"
WARM_START="${REPO_DIR}/best_models/phase8/b4_seed42/best_model.pth"

for d in "$FER_IN_DIR/train" "$FER_IN_DIR/val" "$FER_OUT_DIR/train" "$FER_OUT_DIR/val"; do
    [ -d "$d" ] || { echo "ERROR: missing $d"; exit 1; }
done
[ -f "$FER_IN_DIR/train/herdnet_format.csv" ] || { echo "ERROR: missing fer_in train csv"; exit 1; }
[ -f "$FER_OUT_DIR/train/herdnet_format.csv" ] || { echo "ERROR: missing fer_out train csv"; exit 1; }
[ -e "$WARM_START" ] || { echo "ERROR: missing warm-start"; exit 1; }
WARM_START_REAL=$(readlink -f "$WARM_START")

IN_TILES=$(find "$FER_IN_DIR/train/Default" -maxdepth 1 -type f -o -type l | wc -l)
OUT_TILES=$(find "$FER_OUT_DIR/train/Default" -maxdepth 1 -type f -o -type l | wc -l)
VAL_TILES=$(find "$FER_IN_DIR/val/Default" -maxdepth 1 -type f -o -type l | wc -l)

echo "============================================================"
echo "  Building al_v9 sweep image"
echo "  Train fer_in:  $IN_TILES tiles"
echo "  Train fer_out: $OUT_TILES tiles"
echo "  Val:           $VAL_TILES tiles (shared)"
echo "  Image tag:     $IMAGE_TAG"
echo "============================================================"

STAGE=$(mktemp -d -t al_v9_ctx_XXXXXX)
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

mkdir -p "$STAGE/data/fer_in" "$STAGE/data/fer_out"
echo "  copying fer_in train ($IN_TILES tiles) + val ($VAL_TILES tiles) ..."
cp -rL "$FER_IN_DIR/train" "$STAGE/data/fer_in/train"
cp -rL "$FER_IN_DIR/val"   "$STAGE/data/fer_in/val"
echo "  copying fer_out train ($OUT_TILES tiles) + val ($VAL_TILES tiles) ..."
cp -rL "$FER_OUT_DIR/train" "$STAGE/data/fer_out/train"
cp -rL "$FER_OUT_DIR/val"   "$STAGE/data/fer_out/val"

mkdir -p "$STAGE/docker"
cp docker/Dockerfile.al_v9_sweep        "$STAGE/docker/"
cp docker/al_v9_sweep_entrypoint.sh     "$STAGE/docker/"
cp docker/upload_model.py               "$STAGE/docker/"

echo ""
echo "Context size: $(du -sh "$STAGE" | cut -f1)"
docker build --tag "$IMAGE_TAG" --file "$STAGE/docker/Dockerfile.al_v9_sweep" "$STAGE"

echo ""
echo "============================================================"
echo "  Built: $IMAGE_TAG"
echo "============================================================"
echo ""
echo "Run on 80 GB GPU (4 sequential experiments, ~30 epochs each):"
echo "  docker run --rm --gpus all --shm-size=16g \\"
echo "    -e WANDB_API_KEY=\$WANDB_API_KEY \\"
echo "    -v \$(pwd)/output:/app/output \\"
echo "    $IMAGE_TAG"
echo ""
echo "Push:"
echo "  docker tag $IMAGE_TAG dockerkartok/herdnet:al_v9_sweep-latest"
echo "  docker push dockerkartok/herdnet:al_v9_sweep-latest"
