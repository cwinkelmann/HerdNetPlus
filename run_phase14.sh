#!/bin/bash
# Phase 14 Stage A — train the 6 cross-arch ensemble members.
#
# Defaults: sequential local runs. Set ARCHS / SEEDS to subset, e.g.:
#   ARCHS=b4 SEEDS=42 bash run_phase14.sh        # one member only
#   ARCHS="b3 b4" SEEDS="7 42 123" bash run_phase14.sh
#
# Each training:
#   - N=full pool (Phase-13 split, hardcoded path below)
#   - F5 recipe (validate_on=f5_score, early-stop patience=3,
#     CE foreground 5.0->2.0, augmentation_multiplier=1)
#   - warm-start from best_models/phase8/<arch>_seed<seed>/best_model.pth
#   - 10 epochs max, expected wall-clock ~5h each
#   - wandb project hn_phase14_ensemble

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ARCHS="${ARCHS:-b3 b4}"
SEEDS="${SEEDS:-7 42 123}"
DATA_ROOT="${DATA_ROOT:-/home/christian/data/training_data/2026_05_08_data_scaling}"
WANDB_PROJECT="${WANDB_PROJECT:-hn_phase14_ensemble}"
WANDB_FLAG="${WANDB_FLAG:-True}"
RUN_TAG="${RUN_TAG:-phase14_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$SCRIPT_DIR/output/${RUN_TAG}_logs"

mkdir -p "$LOG_DIR"

TRAIN_CSV="$DATA_ROOT/train_Nfull_s42/herdnet_format.csv"
TRAIN_ROOT="$DATA_ROOT/train_Nfull_s42/Default"
VAL_CROP_CSV="$DATA_ROOT/val/herdnet_format_512_0_crops.csv"
VAL_CROP_ROOT="$DATA_ROOT/val/crops_512_numNone_overlap0"
VAL_FULL_CSV="$DATA_ROOT/val/herdnet_format.csv"
VAL_FULL_ROOT="$DATA_ROOT/val/Default"

# Pre-flight: dataset and all 6 warm-starts must exist.
for p in "$TRAIN_CSV" "$TRAIN_ROOT" "$VAL_CROP_CSV" "$VAL_CROP_ROOT" "$VAL_FULL_CSV" "$VAL_FULL_ROOT"; do
  [ -e "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done
for arch in $ARCHS; do
  for seed in $SEEDS; do
    p="$SCRIPT_DIR/best_models/phase8/${arch}_seed${seed}/best_model.pth"
    [ -e "$p" ] || { echo "ERROR: missing warm-start $p"; exit 1; }
  done
done

echo "============================================================"
echo "  Phase 14 Stage A — ensemble training"
echo "  Archs:   $ARCHS"
echo "  Seeds:   $SEEDS"
echo "  Logs:    $LOG_DIR"
echo "  Wandb:   $WANDB_PROJECT"
echo "============================================================"

for ARCH in $ARCHS; do
  CONFIG_NAME="phase14_ensemble_${ARCH}"
  for SEED in $SEEDS; do
    RUN_NAME="phase14_${ARCH}_s${SEED}"
    WARM_START="$SCRIPT_DIR/best_models/phase8/${ARCH}_seed${SEED}/best_model.pth"
    OUT_DIR="$SCRIPT_DIR/output/${RUN_NAME}/$(date +%Y-%m-%d)/$(date +%H-%M-%S)"
    LOG="$LOG_DIR/${RUN_NAME}_training.log"

    echo ""
    echo "[$(date '+%H:%M:%S')] ===> $RUN_NAME"
    echo "  config: $CONFIG_NAME  warm-start: $WARM_START"
    echo "  out:    $OUT_DIR"
    echo "  log:    $LOG"

    python tools/train.py \
      --config-path configs/demo \
      --config-name "$CONFIG_NAME" \
      "seed=${SEED}" \
      "datasets.train.csv_file=${TRAIN_CSV}" \
      "datasets.train.root_dir=${TRAIN_ROOT}" \
      "datasets.validate.csv_file=${VAL_CROP_CSV}" \
      "datasets.validate.root_dir=${VAL_CROP_ROOT}" \
      "datasets.test.csv_file=${VAL_FULL_CSV}" \
      "datasets.test.root_dir=${VAL_FULL_ROOT}" \
      "model.load_from=${WARM_START}" \
      "wandb_flag=${WANDB_FLAG}" \
      "wandb_project=${WANDB_PROJECT}" \
      "wandb_run=${RUN_NAME}" \
      "+wandb_tags=[phase14,ensemble,${ARCH},f5_recipe,s${SEED}]" \
      "hydra.run.dir=${OUT_DIR}" \
      2>&1 | tee "$LOG" \
      || { echo "FAILED: $RUN_NAME (exit $?)"; continue; }

    BEST=$(find "$OUT_DIR" -name 'best_model.pth' | head -1)
    if [ -n "$BEST" ]; then
      echo "  best_model.pth: $BEST"
    else
      echo "  WARNING: no best_model.pth in $OUT_DIR"
    fi
  done
done

echo ""
echo "============================================================"
echo "  Phase 14 Stage A complete"
echo "  Member logs:   $LOG_DIR"
echo "  Next: Stage B (ensemble inference + adapt_ts sweep)"
echo "============================================================"
