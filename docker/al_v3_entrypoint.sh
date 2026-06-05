#!/bin/bash
# Active-learning iter-3 training entrypoint.
#
# Defaults (set by Dockerfile.al_v3 ENV) implement the recall-tuned
# strategy: validate_on=f5_score, adapt_ts=0.15, BATCH_SIZE=32,
# NUM_WORKERS=24, EPOCHS=30, LR=6.4e-4 (linear scaling rule for the 8x
# bigger batch vs the 16 GB recipe), AUG_MULT=75.
#
# All settings can be overridden via `docker run -e KEY=value`.

set -e
export LC_NUMERIC=C
cd /app

echo "============================================================"
echo "  HerdNet active-learning iter-3 training container"
echo "  Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "  SEED:          ${SEED}"
echo "  WANDB_PROJECT: ${WANDB_PROJECT}"
echo "  WANDB_FLAG:    ${WANDB_FLAG}"
echo "  EPOCHS:        ${EPOCHS}"
echo "  BATCH_SIZE:    ${BATCH_SIZE}"
echo "  NUM_WORKERS:   ${NUM_WORKERS}"
echo "  AUG_MULT:      ${AUG_MULT}"
echo "  LR:            ${LR}"
echo "  VALIDATE_ON:   ${VALIDATE_ON}"
echo "  ADAPT_TS:      ${ADAPT_TS}"
echo "  UPLOAD_MODEL:  ${UPLOAD_MODEL}"
echo "============================================================"

if [ "${WANDB_FLAG}" = "True" ] && [ -z "${WANDB_API_KEY}" ]; then
  echo "WARNING: WANDB_FLAG=True but WANDB_API_KEY is unset. Running offline."
  export WANDB_MODE=offline
fi

TRAIN_CSV=/app/data/train/herdnet_format.csv
TRAIN_ROOT=/app/data/train/Default
VAL_CSV=/app/data/val/herdnet_format.csv
VAL_ROOT=/app/data/val/Default
WARM_START=/app/best_models/phase8/b4_seed42/best_model.pth

for p in "$TRAIN_CSV" "$TRAIN_ROOT" "$VAL_CSV" "$VAL_ROOT" "$WARM_START"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: missing $p"; exit 1
  fi
done

OUT_DIR=output/al_v3_b4_recall_$(date '+%Y-%m-%d')/$(date '+%H-%M-%S')

# Build training overrides. The validate path uses full-size + stitcher
# (test block); loss_evaluation is disabled because the canonical
# Phase-13 cropped-val path is not bundled — F1/recall on full-size is
# the only eval signal.
python tools/train.py \
  --config-path /app/configs/demo \
  --config-name data_scaling_b4 \
  seed="${SEED}" \
  datasets.train.csv_file="${TRAIN_CSV}" \
  datasets.train.root_dir="${TRAIN_ROOT}" \
  datasets.train.augmentation_multiplier="${AUG_MULT}" \
  datasets.validate.csv_file="${VAL_CSV}" \
  datasets.validate.root_dir="${VAL_ROOT}" \
  datasets.test.csv_file="${VAL_CSV}" \
  datasets.test.root_dir="${VAL_ROOT}" \
  training_settings.epochs="${EPOCHS}" \
  training_settings.valid_freq=1 \
  training_settings.batch_size="${BATCH_SIZE}" \
  training_settings.num_workers="${NUM_WORKERS}" \
  training_settings.lr="${LR}" \
  training_settings.evaluator.validate_on="${VALIDATE_ON}" \
  training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts="${ADAPT_TS}" \
  ~training_settings.loss_evaluation \
  model.load_from="${WARM_START}" \
  wandb_flag="${WANDB_FLAG}" \
  wandb_project="${WANDB_PROJECT}" \
  wandb_run="${WANDB_RUN}" \
  +wandb_tags=[active_learning,v3,b4,recall_tuned,80gb,docker] \
  hydra.run.dir="${OUT_DIR}"

# Resolve final hydra output dir (now contains best_model.pth).
HYDRA_OUT=$(ls -td "${OUT_DIR%/*}"/*/ 2>/dev/null | head -1 | sed 's:/$::')
echo "Training output dir: ${HYDRA_OUT}"

if [ "${UPLOAD_MODEL}" = "1" ] && [ -n "${WANDB_API_KEY}" ]; then
  python /app/docker/upload_model.py \
    --output-dir "${HYDRA_OUT}" \
    --artifact-name "${ARTIFACT_NAME}" \
    --train-n al_v3 \
    --seed "${SEED}" \
  || echo "wandb upload_model.py failed (training succeeded — checkpoint is on disk)"
fi

echo "============================================================"
echo "  Done. Best checkpoint at: ${HYDRA_OUT}/best_model.pth"
echo "============================================================"
