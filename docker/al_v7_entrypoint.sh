#!/bin/bash
# Single training run on the al_v7 split (post iter-v2 merge).
#
# B4 ConvNeXtV2, balanced f2_score selection, warm-started from phase8.
# Defaults sized for an 80 GB GPU (BATCH_SIZE=32, NUM_WORKERS=24).

set -e
export LC_NUMERIC=C
cd /app

echo "============================================================"
echo "  al_v7 training (post iter-v2 merge)"
echo "  Started:         $(date '+%Y-%m-%d %H:%M:%S')"
echo "  SEED:            ${SEED}"
echo "  BATCH_SIZE:      ${BATCH_SIZE}"
echo "  NUM_WORKERS:     ${NUM_WORKERS}"
echo "  AUG_MULT:        ${AUG_MULT}"
echo "  EPOCHS:          ${EPOCHS}"
echo "  LR:              ${LR}"
echo "  EMP_PROB:        ${EMP_PROB}"
echo "  VALIDATE_ON:     ${VALIDATE_ON}"
echo "  ADAPT_TS:        ${ADAPT_TS}"
echo "  SCORE_THRESHOLD: ${SCORE_THRESHOLD}"
echo "  WANDB_RUN:       ${WANDB_RUN}"
echo "  UPLOAD_MODEL:    ${UPLOAD_MODEL}"
echo "============================================================"

if [ "${WANDB_FLAG}" = "True" ] && [ -z "${WANDB_API_KEY}" ]; then
    echo "WARNING: WANDB_FLAG=True but WANDB_API_KEY unset. Running offline."
    export WANDB_MODE=offline
fi

TRAIN_CSV=/app/data/train/herdnet_format.csv
TRAIN_ROOT=/app/data/train/Default
VAL_CSV=/app/data/val/herdnet_format.csv
VAL_ROOT=/app/data/val/Default
WARM_START=/app/best_models/phase8/b4_seed42/best_model.pth

for p in "$TRAIN_CSV" "$TRAIN_ROOT" "$VAL_CSV" "$VAL_ROOT" "$WARM_START"; do
    [ -e "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done

OUT_DIR=/app/output/al_v7_b4_seed42/$(date '+%Y-%m-%d')/$(date '+%H-%M-%S')
mkdir -p "$OUT_DIR"

echo ""
echo "Output dir: $OUT_DIR"
echo ""

python tools/train.py \
    --config-path /app/configs/demo \
    --config-name data_scaling_b4 \
    seed="${SEED}" \
    datasets.train.csv_file="${TRAIN_CSV}" \
    datasets.train.root_dir="${TRAIN_ROOT}" \
    datasets.train.augmentation_multiplier="${AUG_MULT}" \
    datasets.train.albu_transforms.ObjectAwareRandomCrop.empty_probability="${EMP_PROB}" \
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
    +training_settings.evaluator.kwargs.lmds_kwargs.score_threshold="${SCORE_THRESHOLD}" \
    ~training_settings.loss_evaluation \
    model.load_from="${WARM_START}" \
    wandb_flag="${WANDB_FLAG}" \
    wandb_project="${WANDB_PROJECT}" \
    wandb_run="${WANDB_RUN}" \
    +wandb_tags=[active_learning,v7,b4,balanced,f2_score,post_iterv2_merge,espanola_floreana_val] \
    hydra.run.dir="${OUT_DIR}"

echo ""
echo "============================================================"
echo "  Training complete: $(date '+%F %T')"
echo "============================================================"

if [ "${UPLOAD_MODEL}" = "1" ] && [ -n "${WANDB_API_KEY}" ]; then
    python /app/docker/upload_model.py \
        --output-dir "${OUT_DIR}" \
        --artifact-name "al_v7_model" \
        --train-n "al_v7" \
        --seed "${SEED}" \
    || echo "wandb upload failed (continuing)"
fi
