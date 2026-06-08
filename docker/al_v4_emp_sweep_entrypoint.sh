#!/bin/bash
# Sequential empty_probability sweep on the al_v4 split.
#
# For each EMP value in EMP_VALUES (comma-separated), runs one
# `tools/train.py` with that ObjectAwareRandomCrop.empty_probability.
# Outputs land in /app/output/emp_<value>/<date>/<time>/, with the
# best_model.pth saved on F5_score. Run summary at the end.

set -e
export LC_NUMERIC=C
cd /app

echo "============================================================"
echo "  al_v4 empty_probability sweep"
echo "  Started:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "  SEED:         ${SEED}"
echo "  EMP_VALUES:   ${EMP_VALUES}"
echo "  BATCH_SIZE:   ${BATCH_SIZE}"
echo "  NUM_WORKERS:  ${NUM_WORKERS}"
echo "  AUG_MULT:     ${AUG_MULT}"
echo "  EPOCHS:       ${EPOCHS}"
echo "  LR:           ${LR}"
echo "  VALIDATE_ON:  ${VALIDATE_ON}"
echo "  ADAPT_TS:     ${ADAPT_TS}"
echo "  UPLOAD_MODEL: ${UPLOAD_MODEL}"
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

SUMMARY=/app/output/emp_sweep_summary.tsv
mkdir -p /app/output
printf "empty_prob\tbest_F5\tbest_F1\trecall\tprecision\tepoch\toutput_dir\n" > "$SUMMARY"

# IFS-split EMP_VALUES on commas
IFS=',' read -ra EMP_ARR <<< "${EMP_VALUES}"
for EMP in "${EMP_ARR[@]}"; do
    EMP=$(echo "$EMP" | tr -d ' ')
    OUT_DIR=/app/output/emp_${EMP}/$(date '+%Y-%m-%d')/$(date '+%H-%M-%S')
    echo ""
    echo "============================================================"
    echo "  Training with empty_probability=$EMP"
    echo "  Out:    $OUT_DIR"
    echo "============================================================"

    python tools/train.py \
        --config-path /app/configs/demo \
        --config-name data_scaling_b4 \
        seed="${SEED}" \
        datasets.train.csv_file="${TRAIN_CSV}" \
        datasets.train.root_dir="${TRAIN_ROOT}" \
        datasets.train.augmentation_multiplier="${AUG_MULT}" \
        datasets.train.albu_transforms.ObjectAwareRandomCrop.empty_probability="${EMP}" \
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
        wandb_run="al_v4_emp_${EMP}" \
        +wandb_tags=[active_learning,v4,b4,emp_sweep,emp_${EMP}] \
        hydra.run.dir="${OUT_DIR}"

    HYDRA_OUT=$(ls -td "${OUT_DIR%/*}"/*/ 2>/dev/null | head -1 | sed 's:/$::')
    BEST_LINE=$(grep "\[BEST_METRICS\]" "${HYDRA_OUT}"/*_training.log 2>/dev/null | tail -1)
    F5=$(echo "$BEST_LINE" | grep -oE "f5=[0-9.]+" | awk -F= '{print $2}')
    F1=$(echo "$BEST_LINE" | grep -oE "f1=[0-9.]+" | awk -F= '{print $2}')
    R=$(echo "$BEST_LINE"  | grep -oE "recall=[0-9.]+"    | awk -F= '{print $2}')
    P=$(echo "$BEST_LINE"  | grep -oE "precision=[0-9.]+" | awk -F= '{print $2}')
    EP=$(echo "$BEST_LINE" | grep -oE "Epoch: \[[0-9]+\]" | grep -oE "[0-9]+")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$EMP" "$F5" "$F1" "$R" "$P" "$EP" "$HYDRA_OUT" >> "$SUMMARY"

    if [ "${UPLOAD_MODEL}" = "1" ] && [ -n "${WANDB_API_KEY}" ]; then
        python /app/docker/upload_model.py \
            --output-dir "${HYDRA_OUT}" \
            --artifact-name "al_v4_emp_${EMP}_model" \
            --train-n "al_v4_emp_${EMP}" \
            --seed "${SEED}" \
        || echo "wandb upload failed for emp=${EMP} (continuing)"
    fi
done

echo ""
echo "============================================================"
echo "  Sweep complete: $(date '+%F %T')"
echo "============================================================"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
