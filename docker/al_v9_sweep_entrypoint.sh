#!/bin/bash
# al_v9 4-experiment sweep (2x2: dataloader × Fer-in-train).
#
# Same val for all 4 runs. Trains B4 ConvNeXtV2 (tiny) from the same
# phase8 warm-start. Each experiment runs ${EPOCHS} (default 30) with
# validate_on=f2_score so we pick a balanced-recall snapshot.
#
# EXPERIMENT env var controls which experiment(s) to run:
#   EXPERIMENT=all              (default) — runs A, B, C, D sequentially
#   EXPERIMENT=A | A_old_ferout — just A (old loader, fer out)
#   EXPERIMENT=B | B_old_ferin  — just B (old loader, fer in)
#   EXPERIMENT=C | C_new_ferout — just C (H-aware loader, fer out)
#   EXPERIMENT=D | D_new_ferin  — just D (H-aware loader, fer in)
#
# For 8-GPU parallel use: launch 4 containers each pinned to one GPU:
#   for i in 0 1 2 3; do
#     EXP=(A B C D); docker run -d --gpus "\"device=$i\"" \
#       -e WANDB_API_KEY=$WANDB_API_KEY -e EXPERIMENT=${EXP[$i]} \
#       -v $(pwd)/al_v9_output:/app/output --name al_v9_${EXP[$i]} \
#       dockerkartok/herdnet:al_v9_sweep-latest
#   done
#
# Tracks one wandb run per experiment, with tag matrix encoding both
# axes. Summary tsv at /app/output/al_v9_sweep_summary_${EXPERIMENT}.tsv
# (one per experiment when running in parallel, single combined file
# when EXPERIMENT=all).

set -e
export LC_NUMERIC=C
cd /app

EXPERIMENT="${EXPERIMENT:-all}"

echo "============================================================"
echo "  al_v9 sweep — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  EXPERIMENT:      ${EXPERIMENT}"
echo "  SEED:            ${SEED}"
echo "  EPOCHS:          ${EPOCHS}"
echo "  BATCH_SIZE:      ${BATCH_SIZE}"
echo "  NUM_WORKERS:     ${NUM_WORKERS}"
echo "  LR:              ${LR}"
echo "  VALIDATE_ON:     ${VALIDATE_ON}"
echo "  ADAPT_TS:        ${ADAPT_TS}"
echo "  SCORE_THRESHOLD: ${SCORE_THRESHOLD}"
echo "  UPLOAD_MODEL:    ${UPLOAD_MODEL}"
echo "============================================================"

if [ "${WANDB_FLAG}" = "True" ] && [ -z "${WANDB_API_KEY}" ]; then
    echo "WARNING: WANDB_FLAG=True but WANDB_API_KEY unset. Running offline."
    export WANDB_MODE=offline
fi

WARM_START=/app/best_models/phase8/b4_seed42/best_model.pth
[ -e "$WARM_START" ] || { echo "ERROR: missing warm-start $WARM_START"; exit 1; }

SUMMARY=/app/output/al_v9_sweep_summary_${EXPERIMENT}.tsv
mkdir -p /app/output
printf "exp\tfer_mode\tloader\tbest_F2\tbest_F1\trecall\tprecision\tepoch\toutput_dir\n" > "$SUMMARY"

# Full experiment matrix: (NAME, FER_MODE, HARD_NEG_PROB, EMPTY_PROB, TAG_LOADER).
ALL_EXPERIMENTS=(
    "A_old_ferout fer_out 0.00 0.20 old_loader"
    "B_old_ferin  fer_in  0.00 0.20 old_loader"
    "C_new_ferout fer_out 0.25 0.10 hnp25_ep10"
    "D_new_ferin  fer_in  0.25 0.10 hnp25_ep10"
)

# Filter by EXPERIMENT env var. Accept short names (A,B,C,D) or full names.
EXPERIMENTS=()
case "${EXPERIMENT,,}" in
    all)
        EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
        ;;
    a|a_old_ferout)
        EXPERIMENTS=("${ALL_EXPERIMENTS[0]}")
        ;;
    b|b_old_ferin)
        EXPERIMENTS=("${ALL_EXPERIMENTS[1]}")
        ;;
    c|c_new_ferout)
        EXPERIMENTS=("${ALL_EXPERIMENTS[2]}")
        ;;
    d|d_new_ferin)
        EXPERIMENTS=("${ALL_EXPERIMENTS[3]}")
        ;;
    *)
        echo "ERROR: unknown EXPERIMENT='${EXPERIMENT}'. Use A|B|C|D|all."
        exit 1
        ;;
esac
echo "Will run ${#EXPERIMENTS[@]} experiment(s): "
for e in "${EXPERIMENTS[@]}"; do echo "  - $e"; done
echo "============================================================"

for EXP in "${EXPERIMENTS[@]}"; do
    read -r NAME FER_MODE HNP EP TAG_LOADER <<< "$EXP"
    TRAIN_CSV=/app/data/${FER_MODE}/train/herdnet_format.csv
    TRAIN_ROOT=/app/data/${FER_MODE}/train/Default
    VAL_CSV=/app/data/${FER_MODE}/val/herdnet_format.csv
    VAL_ROOT=/app/data/${FER_MODE}/val/Default

    for p in "$TRAIN_CSV" "$TRAIN_ROOT" "$VAL_CSV" "$VAL_ROOT"; do
        [ -e "$p" ] || { echo "ERROR: missing $p for exp $NAME"; exit 1; }
    done

    OUT_DIR=/app/output/al_v9_${NAME}/$(date '+%Y-%m-%d')/$(date '+%H-%M-%S')
    mkdir -p "$OUT_DIR"

    echo ""
    echo "============================================================"
    echo "  [$NAME] fer_mode=$FER_MODE  hard_negative_probability=$HNP"
    echo "          empty_probability=$EP"
    echo "          out: $OUT_DIR"
    echo "============================================================"

    python tools/train.py \
        --config-path /app/configs/demo \
        --config-name data_scaling_b4 \
        seed="${SEED}" \
        datasets.train.csv_file="${TRAIN_CSV}" \
        datasets.train.root_dir="${TRAIN_ROOT}" \
        datasets.train.augmentation_multiplier="${AUG_MULT}" \
        datasets.train.albu_transforms.ObjectAwareRandomCrop.empty_probability="${EP}" \
        +datasets.train.albu_transforms.ObjectAwareRandomCrop.hard_negative_probability="${HNP}" \
        +datasets.train.albu_transforms.ObjectAwareRandomCrop.hard_negative_label=2 \
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
        wandb_run="al_v9_${NAME}" \
        +wandb_tags=[active_learning,v9,b4,${FER_MODE},${TAG_LOADER},sweep_2x2] \
        hydra.run.dir="${OUT_DIR}"

    HYDRA_OUT=$(ls -td "${OUT_DIR%/*}"/*/ 2>/dev/null | head -1 | sed 's:/$::')
    BEST_LINE=$(grep "\[BEST_METRICS\]" "${HYDRA_OUT}"/*_training.log 2>/dev/null | tail -1)
    F2=$(echo "$BEST_LINE" | grep -oE "f2=[0-9.]+" | awk -F= '{print $2}')
    F1=$(echo "$BEST_LINE" | grep -oE "f1=[0-9.]+" | awk -F= '{print $2}')
    R=$(echo "$BEST_LINE"  | grep -oE "recall=[0-9.]+"    | awk -F= '{print $2}')
    P=$(echo "$BEST_LINE"  | grep -oE "precision=[0-9.]+" | awk -F= '{print $2}')
    EP_BEST=$(echo "$BEST_LINE" | grep -oE "Epoch: \[[0-9]+\]" | grep -oE "[0-9]+")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$NAME" "$FER_MODE" "$TAG_LOADER" "$F2" "$F1" "$R" "$P" "$EP_BEST" "$HYDRA_OUT" >> "$SUMMARY"

    if [ "${UPLOAD_MODEL}" = "1" ] && [ -n "${WANDB_API_KEY}" ]; then
        python /app/docker/upload_model.py \
            --output-dir "${HYDRA_OUT}" \
            --artifact-name "al_v9_${NAME}_model" \
            --train-n "al_v9_${NAME}" \
            --seed "${SEED}" \
        || echo "wandb upload failed for ${NAME} (continuing)"
    fi
done

echo ""
echo "============================================================"
echo "  Sweep complete: $(date '+%F %T')"
echo "============================================================"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
