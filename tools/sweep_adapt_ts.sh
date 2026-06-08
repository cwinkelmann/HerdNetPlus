#!/bin/bash
# Sweep adapt_ts values for full-size inference
# Runs inference_test.py with different adapt_ts and collects detections + metrics

set -e

RESULTS_DIR="output/inference_sweep/results"
mkdir -p "$RESULTS_DIR"

echo "=== Full-Size Inference Sweep ==="
echo "Model: best_models/fmo03_new_clean_convnext/best_model.pth"
echo ""

for ts in 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50; do
    echo "─── adapt_ts=${ts} ───"
    python tools/inference_test.py \
        --config-path ../configs/demo/inference \
        --config-name fmo03_fullsize \
        training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=${ts} \
        hydra.run.dir=./output/inference_sweep/ts_${ts} \
        2>&1 | tee "${RESULTS_DIR}/ts_${ts}.log"

    # Copy detections CSV
    if [ -f "output/inference_sweep/ts_${ts}/detections.csv" ]; then
        cp "output/inference_sweep/ts_${ts}/detections.csv" "${RESULTS_DIR}/detections_ts_${ts}.csv"
    fi
    echo ""
done

echo "=== Sweep complete ==="
echo "Results in: ${RESULTS_DIR}/"

# Summary: parse ME from each run
echo ""
echo "=== Summary ==="
printf "%-10s %6s %6s %6s %6s %6s %6s\n" "adapt_ts" "TP" "FP" "FN" "Prec" "Recall" "F1"
echo "────────────────────────────────────────────────────────────"
for ts in 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50; do
    LOG="${RESULTS_DIR}/ts_${ts}.log"
    if [ -f "$LOG" ]; then
        # Extract final metrics from the evaluator output
        METRICS=$(grep "f1_score:" "$LOG" | tail -1)
        if [ -n "$METRICS" ]; then
            echo "ts=${ts}: ${METRICS}"
        fi
    fi
done
