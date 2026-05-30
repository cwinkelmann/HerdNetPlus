#!/bin/bash
# Run all Phase-13/14 model variants on the held-out test set with a
# coarse threshold sweep, then build a per-site precision/recall table.
#
# Models:
#   - Phase 13 single B4    (production candidate, F1 recipe)
#   - Phase 14 ensemble     (F5 recipe, 6 members)
#   - Phase 14b ensemble    (F2 recipe, 6 members)
#   - Phase 14c ensemble    (F1 recipe + aug_mult=1, 6 members)
#
# Thresholds: 0.20, 0.30, 0.40, 0.50 each (Phase 13 also at 0.25 and 0.35
# since that single-model has been our F1-best baseline).
set -e
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-/home/christian/data/training_data/2026_05_08_data_scaling}"
TEST_CSV="$DATA_ROOT/test/herdnet_format.csv"
TEST_IMAGES="$DATA_ROOT/test/Default"

OUT_DIR="$SCRIPT_DIR/output/test_sweep_$(date +%Y%m%d_%H%M%S)"
AGG="$OUT_DIR/per_site_aggregate.csv"
mkdir -p "$OUT_DIR"

# Preconditions
[ -f "$TEST_CSV" ] || { echo "ERROR: missing $TEST_CSV"; exit 1; }
[ -d "$TEST_IMAGES" ] || { echo "ERROR: missing $TEST_IMAGES"; exit 1; }

echo "============================================================"
echo "  Test-set evaluation sweep"
echo "  Test:  $TEST_IMAGES"
echo "  Out:   $OUT_DIR"
echo "  Aggregate: $AGG"
echo "============================================================"

PHASE13_MODEL="$SCRIPT_DIR/output/phase13_Nfull_s42/2026-05-16/05-50-52/best_model.pth"
PHASE14_MODELS=(
  "$SCRIPT_DIR/output/phase14_b4_s42/2026-05-27/12-27-24/best_model.pth"
  "$SCRIPT_DIR/output/phase14_b4_s7/2026-05-27/13-44-03/best_model.pth"
  "$SCRIPT_DIR/output/phase14_b4_s123/2026-05-27/14-38-36/best_model.pth"
  "$SCRIPT_DIR/output/phase14_b3_s7/2026-05-27/16-03-43/best_model.pth"
  "$SCRIPT_DIR/output/phase14_b3_s42/2026-05-27/17-28-17/best_model.pth"
  "$SCRIPT_DIR/output/phase14_b3_s123/2026-05-27/18-52-41/best_model.pth"
)
PHASE14B_MODELS=(
  "$SCRIPT_DIR/output/phase14b_b4_s42/2026-05-28/03-19-14/best_model.pth"
  "$SCRIPT_DIR/output/phase14b_b4_s7/2026-05-28/01-33-41/best_model.pth"
  "$SCRIPT_DIR/output/phase14b_b4_s123/2026-05-28/05-04-46/best_model.pth"
  "$SCRIPT_DIR/output/phase14b_b3_s7/2026-05-27/21-20-35/best_model.pth"
  "$SCRIPT_DIR/output/phase14b_b3_s42/2026-05-27/22-45-00/best_model.pth"
  "$SCRIPT_DIR/output/phase14b_b3_s123/2026-05-28/00-09-28/best_model.pth"
)
PHASE14C_MODELS=(
  "$SCRIPT_DIR/output/phase14c_b4_s42/2026-05-28/15-05-30/best_model.pth"
  "$SCRIPT_DIR/output/phase14c_b4_s7/2026-05-28/13-04-32/best_model.pth"
  "$SCRIPT_DIR/output/phase14c_b4_s123/2026-05-28/17-28-17/best_model.pth"
  "$SCRIPT_DIR/output/phase14c_b3_s7/2026-05-28/08-43-34/best_model.pth"
  "$SCRIPT_DIR/output/phase14c_b3_s42/2026-05-28/10-08-04/best_model.pth"
  "$SCRIPT_DIR/output/phase14c_b3_s123/2026-05-28/11-32-47/best_model.pth"
)

for p in "$PHASE13_MODEL" "${PHASE14_MODELS[@]}" "${PHASE14B_MODELS[@]}" "${PHASE14C_MODELS[@]}"; do
  [ -f "$p" ] || { echo "ERROR: missing model $p"; exit 1; }
done

# Helper: run single-model inference at one threshold + per-site analysis.
run_single() {
  local MODEL="$1"; local LABEL="$2"; local TS="$3"; local CFG="$4"
  local SUB="$OUT_DIR/${LABEL}_ts${TS}"
  local LOG="$OUT_DIR/${LABEL}_ts${TS}.log"
  mkdir -p "$SUB"
  echo "[$(date '+%H:%M:%S')] ===> $LABEL @ ts=$TS"
  (cd "$SUB" && python "$SCRIPT_DIR/tools/infer.py" \
      --config-dir "$SCRIPT_DIR/configs/demo" \
      --config-name "$CFG" \
      --images "$TEST_IMAGES" \
      --model "$MODEL" \
      --evaluate \
      --overrides \
        "wandb_flag=False" \
        "datasets.test.csv_file=$TEST_CSV" \
        "datasets.test.root_dir=$TEST_IMAGES" \
        "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=$TS") \
      > "$LOG" 2>&1 || { echo "FAILED: $LABEL ts=$TS"; return 1; }

  local DET="$SUB/detections.csv"
  [ -f "$DET" ] || { echo "  no detections.csv produced"; return 1; }
  python "$SCRIPT_DIR/tools/per_site_metrics.py" \
    --gt "$TEST_CSV" --det "$DET" \
    --model "$LABEL" --threshold "$TS" --dataset "test" \
    --out "$AGG"
}

# Helper: run ensemble inference + per-site analysis.
run_ensemble() {
  local LABEL="$1"; local CFG="$2"; local TS="$3"; shift 3
  local MODELS=("$@")
  local SUB="$OUT_DIR/${LABEL}_ts${TS}"
  local LOG="$OUT_DIR/${LABEL}_ts${TS}.log"
  mkdir -p "$SUB"
  echo "[$(date '+%H:%M:%S')] ===> $LABEL @ ts=$TS"
  (cd "$SUB" && python "$SCRIPT_DIR/tools/ensemble_infer.py" \
      --config-dir "$SCRIPT_DIR/configs/demo" \
      --config-name "$CFG" \
      --images "$TEST_IMAGES" \
      --models "${MODELS[@]}" \
      --evaluate \
      --overrides \
        "wandb_flag=False" \
        "datasets.test.csv_file=$TEST_CSV" \
        "datasets.test.root_dir=$TEST_IMAGES" \
        "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=$TS") \
      > "$LOG" 2>&1 || { echo "FAILED: $LABEL ts=$TS"; return 1; }

  local DET="$SUB/detections.csv"
  [ -f "$DET" ] || { echo "  no detections.csv produced"; return 1; }
  python "$SCRIPT_DIR/tools/per_site_metrics.py" \
    --gt "$TEST_CSV" --det "$DET" \
    --model "$LABEL" --threshold "$TS" --dataset "test" \
    --out "$AGG"
}

# ---- Phase 13 single B4 — 5 thresholds (the production candidate gets a wider sweep) ----
for TS in 0.20 0.25 0.30 0.35 0.40 0.50; do
  run_single "$PHASE13_MODEL" "phase13_single_b4" "$TS" "data_scaling_b4"
done

# ---- Phase 14 ensemble (F5 recipe) — peak band around ts=0.50 on val ----
for TS in 0.30 0.40 0.50; do
  run_ensemble "phase14_ens" "phase14_ensemble_b4" "$TS" "${PHASE14_MODELS[@]}"
done

# ---- Phase 14b ensemble (F2 recipe) — peak band around ts=0.50 on val ----
for TS in 0.30 0.40 0.50; do
  run_ensemble "phase14b_ens" "phase14b_ensemble_b4" "$TS" "${PHASE14B_MODELS[@]}"
done

# ---- Phase 14c ensemble (F1 recipe, aug_mult=1) — peak band around ts=0.50 on val ----
for TS in 0.30 0.40 0.50; do
  run_ensemble "phase14c_ens" "phase14c_ensemble_b4" "$TS" "${PHASE14C_MODELS[@]}"
done

echo ""
echo "============================================================"
echo "  Test-set sweep complete"
echo "  Per-site aggregate: $AGG"
echo "============================================================"
echo ""
echo "OVERALL rows:"
grep ",OVERALL," "$AGG" | awk -F, '{ printf "  %-25s ts=%s F1=%s P=%s R=%s tp=%s fp=%s fn=%s\n", $1, $3, $11, $9, $10, $6, $7, $8 }'
