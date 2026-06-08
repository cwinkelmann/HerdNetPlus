#!/bin/bash
# Launch al_v10 training on local 4090.
# H-aware loader (hnp=0.25, ep=0.10), 15 epochs, batch=4 (proven safe locally).
# Same warm-start (phase8 b4_seed42), same val regime as al_v8/v9.
# Train: 5,991 imgs (master \ val), Val: 646 imgs (5-island ortho holdout).
set -eu
cd /home/christian/hnee/HerdNet

LOG=20260608_al_v10_b4_train.log
echo "[$(date '+%T')] al_v10 launch — logging to $LOG"

python tools/train.py \
    --config-path /home/christian/hnee/HerdNet/configs/demo \
    --config-name al_v10_balanced \
    >"$LOG" 2>&1
