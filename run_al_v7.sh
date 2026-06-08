#!/bin/bash
# Launch al_v7 training (B4 ConvNeXtV2, balanced f2_score, post iter-v2 merge).
# Master: 17,151 imgs / 75,755 iguana_point keypoints / 20,692 boxes.
# Train: 5,658 imgs / Val: 610 imgs (net-new Espanola+Floreana from iter-v2).
set -eu
cd /home/christian/hnee/HerdNet

LOG=20260603_al_v7_b4_train.log
echo "[$(date '+%T')] al_v7 launch — logging to $LOG"

python tools/train.py \
    --config-path /home/christian/hnee/HerdNet/configs/demo \
    --config-name al_v7_balanced \
    >"$LOG" 2>&1
