#!/bin/bash
# Launch al_v8 training (B4 ConvNeXtV2 tiny, balanced f2_score, post phase-15 iter-5).
# Master: 17,151 imgs / 75,821 iguana_point / 815 not_iguana / 20,692 boxes.
# Train: 5,142 imgs / Val: 1,125 imgs (14 untouched non-Fern datasets, 8 islands).
set -eu
cd /home/christian/hnee/HerdNet

LOG=20260603_al_v8_b4_train.log
echo "[$(date '+%T')] al_v8 launch — logging to $LOG"

python tools/train.py \
    --config-path /home/christian/hnee/HerdNet/configs/demo \
    --config-name al_v8_balanced \
    >"$LOG" 2>&1
