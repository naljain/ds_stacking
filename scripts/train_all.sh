#!/bin/bash
# Train learned DS primitives, one checkpoint per (arm, primitive) pair.
# Pooling left + right into one DS averages out the via-induced curvature
# because the two arms produce mirror-image (e, q_dot) labels at the same
# error e — so we train per-arm.
# Run after collect_ik.py has been run for both arms.

set -e

ARMS=("left" "right")
PRIMITIVES=("reach" "transport")

for a in "${ARMS[@]}"; do
    for p in "${PRIMITIVES[@]}"; do
        echo ""
        echo "==========================================="
        echo "Training arm=$a primitive=$p"
        echo "==========================================="
        python scripts/train_ds.py --primitive "$p" --arm "$a"
    done
done

echo ""
echo "[DONE] Learned DS primitives trained. Checkpoints in data/checkpoints/"
