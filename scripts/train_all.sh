#!/bin/bash
# Train all five primitives across both arms.
# Run after collect_ik.py has been run for both arms.

set -e

PRIMITIVES=("reach" "grasp" "lift" "transport" "place")

for p in "${PRIMITIVES[@]}"; do
    echo ""
    echo "==========================================="
    echo "Training primitive: $p"
    echo "==========================================="
    python scripts/train_ds.py --primitive "$p" --arm both
done

echo ""
echo "[DONE] All primitives trained. Checkpoints in data/checkpoints/"
