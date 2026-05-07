#!/bin/bash
# Train learned DS primitives across both arms.
# Run after collect_ik.py has been run for both arms.

set -e

PRIMITIVES=("reach" "transport")

for p in "${PRIMITIVES[@]}"; do
    echo ""
    echo "==========================================="
    echo "Training primitive: $p"
    echo "==========================================="
    python scripts/train_ds.py --primitive "$p" --arm both
done

echo ""
echo "[DONE] Learned DS primitives trained. Checkpoints in data/checkpoints/"
