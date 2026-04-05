#!/bin/bash
# Run SNNTorch comparison training
# Uses gilgamesh configuration: 36→12→10 LIF SNN, 15 epochs, batch 128

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Default arguments matching gilgamesh configuration
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./comparison/results}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_STEPS="${NUM_STEPS:-25}"

echo "====================================="
echo "SNNTorch Comparison Training"
echo "====================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Timesteps: $NUM_STEPS"
echo "====================================="

python3 comparison/snntorch_comparison.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-steps "$NUM_STEPS" \
    "$@"

echo ""
echo "====================================="
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "====================================="
