#!/bin/bash
# Check LoRA training progress across all classes
ROOT="/Users/imds/Desktop/wbc_synthesis"
WEIGHTS_DIR="$ROOT/lora/weights"
LOG_DIR="$ROOT/logs"

echo "=== WBC LoRA Training Progress ==="
echo "$(date)"
echo ""

CLASSES=(basophil eosinophil erythroblast ig lymphocyte monocyte neutrophil platelet)
DONE=0
TOTAL=${#CLASSES[@]}

for cls in "${CLASSES[@]}"; do
    LORA_FILE="$WEIGHTS_DIR/$cls/pytorch_lora_weights.safetensors"
    LOG_FILE="$LOG_DIR/lora_${cls}.log"

    if [ -f "$LORA_FILE" ]; then
        size=$(du -sh "$LORA_FILE" | cut -f1)
        echo "  ✓ $cls: DONE ($size)"
        DONE=$((DONE + 1))
    elif [ -f "$LOG_FILE" ]; then
        # Get current step from log
        last_step=$(grep "Steps:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '[0-9]+/100' | head -1)
        last_loss=$(grep "loss=" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE 'loss=[0-9.e+-]+' | head -1)
        if [ -n "$last_step" ]; then
            echo "  ⏳ $cls: step $last_step, $last_loss"
        else
            echo "  ⏳ $cls: training started (loading model...)"
        fi
    else
        echo "  ○ $cls: pending"
    fi
done

echo ""
echo "Completed: $DONE / $TOTAL"
echo ""

# Show master log tail
if [ -f "$LOG_DIR/lora_master.log" ]; then
    echo "=== Master log (last 5 lines) ==="
    tail -5 "$LOG_DIR/lora_master.log"
fi
