#!/bin/bash
# Quick start script - generates and executes nohup command for translation training
# Usage: ./start_training.sh [config_file]

# Configuration
CONFIG_FILE="${1:-configs/train_config.yaml}"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract parameters from config
get_value() {
    grep "$1:" "$CONFIG_FILE" | grep -v "#" | head -1 | sed "s/.*$1: *\"\?\([^\"]*\)\"\?.*/\1/"
}

BASE_MODEL=$(get_value "base_model")
TRAIN_SAMPLES=$(get_value "train_samples")
SOURCE_LANG=$(get_value "source_lang")
TARGET_LANG=$(get_value "target_lang")

# Determine model size
if [[ "$BASE_MODEL" =~ ([0-9]+)b ]]; then
    MODEL_SIZE="${BASH_REMATCH[1]}"
else
    MODEL_SIZE="1"
fi

# Create short language pair code
SRC_SHORT=$(echo "$SOURCE_LANG" | cut -c1-3 | tr '[:upper:]' '[:lower:]')
TGT_SHORT=$(echo "$TARGET_LANG" | cut -c1-3 | tr '[:upper:]' '[:lower:]')
LANG_PAIR="${SRC_SHORT}_${TGT_SHORT}"

# Create log directory: logs/translation_{size}b/{samples}_samples/{lang}/
LOG_DIR="logs/translation_${MODEL_SIZE}b/${TRAIN_SAMPLES}_samples/${LANG_PAIR}"
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
TIMESTAMP=$(date +%d%m%Y_%H%M)
LOG_FILE="${LOG_DIR}/${TRAIN_SAMPLES}_${LANG_PAIR}_${TIMESTAMP}.log"

# Display info
echo "=========================================="
echo "Translation Training"
echo "=========================================="
echo "Model: $BASE_MODEL (${MODEL_SIZE}B)"
echo "Samples: $TRAIN_SAMPLES"
echo "Language: $SOURCE_LANG → $TARGET_LANG ($LANG_PAIR)"
echo "Log: $LOG_FILE"
echo "=========================================="
echo ""

# Generate and display command
CMD="nohup python -u train_translation.py --config $CONFIG_FILE > $LOG_FILE 2>&1 &"
echo "Executing:"
echo "  $CMD"
echo ""

# Execute
eval $CMD
PID=$!

# Save PID
echo $PID > "training_${LANG_PAIR}.pid"

echo "✓ Training started (PID: $PID)"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Stop with:"
echo "  kill $PID"
