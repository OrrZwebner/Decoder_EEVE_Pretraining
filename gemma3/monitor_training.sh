#!/bin/bash
# Monitor current training progress with new log structure
# Log structure: logs/pre_train_Xb/N_samples/n_tokens/N_n_timestamp.log

CONFIG_TEMPLATE="gemma_config_template.yaml"

# Function to extract model size from config template
get_model_size() {
    local model_line=$(grep -A 1 "^model:" "$CONFIG_TEMPLATE" | grep "name:")
    local model_size=$(echo "$model_line" | sed -n 's/.*gemma-3-\([0-9]\+\)b.*/\1/p')
    echo "$model_size"
}

# Function to extract training samples from config template
get_train_samples() {
    local test_mode_line=$(grep "^  test_mode:" "$CONFIG_TEMPLATE" | grep -v "#.*test_mode:" | tail -1)
    local test_mode=$(echo "$test_mode_line" | sed 's/.*: *\(true\|false\).*/\1/')
    
    if [ "$test_mode" = "true" ]; then
        local samples=$(grep "^  limit_train_samples:" "$CONFIG_TEMPLATE" | grep -v "#" | tail -1 | sed 's/.*limit_train_samples: *\([0-9]*\).*/\1/')
        echo "$samples"
    else
        echo "full"
    fi
}

# Extract model size and training samples
MODEL_SIZE=$(get_model_size)
TRAIN_SAMPLES=$(get_train_samples)

echo "========================================" 
echo "Training Monitor"
echo "========================================"
echo "Model: ${MODEL_SIZE}B"
echo "Samples: ${TRAIN_SAMPLES}"
echo ""

# Find latest log file in new structure
LOG_BASE="logs/pre_train_${MODEL_SIZE}b/${TRAIN_SAMPLES}_samples"

if [ ! -d "$LOG_BASE" ]; then
    echo "No training logs directory found: $LOG_BASE"
    echo ""
    echo "Looking for any recent logs..."
    latest_log=$(find logs/ -name "*.log" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
else
    # Find the most recent log file across all token counts
    latest_log=$(find "$LOG_BASE" -name "*.log" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
fi

if [ -z "$latest_log" ]; then
    echo "❌ No training logs found"
    echo ""
    echo "Expected location: $LOG_BASE"
    exit 1
fi

echo "📄 Monitoring: $latest_log"
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo "========================================"
echo ""

# Show training progress with relevant keywords
tail -f "$latest_log" | grep --line-buffered -E "(loss:|eval_loss:|epoch|step|Stage [0-9]|ERROR|✓|✗|🎯|Trainable|GPU)"
