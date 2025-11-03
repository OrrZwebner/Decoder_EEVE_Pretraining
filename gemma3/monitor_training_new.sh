#!/bin/bash
# Monitor current training progress with new log structure

# Function to extract model size from config template
get_model_size() {
    local model_name=$(grep "^  name:" "gemma_config_template.yaml" | head -1 | sed 's/.*gemma-3-\([0-9]*\)b.*/\1/')
    echo "$model_name"
}

# Function to extract training samples from config template
get_train_samples() {
    local test_mode=$(grep "test_mode:" "gemma_config_template.yaml" | grep -v "#" | tail -1 | grep -o "true\|false")
    
    if [ "$test_mode" = "true" ]; then
        local samples=$(grep "limit_train_samples:" "gemma_config_template.yaml" | grep -v "#" | tail -1 | sed 's/.*limit_train_samples: *\([0-9]*\).*/\1/')
        echo "$samples"
    else
        echo "full"
    fi
}

# Extract model size and training samples
MODEL_SIZE=$(get_model_size)
TRAIN_SAMPLES=$(get_train_samples)

echo "Looking for logs: Model ${MODEL_SIZE}B, ${TRAIN_SAMPLES} samples"

# Find latest log file in new structure
# logs/pre_train_Xb/N_samples/*/N_n_*.log
LOG_BASE="logs/pre_train_${MODEL_SIZE}b/${TRAIN_SAMPLES}_samples"

if [ ! -d "$LOG_BASE" ]; then
    echo "No training logs directory found: $LOG_BASE"
    echo "Looking for any logs in logs/ directory..."
    latest_log=$(find logs/ -name "*.log" -type f | xargs ls -t 2>/dev/null | head -1)
else
    # Find the most recent log file
    latest_log=$(find "$LOG_BASE" -name "*.log" -type f | xargs ls -t 2>/dev/null | head -1)
fi

if [ -z "$latest_log" ]; then
    echo "No training logs found"
    exit 1
fi

echo "Monitoring: $latest_log"
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo "----------------------------------------"

# Show training progress
tail -f "$latest_log" | grep --line-buffered -E "(loss:|eval_loss:|epoch|step|Stage [0-9]|ERROR|✓|✗)"
