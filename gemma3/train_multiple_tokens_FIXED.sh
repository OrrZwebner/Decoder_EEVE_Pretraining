#!/bin/bash
# Sequential training with different token counts
# New log structure: logs/pre_train_Xb/N_samples/n_tokens/N_n_timestamp.log

# Configuration
# TOKEN_COUNTS=(4096 8192 16384)  # List of token counts to train
# TOKEN_COUNTS=(256 512 1024)  # List of token counts to train
# TOKEN_COUNTS=(32768)  # List of token counts to train
TOKEN_COUNTS=(32 64 128 256 512 1024 2048 4096 8192 16384)  # List of token counts to train
# TOKEN_COUNTS=(4096 8192 16384 32768)  # List of token counts to train

# CONFIG_TEMPLATE="gemma_config_template_2.yaml"
# CONFIG_TEMPLATE="gemma_config_template_1.yaml"
CONFIG_TEMPLATE="gemma_config_template.yaml"
CONFIG_DIR="configs/token_configs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to extract model size from config template
# Returns: 1, 4, or 12 (the number before 'b' in gemma-3-Xb)
get_model_size() {
    # Use grep to find the model section, then filter out commented lines (lines starting with #)
    # This ensures we only match the active configuration line, not commented alternatives
    local model_line=$(grep -A 10 "^model:" "$CONFIG_TEMPLATE" | grep -v "^[[:space:]]*#" | grep "name:" | head -1)
    local model_size=$(echo "$model_line" | sed -n 's/.*gemma-3-\([0-9]\+\)b.*/\1/p')
    echo "$model_size"
}

# Function to extract training samples from config template
# Returns: number of samples when test_mode=true, "full" otherwise
get_train_samples() {
    # Check if test_mode is true (excluding commented lines)
    local test_mode_line=$(grep "^  test_mode:" "$CONFIG_TEMPLATE" | grep -v "#.*test_mode:" | tail -1)
    local test_mode=$(echo "$test_mode_line" | sed 's/.*: *\(true\|false\).*/\1/')
    
    if [ "$test_mode" = "true" ]; then
        # Extract limit_train_samples value (excluding commented lines)
        local samples=$(grep "^  limit_train_samples:" "$CONFIG_TEMPLATE" | grep -v "#" | tail -1 | sed 's/.*limit_train_samples: *\([0-9]*\).*/\1/')
        echo "$samples"
    else
        # If test_mode is false, return "full" to indicate unlimited
        echo "full"
    fi
}

# Extract model size and training samples from config
MODEL_SIZE=$(get_model_size)
TRAIN_SAMPLES=$(get_train_samples)

if [ -z "$MODEL_SIZE" ]; then
    echo -e "${RED}Error: Could not extract model size from config${NC}"
    echo "Check that gemma_config_template.yaml contains model name like 'gemma-3-4b-it'"
    exit 1
fi

if [ -z "$TRAIN_SAMPLES" ]; then
    echo -e "${YELLOW}Warning: Could not determine training samples, defaulting to 'full'${NC}"
    TRAIN_SAMPLES="full"
fi

echo ""
echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "  Model size: ${MODEL_SIZE}B"
echo "  Training samples: ${TRAIN_SAMPLES}"
echo "  Token counts: ${TOKEN_COUNTS[@]}"
echo "========================================"
echo ""

# Create config directory
mkdir -p $CONFIG_DIR

# Function to generate config for specific token count
generate_config() {
    local token_count=$1
    local config_file="${CONFIG_DIR}/gemma_config_${token_count}.yaml"
    
    echo -e "${YELLOW}Generating config for ${token_count} tokens...${NC}"
    
    # Copy template and replace placeholders
    cp $CONFIG_TEMPLATE $config_file
    
    # Replace TOKEN_COUNT placeholder
    sed -i "s/TOKEN_COUNT/${token_count}/g" $config_file
    
    # Update run name with token count
    sed -i "s/run_name: \"gemma3_sanskrit_eeve\"/run_name: \"gemma3_sanskrit_eeve_${token_count}\"/g" $config_file
    
    # update num_tokens in vocabulary_generation
    sed -i "s/num_tokens: .*/num_tokens: ${token_count}/g" $config_file
    
    # Update wandb run name
    sed -i "s/wandb_run_name: \"gemma3-1b-sanskrit-local\"/wandb_run_name: \"gemma3-${MODEL_SIZE}b-${TRAIN_SAMPLES}samples-${token_count}tokens\"/g" $config_file
    
    echo -e "${GREEN}Config generated: ${config_file}${NC}"
    return 0
}

# Function to wait for training to complete
wait_for_completion() {
    local pid=$1
    local token_count=$2
    local log_file=$3
    
    echo -e "${YELLOW}Training with ${token_count} tokens started (PID: $pid)${NC}"
    echo -e "${YELLOW}Log file: ${log_file}${NC}"
    echo -e "${YELLOW}Waiting for completion...${NC}"
    
    # Wait for the process to finish and capture its exit code
    wait $pid
    local status=$?
    
    # Check the log file for the success message
    if grep -q -E "(Script reached the end of main function, exiting cleanly|ALL TRAINING COMPLETED SUCCESSFULLY)" "$log_file" 2>/dev/null; then
        echo -e "${GREEN}✓ Training with ${token_count} tokens completed successfully${NC}"
        return 0
    elif [ $status -ne 0 ]; then
        echo -e "${RED}✗ Training with ${token_count} tokens failed (Process exited with status $status)${NC}"
        return 1
    else
        echo -e "${RED}✗ Training with ${token_count} tokens failed or was interrupted (Success message not found in log)${NC}"
        return 1
    fi
}

# Main training loop
echo "Starting sequential training pipeline..."
echo ""

for token_count in "${TOKEN_COUNTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training: ${token_count} tokens"
    echo "========================================"
    
    # Generate config for this token count
    generate_config $token_count
    config_file="${CONFIG_DIR}/gemma_config_${token_count}.yaml"
    
    # Verify config was created
    if [ ! -f "$config_file" ]; then
        echo -e "${RED}Error: Failed to generate config: ${config_file}${NC}"
        continue
    fi

    # Create log directory structure: logs/pre_train_Xb/N_samples/n_tokens/
    LOG_DIR="logs/pre_train_${MODEL_SIZE}b/${TRAIN_SAMPLES}_samples/${token_count}_tokens"
    mkdir -p "$LOG_DIR"
    
    # Generate log filename: N_n_timestamp.log
    timestamp=$(date +%d%m%Y_%H%M)
    log_file="${LOG_DIR}/${TRAIN_SAMPLES}_${token_count}_${timestamp}.log"

    echo -e "${GREEN}Log directory: ${LOG_DIR}${NC}"
    echo -e "${GREEN}Log file: $(basename ${log_file})${NC}"
    echo -e "${GREEN}Config: ${config_file}${NC}"
    echo ""

    # Start training with specific config file
    nohup python -u src/train.py --config "$config_file" > "$log_file" 2>&1 &
    training_pid=$!
    
    echo "Command: python -u src/train.py --config $config_file" >> "$log_file"
    echo "Started at: $(date)" >> "$log_file"

    # Save PID to file for emergency stopping
    echo $training_pid > "training_${token_count}.pid"
    
    # Wait for this training to complete
    if wait_for_completion $training_pid $token_count "$log_file"; then
        echo -e "${GREEN}Moving to next token count...${NC}"

        # Add experiment to tracking system
        echo -e "${YELLOW}Adding experiment to tracking system...${NC}"
        if python scripts/experiment_tracker.py --log "$log_file" 2>&1 | tee -a "$log_file"; then
            echo -e "${GREEN}✓ Experiment tracked successfully${NC}"
        else
            echo -e "${YELLOW}⚠ Warning: Failed to track experiment (training succeeded but tracking failed)${NC}"
        fi

        # Optional: Move model checkpoint to named directory
        if [ -d "outputs/eeve_stages/gemma3_sanskrit_eeve_stage4" ]; then
            final_model_dir="outputs/final_models/gemma3_${MODEL_SIZE}b_${TRAIN_SAMPLES}samples_${token_count}tokens"
            mkdir -p "$final_model_dir"
            cp -r outputs/eeve_stages/gemma3_sanskrit_eeve_stage4/* "$final_model_dir/"
            echo -e "${GREEN}Model saved to: ${final_model_dir}${NC}"
        fi
    else
        echo -e "${RED}Training failed. Check log: ${log_file}${NC}"
        echo -e "${YELLOW}Continue with next token count? (y/n)${NC}"
        read -r response
        if [[ "$response" != "y" ]]; then
            break
        fi
    fi
    
    # Clean up PID file
    rm -f "training_${token_count}.pid"
    
    echo -e "${GREEN}Completed ${token_count} tokens${NC}"
    sleep 10  # Brief pause between runs
done

echo ""
echo "========================================"
echo "All training runs completed!"
echo "========================================"

# Summary
echo ""
echo "Training Summary:"
echo "========================================"
for token_count in "${TOKEN_COUNTS[@]}"; do
    LOG_DIR="logs/pre_train_${MODEL_SIZE}b/${TRAIN_SAMPLES}_samples/${token_count}_tokens"
    log_pattern="${LOG_DIR}/${TRAIN_SAMPLES}_${token_count}_*.log"
    if ls $log_pattern 1> /dev/null 2>&1; then
        latest_log=$(ls -t $log_pattern | head -1)
        if grep -q -E "(Script reached the end of main function, exiting cleanly|ALL TRAINING COMPLETED SUCCESSFULLY)" "$latest_log" 2>/dev/null; then
            echo -e "${GREEN}✓ ${token_count} tokens: SUCCESS${NC}"
            echo "   Log: $latest_log"
        else
            echo -e "${RED}✗ ${token_count} tokens: FAILED/INCOMPLETE${NC}"
            echo "   Log: $latest_log"
        fi
    else
        echo -e "${YELLOW}○ ${token_count} tokens: NOT RUN${NC}"
    fi
done
echo "========================================"
