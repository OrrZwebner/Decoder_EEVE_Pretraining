#!/bin/bash
# Sequential training with different token counts
# Each training starts after the previous one completes

# Configuration
# TOKEN_COUNTS=(128 256 512 1024 2048 4096)  # List of token counts to train
# TOKEN_COUNTS=(128 256 512 1024)  # List of token counts to train
TOKEN_COUNTS=(16384 32768)  # List of token counts to train
# TOKEN_COUNTS=(32 64 128 256 512 1024 2048 4096 8192 16384 32768)  # List of token counts to train
# TOKEN_COUNTS=(32)  # List of token counts to train
# ALGORITHM="sentencepiece_bpe"  # or get from command line
BASE_LOG_DIR="logs/pre_train_1b/sanskrit"
# VOCAB_BASE_PATH="/home/orrz/gpufs/projects/Tokenizers/vocabularies/sanskrit/spm_bpe"
CONFIG_TEMPLATE="gemma_config_template.yaml"
CONFIG_DIR="configs/token_configs"
# CONFIG_BACKUP="gemma_config_backup.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create log directory if it doesn't exist
mkdir -p $BASE_LOG_DIR
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
    
    # Replace ALGORITHM placeholder
    # sed -i "s/ALGORITHM/${ALGORITHM}/g" $config_file
    
    # Update run name with token count
    sed -i "s/run_name: \"gemma3_sanskrit_eeve\"/run_name: \"gemma3_sanskrit_eeve_${token_count}\"/g" $config_file
    

    # update num_tokens in vocabulary_generation
    sed -i "s/num_tokens: .*/num_tokens: ${token_count}/g" $config_file
    
    # Update wandb run name
    sed -i "s/wandb_run_name: \"gemma3-1b-sanskrit-local\"/wandb_run_name: \"gemma3-sanskrit-${token_count}tokens\"/g" $config_file
    
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
    
    # Wait for the process to finish and capture its exit code.
    # This is more efficient than the while loop.
    wait $pid
    local status=$?
    
    # First, check the log file for the success message.
    # if grep -q "Standard training completed" "$log_file" 2>/dev/null; then
    if grep -q -E "(Script reached the end of main function, exiting cleanly|ALL TRAINING COMPLETED SUCCESSFULLY)" "$log_file" 2>/dev/null; then
        echo -e "${GREEN}✓ Training with ${token_count} tokens completed successfully${NC}"
        return 0
    # If the success message isn't there, check the exit code from 'wait'.
    elif [ $status -ne 0 ]; then
        echo -e "${RED}✗ Training with ${token_count} tokens failed (Process exited with status $status)${NC}"
        return 1
    # Fallback for any other unexpected case.
    else
        echo -e "${RED}✗ Training with ${token_count} tokens failed or was interrupted (Success message not found in log)${NC}"
        return 1
    fi
}


# Main training loop
echo "========================================"
echo "Starting sequential training pipeline"
echo "Token counts: ${TOKEN_COUNTS[@]}"
# echo "Algorithm: ${ALGORITHM}"
echo "========================================"

for token_count in "${TOKEN_COUNTS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting training with ${token_count} tokens"
    echo "========================================"
    

    # Generate config for this token count
    generate_config $token_count
    config_file="${CONFIG_DIR}/gemma_config_${token_count}.yaml"
    
    # Verify config was created
    if [ ! -f "$config_file" ]; then
        echo -e "${RED}Error: Failed to generate config: ${config_file}${NC}"
        continue
    fi

    # Generate log filename with timestamp
    timestamp=$(date +%d%m%Y_%H%M)
    log_file="${BASE_LOG_DIR}/gemma_sanskrit_spm_bpe_${token_count}_${timestamp}.log"

    echo -e "${GREEN}Starting training with config: ${config_file}${NC}"

    # echo -e "${GREEN}Starting training...${NC}"
    # nohup python -u src/train.py > "$log_file" 2>&1 &
    # Start training with specific config file
    nohup python -u src/train.py --config "$config_file" > "$log_file" 2>&1 &
    training_pid=$!
    
    echo "Command: python -u src/train.py --config $config_file" >> "$log_file"

    # Save PID to file for emergency stopping
    echo $training_pid > "training_${token_count}.pid"
    
    # Wait for this training to complete
    if wait_for_completion $training_pid $token_count "$log_file"; then
        echo -e "${GREEN}Moving to next token count...${NC}"
        
        # Optional: Move model checkpoint to named directory
        if [ -d "outputs/eeve_stages/gemma3_sanskrit_eeve_stage4" ]; then
            final_model_dir="outputs/final_models/gemma3_sanskrit_${token_count}_tokens"
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
    
    # Optional: Clean intermediate checkpoints to save space
    # rm -rf outputs/eeve_stages/gemma3_sanskrit_eeve_*
    
    echo -e "${GREEN}Completed ${token_count} tokens${NC}"
    sleep 10  # Brief pause between runs
done

# Restore original config
echo -e "${YELLOW}Restoring original config...${NC}"
cp $CONFIG_BACKUP $CONFIG_FILE

echo ""
echo "========================================"
echo "All training runs completed!"
echo "========================================"

# Summary
echo ""
echo "Training Summary:"
for token_count in "${TOKEN_COUNTS[@]}"; do
    log_pattern="${BASE_LOG_DIR}/gemma_sanskrit_spm_bpe_${token_count}_*.log"
    if ls $log_pattern 1> /dev/null 2>&1; then
        latest_log=$(ls -t $log_pattern | head -1)
        # if grep -q "Training completed successfully" "$latest_log" 2>/dev/null; then
        if grep -q -E "(Script reached the end of main function, exiting cleanly|ALL TRAINING COMPLETED SUCCESSFULLY)" "$latest_log" 2>/dev/null; then
            echo -e "${GREEN}✓ ${token_count} tokens: SUCCESS${NC}"
        else
            echo -e "${RED}✗ ${token_count} tokens: FAILED/INCOMPLETE${NC}"
        fi
    else
        echo -e "${YELLOW}○ ${token_count} tokens: NOT RUN${NC}"
    fi
done