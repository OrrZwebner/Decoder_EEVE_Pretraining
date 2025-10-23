#!/bin/bash

# =============================================================================
# COLORS
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# List of models (local paths or HuggingFace model names)
MODELS=(
    
    
    
)

# Source and target languages
SOURCE_LANG="English"
TARGET_LANG="Hebrew"

# GPU devices
DEVICES="0,1,2,3"

# Number of samples (null for all, or integer like 50)
NUM_SAMPLES="50"

# Config template and output directory
CONFIG_TEMPLATE="config_template.yaml"
CONFIG_DIR="configs/model_configs"

# =============================================================================
# SCRIPT START - DO NOT EDIT BELOW
# =============================================================================

# Navigate to project directory
cd /home/orrz/gpufs/projects/EVALUATION

# Create directories
mkdir -p logs
mkdir -p $CONFIG_DIR

# Function to wait for evaluation to complete
wait_for_completion() {
    local pid=$1
    local model=$2
    local log_file=$3
    
    echo -e "${YELLOW}Evaluation of ${model} started (PID: $pid)${NC}"
    echo -e "${YELLOW}Log file: ${log_file}${NC}"
    echo -e "${YELLOW}Waiting for completion...${NC}"
    
    # Wait for the process to finish
    wait $pid
    local status=$?
    
    # Check log for success message
    if grep -q "EVALUATION COMPLETE" "$log_file" 2>/dev/null; then
        echo -e "${GREEN}✓ Evaluation of ${model} completed successfully${NC}"
        return 0
    elif [ $status -ne 0 ]; then
        echo -e "${RED}✗ Evaluation of ${model} failed (Process exited with status $status)${NC}"
        return 1
    else
        echo -e "${RED}✗ Evaluation of ${model} failed or was interrupted${NC}"
        return 1
    fi
}

# Function to generate config for specific model
generate_config() {
    local model=$1
    local timestamp=$2
    
    # Create safe filename from model name
    local safe_model_name=$(echo "$model" | tr '/' '_' | tr ':' '_')
    local config_file="${CONFIG_DIR}/config_${safe_model_name}_${timestamp}.yaml"
    
    echo -e "${YELLOW}Generating config for ${model}...${NC}" >&2
    
    # Copy template and replace placeholders
    cp $CONFIG_TEMPLATE $config_file
    
# Replace placeholders using sed
    sed -i "s|MODEL_NAME|${model}|g" $config_file
    sed -i "s|SOURCE_LANG|${SOURCE_LANG}|g" $config_file
    sed -i "s|TARGET_LANG|${TARGET_LANG}|g" $config_file
    sed -i "s|DEVICES|${DEVICES}|g" $config_file
    
    # Handle NUM_SAMPLES: if it's "null" or empty, leave as null; otherwise replace with number
    if [ "$NUM_SAMPLES" = "null" ]; then
        sed -i "s|num_samples: null|num_samples: null|g" $config_file
    else
        sed -i "s|num_samples: null|num_samples: ${NUM_SAMPLES}|g" $config_file
    fi
    
    echo -e "${GREEN}Config generated: ${config_file}${NC}" >&2
    
    # Return only the filename
    echo "$config_file"
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Multi-Model Evaluation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Source Language: $SOURCE_LANG${NC}"
echo -e "${GREEN}Target Language: $TARGET_LANG${NC}"
echo -e "${GREEN}Number of Models: ${#MODELS[@]}${NC}"
echo -e "${GREEN}Devices: $DEVICES${NC}"
echo -e "${GREEN}Samples: $NUM_SAMPLES${NC}"
echo ""

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Processing Model: $MODEL${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Generate timestamp
    TIMESTAMP=$(date +%d%m%Y_%H%M)
    
    # Create safe model name for log file
    SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
    
    # Generate config for this model
    CONFIG_FILE=$(generate_config "$MODEL" "$TIMESTAMP")
    
    # Verify config was created
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Error: Failed to generate config: ${CONFIG_FILE}${NC}"
        continue
    fi
    
    # Generate log filename
    LOGFILE="logs/${SAFE_MODEL_NAME}_${SOURCE_LANG}-${TARGET_LANG}_${TIMESTAMP}.log"
    
    echo -e "${BLUE}Config file: $CONFIG_FILE${NC}"
    echo -e "${BLUE}Log file: $LOGFILE${NC}"
    echo -e "${GREEN}Starting evaluation...${NC}"
    
    # Run evaluation with generated config
    nohup python -u -m src.main --config "$CONFIG_FILE" > "$LOGFILE" 2>&1 &
    
    # Get PID
    PID=$!
    echo -e "${GREEN}✓ Started with PID: $PID${NC}"    
    # Wait for this evaluation to complete
    if wait_for_completion $PID "$MODEL" "$LOGFILE"; then
        echo -e "${GREEN}Moving to next model...${NC}"
    else
        echo -e "${RED}Evaluation failed. Check log: ${LOGFILE}${NC}"
        echo -e "${YELLOW}Continue with next model? (y/n)${NC}"
        read -r response
        if [[ "$response" != "y" ]]; then
            echo -e "${RED}Stopping evaluation pipeline${NC}"
            break
        fi
    fi
    
    
    echo -e "${GREEN}Completed ${MODEL}${NC}"
    echo ""
    sleep 5  # Brief pause between runs
    
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All evaluations completed!${NC}"
echo -e "${GREEN}========================================${NC}"

# Summary
echo ""
echo -e "${BLUE}Evaluation Summary:${NC}"
for MODEL in "${MODELS[@]}"; do
    SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
    log_pattern="logs/${SAFE_MODEL_NAME}_${SOURCE_LANG}-${TARGET_LANG}_*.log"
    if ls $log_pattern 1> /dev/null 2>&1; then
        latest_log=$(ls -t $log_pattern | head -1)
        if grep -q "EVALUATION COMPLETE" "$latest_log" 2>/dev/null; then
            echo -e "${GREEN}✓ ${MODEL}: SUCCESS${NC}"
        else
            echo -e "${RED}✗ ${MODEL}: FAILED/INCOMPLETE${NC}"
        fi
    else
        echo -e "${YELLOW}○ ${MODEL}: NOT RUN${NC}"
    fi
done

