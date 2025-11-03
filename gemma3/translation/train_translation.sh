#!/bin/bash
# Translation Fine-tuning Training Script
# Follows the same logging methodology as main pretraining script

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_FILE="configs/train_config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Function to extract base model name from config
# Returns: Model name (e.g., "gemma-3-1b-it", "local_4b", etc.)
get_base_model() {
    local base_model=$(grep "base_model:" "$CONFIG_FILE" | grep -v "#" | head -1 | sed 's/.*base_model: *"\?\([^"]*\)"\?.*/\1/')

    # If it's a local path, extract meaningful info
    if [[ "$base_model" == /* ]]; then
        # Check if it contains model size info
        if [[ "$base_model" =~ [0-9]+b ]]; then
            # Extract size (e.g., "1b", "4b")
            local size=$(echo "$base_model" | grep -o '[0-9]\+b' | head -1)
            echo "local_${size}"
        else
            # Just use "local" as identifier
            echo "local_model"
        fi
    else
        # HuggingFace model - extract model name
        # e.g., "google/gemma-3-1b-it" -> "gemma-3-1b"
        echo "$base_model" | sed 's|.*/||' | sed 's/-it$//'
    fi
}

# Function to extract model size (1, 4, 12, etc.) from base_model
get_model_size() {
    local base_model=$(grep "base_model:" "$CONFIG_FILE" | grep -v "#" | head -1 | sed 's/.*base_model: *"\?\([^"]*\)"\?.*/\1/')

    # Extract size from model name (e.g., "gemma-3-1b-it" -> "1")
    if [[ "$base_model" =~ ([0-9]+)b ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        # Default to 1 if can't determine
        echo "1"
    fi
}

# Function to extract training samples from config
get_train_samples() {
    local samples=$(grep "train_samples:" "$CONFIG_FILE" | grep -v "#" | head -1 | sed 's/.*train_samples: *\([0-9]*\).*/\1/')

    if [ -z "$samples" ] || [ "$samples" = "0" ]; then
        echo "1000"  # Default
    else
        echo "$samples"
    fi
}

# Function to extract language pair from config
get_language_pair() {
    local source_lang=$(grep "source_lang:" "$CONFIG_FILE" | grep -v "#" | head -1 | sed 's/.*source_lang: *"\?\([^"]*\)"\?.*/\1/')
    local target_lang=$(grep "target_lang:" "$CONFIG_FILE" | grep -v "#" | head -1 | sed 's/.*target_lang: *"\?\([^"]*\)"\?.*/\1/')

    # Convert to short codes (first 3 letters, lowercase)
    local src_short=$(echo "$source_lang" | cut -c1-3 | tr '[:upper:]' '[:lower:]')
    local tgt_short=$(echo "$target_lang" | cut -c1-3 | tr '[:upper:]' '[:lower:]')

    echo "${src_short}_${tgt_short}"
}

# Function to check if config file exists
check_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}Error: Config file not found: ${CONFIG_FILE}${NC}"
        echo -e "${YELLOW}Please create the config file or specify correct path${NC}"
        exit 1
    fi
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Translation Fine-tuning Training${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if config file exists
check_config

# Extract configuration parameters
BASE_MODEL=$(get_base_model)
MODEL_SIZE=$(get_model_size)
TRAIN_SAMPLES=$(get_train_samples)
LANG_PAIR=$(get_language_pair)

echo -e "${YELLOW}Configuration:${NC}"
echo "  Config file: ${CONFIG_FILE}"
echo "  Base model: ${BASE_MODEL}"
echo "  Model size: ${MODEL_SIZE}B"
echo "  Training samples: ${TRAIN_SAMPLES}"
echo "  Language pair: ${LANG_PAIR}"
echo ""

# Create log directory structure following main training convention:
# logs/translation_{model_size}b/{samples}_samples/{lang_pair}/
LOG_DIR="logs/translation_${MODEL_SIZE}b/${TRAIN_SAMPLES}_samples/${LANG_PAIR}"
mkdir -p "$LOG_DIR"

# Generate log filename: {samples}_{lang_pair}_{timestamp}.log
timestamp=$(date +%d%m%Y_%H%M)
LOG_FILE="${LOG_DIR}/${TRAIN_SAMPLES}_${LANG_PAIR}_${timestamp}.log"

echo -e "${GREEN}Log directory: ${LOG_DIR}${NC}"
echo -e "${GREEN}Log file: ${LOG_FILE}${NC}"
echo ""

# Display command that will be executed
echo -e "${YELLOW}Command:${NC}"
echo "  nohup python -u train_translation.py --config ${CONFIG_FILE} > ${LOG_FILE} 2>&1 &"
echo ""

# Confirm before starting
echo -e "${YELLOW}Start training? (y/n)${NC}"
read -r response

if [[ "$response" != "y" ]]; then
    echo -e "${RED}Training cancelled${NC}"
    exit 0
fi

# Start training in background with nohup
echo -e "${GREEN}Starting training...${NC}"
nohup python -u train_translation.py --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
TRAINING_PID=$!

# Save PID to file for monitoring/stopping
PID_FILE="training_translation_${LANG_PAIR}.pid"
echo $TRAINING_PID > "$PID_FILE"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Training Started Successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${YELLOW}Process ID:${NC} $TRAINING_PID"
echo -e "${YELLOW}PID file:${NC} $PID_FILE"
echo -e "${YELLOW}Log file:${NC} $LOG_FILE"
echo ""
echo -e "${BLUE}Monitoring commands:${NC}"
echo "  # Watch log in real-time:"
echo "    tail -f $LOG_FILE"
echo ""
echo "  # Check if training is running:"
echo "    ps -p $TRAINING_PID"
echo ""
echo "  # Kill training if needed:"
echo "    kill $TRAINING_PID"
echo "    # or: kill \$(cat $PID_FILE)"
echo ""
echo "  # Check training progress:"
echo "    grep -E '(Epoch|loss|Train|Eval)' $LOG_FILE | tail -20"
echo ""
echo -e "${GREEN}Training is now running in background${NC}"
echo -e "${YELLOW}Tip: Use 'tail -f $LOG_FILE' to monitor progress${NC}"
echo ""

# Append command info to log file
echo "========================================" >> "$LOG_FILE"
echo "Translation Fine-tuning Training" >> "$LOG_FILE"
echo "Started at: $(date)" >> "$LOG_FILE"
echo "PID: $TRAINING_PID" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "Configuration:" >> "$LOG_FILE"
echo "  Config: $CONFIG_FILE" >> "$LOG_FILE"
echo "  Base model: $BASE_MODEL" >> "$LOG_FILE"
echo "  Model size: ${MODEL_SIZE}B" >> "$LOG_FILE"
echo "  Training samples: $TRAIN_SAMPLES" >> "$LOG_FILE"
echo "  Language pair: $LANG_PAIR" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "Command: python -u train_translation.py --config $CONFIG_FILE" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"