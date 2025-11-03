#!/bin/bash
# Monitor current training progress

# Find latest log file
latest_log=$(ls -t logs/pre_train_1b/sanskrit/gemma_sanskrit_spm_bpe_*.log 2>/dev/null | head -1)

if [ -z "$latest_log" ]; then
    echo "No training logs found"
    exit 1
fi

echo "Monitoring: $latest_log"
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo "----------------------------------------"

# Show training progress
tail -f "$latest_log" | grep --line-buffered -E "(loss:|eval_loss:|epoch|step|Stage \d)"