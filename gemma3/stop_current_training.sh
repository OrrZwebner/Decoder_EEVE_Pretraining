#!/bin/bash
# Emergency stop script for current training

# Find and kill current training process
if ls training_*.pid 1> /dev/null 2>&1; then
    for pid_file in training_*.pid; do
        pid=$(cat $pid_file)
        echo "Stopping training process $pid..."
        kill -TERM $pid 2>/dev/null || echo "Process already stopped"
        rm $pid_file
    done
else
    echo "No active training found"
    pkill -f "python src/train.py"
fi