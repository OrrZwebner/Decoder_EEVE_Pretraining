#!/bin/bash
# Test script to verify model size extraction fix

CONFIG_TEMPLATE="gemma_config_template.yaml"

echo "Testing model size extraction..."
echo "================================"
echo ""

# Old (broken) method
echo "OLD METHOD (broken - matches commented lines):"
model_line_old=$(grep -A 1 "^model:" "$CONFIG_TEMPLATE" | grep "name:")
model_size_old=$(echo "$model_line_old" | sed -n 's/.*gemma-3-\([0-9]\+\)b.*/\1/p')
echo "  Matched line: $model_line_old"
echo "  Extracted size: ${model_size_old}B"
echo ""

# New (fixed) method
echo "NEW METHOD (fixed - excludes commented lines):"
model_line_new=$(grep -A 10 "^model:" "$CONFIG_TEMPLATE" | grep -v "^[[:space:]]*#" | grep "name:" | head -1)
model_size_new=$(echo "$model_line_new" | sed -n 's/.*gemma-3-\([0-9]\+\)b.*/\1/p')
echo "  Matched line: $model_line_new"
echo "  Extracted size: ${model_size_new}B"
echo ""

# Verify
echo "================================"
if [ "$model_size_new" = "4" ]; then
    echo "✅ SUCCESS: Correctly extracted 4B model"
else
    echo "❌ FAILED: Expected 4B, got ${model_size_new}B"
fi
