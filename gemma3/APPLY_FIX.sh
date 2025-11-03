#!/bin/bash
# Script to apply the model size extraction fix

echo "================================================"
echo "Applying Model Size Extraction Fix"
echo "================================================"
echo ""

# Check if backup already exists
if [ -f "train_multiple_tokens.sh.old" ]; then
    echo "⚠️  Backup already exists: train_multiple_tokens.sh.old"
    echo "   Skipping backup creation"
else
    echo "📦 Creating backup..."
    cp train_multiple_tokens.sh train_multiple_tokens.sh.old
    echo "   ✅ Backup created: train_multiple_tokens.sh.old"
fi

echo ""
echo "🔧 Applying fix..."
cp train_multiple_tokens_FIXED.sh train_multiple_tokens.sh
chmod +x train_multiple_tokens.sh

echo "   ✅ Fix applied to train_multiple_tokens.sh"
echo ""
echo "🧪 Testing fix..."
CONFIG_TEMPLATE="gemma_config_template.yaml"
source train_multiple_tokens.sh

MODEL_SIZE=$(get_model_size)
echo ""
echo "================================================"
if [ "$MODEL_SIZE" = "4" ]; then
    echo "✅ SUCCESS: Model size correctly extracted as ${MODEL_SIZE}B"
    echo ""
    echo "The script will now correctly log:"
    echo "  - Model size: 4B"
    echo "  - Log path: logs/pre_train_4b/..."
    echo ""
    echo "You can now use train_multiple_tokens.sh normally."
else
    echo "❌ FAILED: Expected 4B, got ${MODEL_SIZE}B"
    echo ""
    echo "Restoring backup..."
    cp train_multiple_tokens.sh.old train_multiple_tokens.sh
fi
echo "================================================"
