# Translation Training Commands

## Summary

### 1. Dataset Download Status
The NLLB-200 dataset download script encountered issues:
- **English-Hebrew**: Would work, but had `trust_remote_code` parameter issue
- **English-Sanskrit**: Not available as a language pair in NLLB-200
- **English-Tibetan**: Available as `bod_Tibt-eng_Latn` (reverse direction)

**Note**: The current training configuration uses **local dataset fallback to streaming**, so you can train directly without downloading! The system will stream the needed samples automatically.

### 2. Training Scripts Created

Two training scripts are available:

#### Option 1: Interactive Script (`train_translation.sh`)
- Prompts for confirmation
- Shows all parameters
- Follows main pretraining logging methodology
- Saves PID for monitoring

#### Option 2: Quick Start Script (`start_training.sh`)
- Minimal prompts
- Fast execution
- Same logging structure

---

## Quick Start: Run Training Now

### English → Hebrew Translation

```bash
cd /home/orrz/gpufs/projects/gemma3/translation

# Option 1: Using interactive script
./train_translation.sh

# Option 2: Using quick start
./start_training.sh

# Option 3: Direct nohup command
nohup python -u train_translation.py --config configs/train_config.yaml \
  > logs/translation_1b/1000_samples/eng_heb/1000_eng_heb_$(date +%d%m%Y_%H%M).log 2>&1 &
```

---

## Log Directory Structure

Following the main training methodology, logs are organized as:

```
logs/translation_{model_size}b/{samples}_samples/{lang_pair}/{samples}_{lang_pair}_{timestamp}.log
```

### Examples:

**For English-Hebrew with 1000 samples and 1B model:**
```
logs/translation_1b/1000_samples/eng_heb/1000_eng_heb_29102025_1445.log
```

**For English-Hebrew with 10000 samples and 4B model:**
```
logs/translation_4b/10000_samples/eng_heb/10000_eng_heb_29102025_1450.log
```

---

## Manual nohup Commands

### Template

```bash
nohup python -u train_translation.py --config CONFIG_FILE \
  > logs/translation_{MODEL_SIZE}b/{SAMPLES}_samples/{LANG_PAIR}/{SAMPLES}_{LANG_PAIR}_TIMESTAMP.log 2>&1 &
```

### For Different Configurations

#### 1. Base Gemma-3-1B, 1K samples, English→Hebrew

```bash
# Create log directory
mkdir -p logs/translation_1b/1000_samples/eng_heb

# Run training
nohup python -u train_translation.py --config configs/train_config.yaml \
  > logs/translation_1b/1000_samples/eng_heb/1000_eng_heb_$(date +%d%m%Y_%H%M).log 2>&1 &

# Save PID
echo $! > training_eng_heb.pid

# Monitor
tail -f logs/translation_1b/1000_samples/eng_heb/1000_eng_heb_*.log
```

#### 2. Base Gemma-3-4B, 10K samples, English→Hebrew

```bash
# Update config first to use 4B model and 10K samples
# Then run:

mkdir -p logs/translation_4b/10000_samples/eng_heb

nohup python -u train_translation.py --config configs/train_config.yaml \
  > logs/translation_4b/10000_samples/eng_heb/10000_eng_heb_$(date +%d%m%Y_%H%M).log 2>&1 &

echo $! > training_eng_heb_4b.pid
```

#### 3. Local Fine-tuned Model (from pretraining), 5K samples

```bash
# Assuming you have a pretrained Hebrew model at:
# /home/orrz/gpufs/projects/gemma3/outputs/hebrew/16384_samples/two_stages/stage2/merged_model

# Update config:
# model:
#   base_model: "/home/orrz/gpufs/projects/gemma3/outputs/hebrew/16384_samples/two_stages/stage2/merged_model"
# translation:
#   train_samples: 5000

mkdir -p logs/translation_4b/5000_samples/eng_heb

nohup python -u train_translation.py --config configs/train_config.yaml \
  > logs/translation_4b/5000_samples/eng_heb/5000_eng_heb_$(date +%d%m%Y_%H%M).log 2>&1 &

echo $! > training_eng_heb_local.pid
```

---

## Configuration Examples

### Current Config (configs/train_config.yaml)

```yaml
model:
  base_model: "google/gemma-3-1b-it"  # 1B model

translation:
  source_lang: "English"
  target_lang: "Hebrew"
  train_samples: 1000  # Quick test
  eval_samples: 200

  # Local dataset (optional - will fall back to streaming if not found)
  local_dataset_path: "/home/orrz/gpufs/projects/gemma3/data/nllb_translation/all_pairs.jsonl"

environment:
  cuda_devices: "2"  # GPU 2
  seed: 42
```

### For 4B Model with More Samples

```yaml
model:
  base_model: "google/gemma-3-4b-it"  # Change to 4B

translation:
  train_samples: 10000  # More samples
  eval_samples: 2000
```

### For Local Pretrained Model

```yaml
model:
  # Use your pretrained Hebrew model
  base_model: "/home/orrz/gpufs/projects/gemma3/outputs/hebrew/16384_samples/two_stages/stage2/merged_model"

translation:
  train_samples: 5000
```

---

## Monitoring Commands

### Check if Training is Running

```bash
# Check by PID file
ps -p $(cat training_eng_heb.pid)

# Or search for process
ps aux | grep train_translation.py
```

### Monitor Log in Real-time

```bash
# Find latest log
LOG=$(ls -t logs/translation_1b/1000_samples/eng_heb/*.log | head -1)

# Watch it
tail -f $LOG

# Or grep for important info
tail -f $LOG | grep -E "(Epoch|loss|Learning|Step|Training|Eval)"
```

### Check Training Progress

```bash
# Show recent training metrics
grep -E "(Epoch|Step.*loss)" $LOG | tail -20

# Check if completed
grep -E "(Training Complete|DONE|Merged model saved)" $LOG
```

---

## Stopping Training

```bash
# Using PID file
kill $(cat training_eng_heb.pid)

# Or find and kill
pkill -f train_translation.py

# Force kill if needed
kill -9 $(cat training_eng_heb.pid)
```

---

## Output Structure

After training completes, you'll find:

```
outputs/translation/gemma3-1b_eng-heb_YYYYMMDD_HHMM/
├── lora_adapter/          # LoRA weights only (~50MB)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files
└── merged_model/          # Full merged model (~2GB)
    ├── config.json
    ├── pytorch_model.bin  # or model.safetensors
    └── tokenizer files
```

**Use `merged_model/` for evaluation** with the EVALUATION pipeline.

---

## Troubleshooting

### Log File Not Created

```bash
# Make sure log directory exists
mkdir -p logs/translation_1b/1000_samples/eng_heb

# Check permissions
ls -la logs/
```

### Training Not Starting

```bash
# Check if config file exists
ls -la configs/train_config.yaml

# Verify Python environment
which python
python --version

# Test config loading
python -c "from utils import load_config; print(load_config('configs/train_config.yaml'))"
```

### CUDA Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 2  # Reduce from 4 or 8
  gradient_accumulation: 8  # Increase to maintain effective batch size
```

---

## Example: Complete Training Session

```bash
# 1. Navigate to translation directory
cd /home/orrz/gpufs/projects/gemma3/translation

# 2. Check configuration
cat configs/train_config.yaml | grep -E "(base_model|train_samples|source_lang|target_lang)"

# 3. Create log directory
mkdir -p logs/translation_1b/1000_samples/eng_heb

# 4. Start training
nohup python -u train_translation.py --config configs/train_config.yaml \
  > logs/translation_1b/1000_samples/eng_heb/1000_eng_heb_$(date +%d%m%Y_%H%M).log 2>&1 &

# 5. Save PID
echo $! > training.pid
PID=$(cat training.pid)

# 6. Monitor
echo "Training started with PID: $PID"
echo "Log file: $(ls -t logs/translation_1b/1000_samples/eng_heb/*.log | head -1)"
tail -f $(ls -t logs/translation_1b/1000_samples/eng_heb/*.log | head -1)
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start training | `./start_training.sh` or `./train_translation.sh` |
| Monitor log | `tail -f logs/translation_1b/1000_samples/eng_heb/*.log` |
| Check if running | `ps -p $(cat training.pid)` |
| Stop training | `kill $(cat training.pid)` |
| Check output | `ls -lh outputs/translation/*/merged_model/` |
| Evaluate model | Use EVALUATION pipeline with `merged_model/` path |

---

## Notes

1. **No Dataset Download Needed**: The training code automatically streams from HuggingFace if local dataset isn't found
2. **GPUFS-Safe**: Config uses `save_strategy: "no"` to minimize I/O
3. **Logging**: Follows exact same structure as main pretraining pipeline
4. **Model Size Detection**: Automatically extracts from base_model path
5. **Merged Model**: Both LoRA adapter and merged model are saved automatically