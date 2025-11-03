# Translation Fine-tuning for Gemma-3

Fine-tune Gemma-3 models for neural machine translation using LoRA (Low-Rank Adaptation) for parameter-efficient training.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Scripts Description](#scripts-description)
- [Training Flow](#training-flow)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What This Does
Fine-tunes Gemma-3 (1B or 4B) models for translation tasks using:
- **Dataset**: NLLB-200 (high-quality parallel translations for 200 languages)
- **Method**: LoRA (trains only ~1% of parameters)
- **Result**: Lightweight adapter (~50MB) that can be loaded with base model

### Why LoRA?
- **Efficient**: Only trains 1-2% of model parameters
- **Fast**: 5-10 minutes for 1K samples
- **Flexible**: Can swap adapters for different language pairs
- **Reusable**: Keep base model, train multiple adapters

### Expected Results

| Samples | Training Time | BLEU Score | Quality |
|---------|--------------|------------|---------|
| 1,000   | 5-10 min     | 10-20      | Basic patterns, proof of concept |
| 5,000   | 20-30 min    | 20-30      | Decent quality, understandable |
| 10,000  | 40-60 min    | 25-35      | Good quality for most uses |
| 50,000  | 3-4 hours    | 30-40+     | High quality, near production |

---

## Quick Start

### 1. Edit Configuration

```bash
cd /home/orrz/gpufs/projects/gemma3/translation
nano configs/train_config.yaml
```

**Minimum required changes**:
```yaml
translation:
  source_lang: "English"      # Change to your source language (just the name!)
  target_lang: "Hebrew"       # Change to your target language (just the name!)
  train_samples: 1000         # Start with 1000 for testing

# Note: No need to specify source_code/target_code - they're auto-resolved!
# See "Supported Languages" section below for the full list of 80+ languages
```

### 2. Run Training

```bash
python train_translation.py --config configs/train_config.yaml
```

### 3. Find Output

Trained model will be saved in TWO formats:
```
outputs/translation/gemma3-1b_en-he_YYYYMMDD_HHMM/
├── lora_adapter/      # LoRA adapter only (~50MB) - for loading with PEFT
└── merged_model/      # Full merged model (~2GB) - ready to use directly
```

### 4. Evaluate (Optional)

```bash
cd /home/orrz/gpufs/projects/EVALUATION

# Edit config.yaml to point to your model
nano config.yaml

# Run evaluation
python -m src.main --config config.yaml
```

---

## Supported Languages

The translation system supports **80+ languages** from the NLLB-200 dataset. Simply specify language names in the config - no need to look up codes!

### Major Languages
- **English**, Chinese, Spanish, French, German, Japanese, Korean, Russian, Portuguese, Italian
- Dutch, Polish, Turkish, Vietnamese, Thai, Indonesian, Malay

### Middle Eastern Languages
- **Arabic**, **Hebrew**, Persian, Urdu

### South Asian Languages
- Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Nepali, Sinhala

### Southeast Asian Languages
- Burmese, Khmer, Lao, Tagalog, Cebuano

### European Languages
- Ukrainian, Czech, Swedish, Danish, Finnish, Norwegian, Hungarian, Greek, Romanian
- Bulgarian, Serbian, Croatian, Slovak, Lithuanian, Latvian, Estonian, Slovenian
- Albanian, Macedonian, Belarusian, Icelandic, Irish, Welsh, Scots Gaelic, Maltese

### African Languages
- Swahili, Amharic, Hausa, Yoruba, Igbo, Zulu, Xhosa, Somali, Afrikaans

### Other Languages
- Esperanto, Latin, Sanskrit

**See `utils.py`** for the complete mapping dictionary (`NLLB_LANGUAGE_CODES`) with all 80+ supported languages.

### Usage

Simply use the language name in your config:
```yaml
translation:
  source_lang: "English"    # No codes needed!
  target_lang: "Hebrew"
```

The system automatically resolves the NLLB-200 codes (e.g., `eng_Latn`, `heb_Hebr`).

---

## Project Structure

```
translation/
├── README.md                          # This file
├── QUICK_START.txt                    # Quick reference guide
├── train.py                           # Main training script (~467 lines)
├── utils.py                           # Data loading utilities (~225 lines)
├── translation_exploration.ipynb      # Jupyter notebook for testing & visualization
├── configs/
│   └── train_config.yaml              # Training configuration
└── outputs/
    └── translation/
        └── gemma3-1b_en-he_YYYYMMDD_HHMM/
            ├── checkpoint-250/        # Intermediate checkpoints
            ├── checkpoint-500/
            └── final_model/           # Final LoRA adapter
                ├── adapter_config.json
                ├── adapter_model.bin  # LoRA weights (~50 MB)
                ├── tokenizer.json
                └── ...
```

---

## Scripts Description

### 1. `train.py` - Main Training Script

**Purpose**: Orchestrates the complete fine-tuning process from loading model to saving adapters.

**Input**:
- `--config`: Path to YAML configuration file (default: `configs/train_config.yaml`)
- `--samples`: Override number of training samples (optional)
- `--model`: Override base model path (optional)

**Output**:
- LoRA adapter weights in `outputs/translation/[model]_[lang_pair]_[timestamp]/final_model/`
- Training logs and metrics
- Tokenizer files for inference

**Functions**:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `parse_args()` | Parse command line arguments | CLI args | Parsed arguments |
| `setup_environment()` | Setup CUDA devices and environment vars | Config dict | Device string |
| `load_model_and_tokenizer()` | Load base model (HF or local) | Config, device | Model, tokenizer |
| `apply_lora()` | Apply LoRA to model | Model, config | Model with LoRA |
| `create_trainer()` | Create HuggingFace Trainer | Model, data, config | Trainer instance |
| `main()` | Main training orchestration | - | Saved model |

**Key Features**:
- Supports HuggingFace models (`google/gemma-3-1b-it`)
- Supports local fine-tuned models (full weights)
- Supports local LoRA adapters (auto-merges with base model)
- Automatic output directory with timestamp
- Progress bars and detailed logging

**Example**:
```bash
# Basic usage
python train.py --config configs/train_config.yaml

# Override samples
python train.py --config configs/train_config.yaml --samples 5000

# Use local fine-tuned model as base
python train.py --config configs/train_config.yaml \
    --model /path/to/outputs/stage2/merged_model
```

---

### 2. `utils.py` - Data Loading Utilities

**Purpose**: Load and format NLLB-200 dataset for translation training.

**Functions**:

#### `load_config(config_path)`
- **Purpose**: Load YAML configuration file
- **Input**: Path to config file (string)
- **Output**: Dictionary with configuration
- **Example**:
  ```python
  config = load_config('configs/train_config.yaml')
  print(config['model']['base_model'])  # 'google/gemma-3-1b-it'
  ```

#### `format_translation_example(example, config, tokenizer)`
- **Purpose**: Format single NLLB-200 example with translation prompt
- **Input**:
  - `example`: Dict with NLLB-200 translation pair
  - `config`: Configuration dictionary
  - `tokenizer`: HuggingFace tokenizer
- **Output**: Dict with `input_ids`, `attention_mask`, `labels`
- **Process**:
  1. Extract source and target text from NLLB format
  2. Filter by length (min/max)
  3. Create instruction prompt: `"Translate from English to Hebrew: {text}"`
  4. Tokenize input and target separately
  5. Return formatted example

- **Example**:
  ```python
  # Input (NLLB-200 format):
  {
      'translation': {
          'eng_Latn': 'Hello, how are you?',
          'heb_Hebr': 'שלום, מה שלומך?'
      }
  }

  # Output (formatted for training):
  {
      'input_ids': [1, 2345, 234, ...],      # Tokenized prompt
      'attention_mask': [1, 1, 1, ...],
      'labels': [5678, 890, ...]              # Tokenized target
  }
  ```

#### `load_nllb_data(tokenizer, config)`
- **Purpose**: Load complete NLLB-200 dataset and prepare for training
- **Input**:
  - `tokenizer`: HuggingFace tokenizer
  - `config`: Configuration dictionary
- **Output**: Tuple of `(train_dataset, eval_dataset)`
- **Process**:
  1. Load NLLB-200 from HuggingFace for specific language pair
  2. Shuffle dataset with seed
  3. Sample train_samples + eval_samples
  4. Split into train and eval sets
  5. Apply formatting to all examples (map operation)
  6. Filter out invalid examples
  7. Return ready-to-train datasets

- **Example**:
  ```python
  train_data, eval_data = load_nllb_data(tokenizer, config)
  print(f"Train: {len(train_data)} samples")  # Train: 1000 samples
  print(f"Eval: {len(eval_data)} samples")    # Eval: 200 samples
  ```

---

### 3. `configs/train_config.yaml` - Configuration File

**Purpose**: Centralized configuration for all training parameters.

**Key Sections**:

| Section | Purpose | Key Parameters |
|---------|---------|----------------|
| `model` | Model selection | `base_model`, `torch_dtype` |
| `translation` | Language pair & data | `source_lang`, `target_lang`, `train_samples` |
| `training` | Training hyperparameters | `num_epochs`, `learning_rate`, `batch_size` |
| `lora` | LoRA configuration | `r`, `alpha`, `target_modules` |
| `logging` | Logging settings | `logging_steps`, `use_wandb` |
| `environment` | Environment setup | `cuda_devices`, `hf_home` |

**Example Modifications**:

```yaml
# Use local fine-tuned model
model:
  base_model: "/home/orrz/gpufs/projects/gemma3/outputs/hebrew/.../merged_model"

# Different language pair
translation:
  source_lang: "French"
  target_lang: "English"
  source_code: "fra_Latn"
  target_code: "eng_Latn"

# More training data
translation:
  train_samples: 10000  # Better quality
  eval_samples: 2000

# Stronger LoRA
lora:
  r: 32  # More capacity
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

---

## Training Flow

### High-Level Overview

```
┌─────────────────┐
│ 1. Load Config  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 2. Setup Env    │ - Set CUDA devices
│                 │ - Set random seed
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 3. Load Model   │ - HuggingFace OR local model
│   & Tokenizer   │ - Detect & merge LoRA adapters
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 4. Apply LoRA   │ - Freeze base model weights
│                 │ - Add trainable LoRA layers
│                 │ - ~1% parameters trainable
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 5. Load Data    │ - Download NLLB-200 (if needed)
│                 │ - Sample train & eval sets
│                 │ - Format with prompts
│                 │ - Tokenize all examples
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 6. Create       │ - TrainingArguments
│    Trainer      │ - DataCollator
│                 │ - Callbacks
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 7. Train        │ - Forward pass
│                 │ - Compute loss
│                 │ - Backward pass (LoRA only)
│                 │ - Update LoRA weights
│                 │ - Evaluate periodically
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 8. Save Model   │ - Save LoRA adapter
│                 │ - Save tokenizer
│                 │ - Save training logs
└─────────────────┘
```

### Detailed Step-by-Step

1. **Configuration Loading**
   - Parse command line arguments
   - Load YAML config file
   - Apply overrides from CLI

2. **Environment Setup**
   - Set `CUDA_VISIBLE_DEVICES`
   - Set HuggingFace cache directory
   - Set random seed for reproducibility
   - Check CUDA availability

3. **Model Loading**
   - **If HuggingFace model**: Download and load from Hub
   - **If local fine-tuned model**: Load from disk
   - **If local LoRA adapter**: Load base model → Load adapter → Merge weights
   - Setup tokenizer (ensure pad_token exists)
   - Print model info (parameters, dtype, vocab size)

4. **LoRA Application**
   - Create LoRA config (rank, alpha, dropout, target modules)
   - Apply LoRA to model using PEFT library
   - Freeze all base model parameters
   - Add trainable LoRA parameters (~1-2% of total)
   - Print trainable vs total parameters

5. **Data Loading & Formatting**
   - Load NLLB-200 from HuggingFace (`allenai/nllb`)
   - Filter by language pair (e.g., `eng_Latn-heb_Hebr`)
   - Shuffle with seed
   - Sample requested number of examples
   - Split into train and eval sets
   - Apply formatting:
     - Create prompt: `"Translate from English to Hebrew: {source_text}"`
     - Tokenize input (prompt + source)
     - Tokenize target (translation)
     - Create attention masks
   - Filter out examples that are too long/short
   - Return PyTorch datasets

6. **Trainer Creation**
   - Create output directory with timestamp
   - Setup TrainingArguments:
     - Batch size, gradient accumulation
     - Learning rate, warmup, scheduler
     - Evaluation and saving strategies
     - Mixed precision (bf16)
     - Gradient checkpointing
   - Create DataCollator (handles padding, batching)
   - Create Trainer instance

7. **Training Loop** (handled by HuggingFace Trainer)
   - For each epoch:
     - For each batch:
       - **Forward pass**: Model generates predictions
       - **Loss computation**: Compare predictions to targets
       - **Backward pass**: Compute gradients (only for LoRA layers!)
       - **Optimizer step**: Update LoRA weights
       - **Logging**: Log loss, learning rate
     - Every N steps:
       - Run evaluation on eval set
       - Save checkpoint if best so far
   - Load best model at end

8. **Model Saving**
   - Save LoRA adapter weights (`adapter_model.bin`)
   - Save adapter config (`adapter_config.json`)
   - Save tokenizer files
   - Print output directory and evaluation instructions

---

## Configuration

### Model Selection

```yaml
# HuggingFace base model
model:
  base_model: "google/gemma-3-1b-it"

# Local fine-tuned model (e.g., from pretraining)
model:
  base_model: "/home/orrz/gpufs/projects/gemma3/outputs/hebrew/.../merged_model"

# Local LoRA adapter (will auto-merge with base)
model:
  base_model: "/home/orrz/gpufs/projects/gemma3/outputs/translation/.../final_model"
```

### Language Pairs

See NLLB-200 language codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md

**Common pairs**:
```yaml
# English → Hebrew
translation:
  source_lang: "English"
  target_lang: "Hebrew"
  source_code: "eng_Latn"
  target_code: "heb_Hebr"

# Hebrew → English (reverse direction)
translation:
  source_lang: "Hebrew"
  target_lang: "English"
  source_code: "heb_Hebr"
  target_code: "eng_Latn"

# English → Arabic
translation:
  source_lang: "English"
  target_lang: "Arabic"
  source_code: "eng_Latn"
  target_code: "ara_Arab"

# French → Spanish
translation:
  source_lang: "French"
  target_lang: "Spanish"
  source_code: "fra_Latn"
  target_code: "spa_Latn"
```

### Sample Size Recommendations

| Use Case | Train Samples | Expected Quality | Training Time |
|----------|---------------|------------------|---------------|
| Quick test | 1,000 | Basic patterns | 5-10 min |
| Proof of concept | 5,000 | Understandable | 20-30 min |
| Research/Demo | 10,000 | Good | 40-60 min |
| Production | 50,000+ | High quality | 3-4 hours |

---

## Usage Examples

### Example 1: Basic Training (English → Hebrew)

```bash
# 1. Edit config
cd /home/orrz/gpufs/projects/gemma3/translation
nano configs/train_config.yaml

# Set: train_samples: 1000

# 2. Train
python train.py --config configs/train_config.yaml

# 3. Output will be in:
# outputs/translation/gemma3-1b_en-he_YYYYMMDD_HHMM/final_model/
```

### Example 2: Larger Training Run

```bash
# Train with 10K samples for better quality
python train.py --config configs/train_config.yaml --samples 10000
```

### Example 3: Continue Training from Pretrained Model

```bash
# Use your Hebrew-pretrained model as base
python train.py --config configs/train_config.yaml \
    --model /home/orrz/gpufs/projects/gemma3/outputs/hebrew/16384_samples/two_stages/stage2/merged_model \
    --samples 5000
```

### Example 4: Multiple Language Pairs

```bash
# Train English → Hebrew
python train.py --config configs/train_config_en_he.yaml

# Train Hebrew → English (reverse)
python train.py --config configs/train_config_he_en.yaml

# Train English → Arabic
python train.py --config configs/train_config_en_ar.yaml
```

### Example 5: Quick Test (Minimal Setup)

```bash
# Train for just 100 samples (very fast, for testing pipeline)
python train.py --config configs/train_config.yaml --samples 100
```

---

## Evaluation

### Using EVALUATION Pipeline

The trained model is compatible with the existing EVALUATION pipeline:

```bash
# 1. Navigate to EVALUATION directory
cd /home/orrz/gpufs/projects/EVALUATION

# 2. Edit config to point to your model
nano config.yaml
```

**Update these lines**:
```yaml
provider:
  huggingface:
    model_name: "/home/orrz/gpufs/projects/gemma3/outputs/translation/gemma3-1b_en-he_20241029_1234/final_model"
    torch_dtype: "bfloat16"

translation:
  source_lang: "English"
  target_lang: "Hebrew"

dataset:
  num_samples: 100  # Test on 100 samples
```

**Run evaluation**:
```bash
python -m src.main --config config.yaml
```

**Output**:
- BLEU score
- chrF score
- CSV with translations
- Results JSON in `results/` directory

### Metrics Explained

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **BLEU** | 0-100 | Word overlap with reference (higher = better) |
| | 10-20 | Basic understanding |
| | 20-30 | Understandable |
| | 30-40 | Good quality |
| | 40+ | High quality |
| **chrF** | 0-100 | Character-level overlap (more fine-grained) |

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation: 8  # Increase to keep effective batch = 16
```

**Solution 2**: Enable gradient checkpointing
```yaml
training:
  gradient_checkpointing: true
```

**Solution 3**: Use smaller model
```yaml
model:
  base_model: "google/gemma-3-1b-it"  # Instead of 4b
```

### Issue: Training Too Slow

**Solution 1**: Increase batch size (if memory allows)
```yaml
training:
  batch_size: 8
  gradient_accumulation: 2
```

**Solution 2**: Reduce evaluation frequency
```yaml
training:
  eval_steps: 500  # Evaluate less often
  save_steps: 500
```

**Solution 3**: Use fewer samples for quick test
```bash
python train.py --config configs/train_config.yaml --samples 1000
```

### Issue: Low Translation Quality

**Solution 1**: Increase training samples
```yaml
translation:
  train_samples: 10000  # Or more
```

**Solution 2**: Increase LoRA rank
```yaml
lora:
  r: 32  # More capacity
```

**Solution 3**: Add more target modules
```yaml
lora:
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

**Solution 4**: Train for more epochs
```yaml
training:
  num_epochs: 2  # Or 3
```

### Issue: Model Not Loading

**Check 1**: Verify model path exists
```bash
ls -la /path/to/model
# Should see: pytorch_model.bin or model.safetensors
```

**Check 2**: For LoRA adapters, verify files
```bash
ls -la /path/to/adapter
# Should see: adapter_config.json, adapter_model.bin
```

**Check 3**: Verify HuggingFace token
```bash
cat /home/orrz/gpufs/hf/.cache/huggingface/token
# Should contain valid token
```

### Issue: Data Loading Fails

**Check 1**: Verify internet connection (for downloading NLLB-200)

**Check 2**: Verify language codes are correct
- See: https://github.com/facebookresearch/flores/blob/main/flores200/README.md

**Check 3**: Clear HuggingFace cache if corrupted
```bash
rm -rf /home/orrz/gpufs/hf/.cache/huggingface/datasets/allenai___nllb
```

---

## Support & Resources

- **NLLB-200 Dataset**: https://huggingface.co/datasets/allenai/nllb
- **Language Codes**: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Gemma Documentation**: https://huggingface.co/google/gemma-3-1b-it

---

## Summary

**This translation fine-tuning pipeline**:
- ✅ Simple: 2 scripts, 1 config, ~500 lines total
- ✅ Fast: 5-10 min for 1K samples
- ✅ Flexible: Supports base models, fine-tuned models, LoRA adapters
- ✅ Efficient: LoRA trains only ~1% of parameters
- ✅ Compatible: Works with existing EVALUATION pipeline
- ✅ Configurable: Easy YAML configuration for all parameters

**Typical workflow**:
1. Edit `configs/train_config.yaml` (2 minutes)
2. Run `python train.py` (5-60 minutes depending on samples)
3. Evaluate with EVALUATION pipeline (10-15 minutes)
4. Iterate: adjust samples, language pair, or LoRA config as needed
---

## Interactive Testing with Jupyter Notebook

A Jupyter notebook `translation_exploration.ipynb` is provided for interactive testing and visualization.

### Features

1. **Dataset Exploration**
   - Load NLLB-200 dataset
   - Sample 5 random English-Hebrew pairs
   - Analyze text length statistics
   - Visualize length distributions

2. **Model Testing**
   - Load base model or fine-tuned model (including LoRA adapters)
   - Generate translations for sampled examples
   - Compare generated vs reference translations
   - Calculate simple quality metrics

3. **Interactive Translation**
   - Test with custom input sentences
   - Batch translate multiple sentences
   - Immediate visual feedback

4. **Visualization**
   - Text length comparisons
   - Character overlap analysis
   - Training loss curves (if training logs available)
   - Length ratio analysis

### Usage

```bash
# Open notebook
jupyter notebook translation_exploration.ipynb

# Or with JupyterLab
jupyter lab translation_exploration.ipynb
```

### Configuration in Notebook

Edit these cells to test your model:

```python
# Cell: Load Model & Tokenizer
BASE_MODEL = "google/gemma-3-1b-it"  # Or your fine-tuned model path
LORA_ADAPTER = None  # Or path to your LoRA adapter
# LORA_ADAPTER = "/home/orrz/gpufs/projects/gemma3/outputs/translation/.../final_model"

# Cell: Interactive Testing
test_text = "Hello, how are you today?"  # Your custom input
```

### Example Outputs

The notebook will show:
- 5 sampled translation pairs from NLLB-200
- Generated translations for those pairs
- Character overlap percentages
- Length comparison plots
- Distribution histograms

---
