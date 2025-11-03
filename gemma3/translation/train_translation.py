#!/usr/bin/env python3
"""
Translation Fine-tuning Script for Gemma-3

Purpose:
    Fine-tune Gemma-3 models for neural machine translation using LoRA
    (Low-Rank Adaptation) for parameter-efficient training.

Usage:
    python train_translation.py --config configs/train_config.yaml

Input:
    - Config YAML file with model, data, and training settings
    - NLLB-200 dataset (automatically downloaded from HuggingFace)

Output:
    - Fine-tuned LoRA adapter weights in outputs/translation/
    - Training logs and metrics
    - Tokenizer files for inference

Features:
    - Supports both base models (google/gemma-3-1b-it) and local fine-tuned models
    - LoRA for efficient fine-tuning (only trains ~1% of parameters)
    - Configurable language pairs and sample sizes
    - Compatible with EVALUATION pipeline for testing
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, get_peft_model, PeftModel

# Local imports
from utils import load_config, load_nllb_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Fine-tune Gemma-3 for translation using LoRA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_translation.py --config configs/train_config.yaml

  # Override config values
  python train_translation.py --config configs/train_config.yaml --samples 5000

  # Use local fine-tuned model as base
  python train_translation.py --config configs/train_config.yaml --model /path/to/local/model
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Override train_samples from config'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Override base_model from config (can be HF model or local path)'
    )

    return parser.parse_args()


def setup_environment(config):
    """
    Setup environment variables and CUDA devices

    Args:
        config: Configuration dictionary

    Returns:
        Device string (e.g., 'cuda:0')
    """
    env_config = config['environment']

    # Set CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = env_config['cuda_devices']
    print(f"CUDA_VISIBLE_DEVICES: {env_config['cuda_devices']}")

    # Set HuggingFace cache
    os.environ['HF_HOME'] = env_config['hf_home']

    # Set random seeds for reproducibility
    seed = env_config['seed']
    print(f"Setting random seed: {seed}")

    # Set seed for all random operations
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (all devices)
    set_seed(seed)  # HuggingFace transformers (sets all of the above + more)
    print(f"✓ Random seeds set")

    # Check CUDA availability
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("⚠ CUDA not available, using CPU")

    return device


def load_model_and_tokenizer(config, device):
    """
    Load base model and tokenizer

    Supports:
    - HuggingFace model names (e.g., 'google/gemma-3-1b-it')
    - Local fine-tuned models (e.g., '/path/to/outputs/model')
    - Local LoRA adapters (automatically detected and merged)

    Args:
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = config['model']
    base_model_path = model_config['base_model']

    print(f"\n{'='*60}")
    print("Loading Model and Tokenizer")
    print(f"{'='*60}")
    print(f"  Base model: {base_model_path}")

    # Check if this is a local path or HuggingFace model
    is_local = os.path.exists(base_model_path)

    if is_local:
        print(f"  ✓ Detected local model path")

        # Check if it's a LoRA adapter directory
        adapter_config_path = os.path.join(base_model_path, 'adapter_config.json')
        is_lora_adapter = os.path.exists(adapter_config_path)

        if is_lora_adapter:
            print(f"  ✓ Detected LoRA adapter, will load with base model")

            # Load adapter config to find base model
            import json
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
                original_base_model = adapter_config.get('base_model_name_or_path', 'google/gemma-3-1b-it')

            print(f"  ✓ Original base model: {original_base_model}")

            # Load base model first
            print(f"\n  Loading base model: {original_base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                original_base_model,
                torch_dtype=getattr(torch, model_config['torch_dtype']),
                attn_implementation=model_config['attn_implementation'],
                trust_remote_code=model_config['trust_remote_code'],
                low_cpu_mem_usage=model_config['low_cpu_mem_usage']
            )

            # Load and merge LoRA adapter
            print(f"  Loading LoRA adapter from: {base_model_path}")
            model = PeftModel.from_pretrained(model, base_model_path)
            print(f"  ✓ LoRA adapter loaded")

            # Optionally merge adapter weights into base model
            print(f"  Merging LoRA adapter into base model...")
            model = model.merge_and_unload()
            print(f"  ✓ LoRA adapter merged")

            # Load tokenizer from adapter directory or base model
            tokenizer_path = base_model_path if os.path.exists(os.path.join(base_model_path, 'tokenizer_config.json')) else original_base_model
            print(f"  Loading tokenizer from: {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        else:
            # Regular fine-tuned model (full weights)
            print(f"  ✓ Loading full fine-tuned model")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=getattr(torch, model_config['torch_dtype']),
                attn_implementation=model_config['attn_implementation'],
                trust_remote_code=model_config['trust_remote_code'],
                low_cpu_mem_usage=model_config['low_cpu_mem_usage']
            )

            tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    else:
        # HuggingFace model
        print(f"  ✓ Loading from HuggingFace Hub")

        # Check for HF token if needed
        token = None
        token_path = config['environment'].get('hf_token_path')
        if token_path and os.path.exists(token_path):
            with open(token_path) as f:
                token = f.read().strip()
            print(f"  ✓ Using HuggingFace token from: {token_path}")

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=getattr(torch, model_config['torch_dtype']),
            attn_implementation=model_config['attn_implementation'],
            trust_remote_code=model_config['trust_remote_code'],
            low_cpu_mem_usage=model_config['low_cpu_mem_usage'],
            token=token
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=token)

    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  ✓ Set pad_token to eos_token")

    # Move model to device
    # model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model loaded successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model dtype: {model.dtype}")
    print(f"  Tokenizer vocab size: {len(tokenizer):,}")
    print(f"{'='*60}\n")

    return model, tokenizer


def apply_lora(model, config):
    """
    Apply LoRA (Low-Rank Adaptation) to model

    LoRA freezes base model weights and adds trainable low-rank matrices
    to attention layers, reducing trainable parameters by ~99%.

    Args:
        model: Base model
        config: Configuration dictionary

    Returns:
        Model with LoRA applied
    """
    lora_config_dict = config['lora']

    print(f"\n{'='*60}")
    print("Applying LoRA")
    print(f"{'='*60}")

    lora_config = LoraConfig(
        r=lora_config_dict['r'],
        lora_alpha=lora_config_dict['alpha'],
        lora_dropout=lora_config_dict['dropout'],
        target_modules=lora_config_dict['target_modules'],
        bias=lora_config_dict['bias'],
        task_type=lora_config_dict['task_type']
    )

    print(f"  LoRA rank: {lora_config.r}")
    print(f"  LoRA alpha: {lora_config.lora_alpha}")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print(f"  Target modules: {lora_config.target_modules}")

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    print(f"\n  Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_percent:.2f}%)")
    print(f"  ✓ LoRA applied successfully")
    print(f"{'='*60}\n")

    return model


def create_trainer(model, tokenizer, train_dataset, eval_dataset, config):
    """
    Create HuggingFace Trainer for fine-tuning

    Args:
        model: Model with LoRA applied
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Configuration dictionary

    Returns:
        Trainer instance
    """
    train_config = config['training']
    trans_config = config['translation']

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = config['model']['base_model'].split('/')[-1]
    lang_pair = f"{trans_config['source_code'].split('_')[0]}-{trans_config['target_code'].split('_')[0]}"
    output_dir = os.path.join(
        train_config['output_dir'],
        f"{model_name}_{lang_pair}_{timestamp}"
    )

    print(f"\n{'='*60}")
    print("Creating Trainer")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")

    # Create training arguments
    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        overwrite_output_dir=True,

        # Training
        num_train_epochs=train_config['num_epochs'],
        per_device_train_batch_size=train_config['batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation'],
        learning_rate=float(train_config['learning_rate']),
        warmup_steps=train_config['warmup_steps'],
        lr_scheduler_type=train_config['scheduler'],
        max_grad_norm=train_config['max_grad_norm'],

        # Optimizer
        optim=config['advanced']['optimizer'],
        weight_decay=config['advanced']['weight_decay'],

        # Precision
        bf16=train_config['bf16'],
        fp16=train_config['fp16'],

        # Memory
        gradient_checkpointing=train_config['gradient_checkpointing'],

        # Evaluation
        eval_strategy=train_config['eval_strategy'],
        eval_steps=train_config['eval_steps'],
        save_strategy=train_config['save_strategy'],
        save_steps=train_config['save_steps'],
        save_total_limit=train_config['save_total_limit'],
        load_best_model_at_end=train_config['load_best_model_at_end'],
        metric_for_best_model=train_config['metric_for_best_model'],

        # Logging
        logging_steps=config['logging']['logging_steps'],
        report_to='wandb' if config['logging']['use_wandb'] else 'none',

        # Misc
        seed=config['environment']['seed'],
    )

    # Data collator (handles padding and batching)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=config['advanced']['label_pad_token_id'],
        padding=True,
        return_tensors='pt'
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(f"  ✓ Trainer created")
    print(f"  Effective batch size: {train_config['batch_size'] * train_config['gradient_accumulation']}")
    print(f"  Total training steps: ~{len(train_dataset) // (train_config['batch_size'] * train_config['gradient_accumulation']) * train_config['num_epochs']}")
    print(f"{'='*60}\n")

    return trainer, output_dir


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    print(f"\n{'='*80}")
    print("TRANSLATION FINE-TUNING - GEMMA-3")
    print(f"{'='*80}\n")

    # Load config
    config = load_config(args.config)
    print(f"✓ Configuration loaded from: {args.config}\n")

    # Override config with command line args
    if args.samples:
        config['translation']['train_samples'] = args.samples
        print(f"  Overriding train_samples: {args.samples}")

    if args.model:
        config['model']['base_model'] = args.model
        print(f"  Overriding base_model: {args.model}")

    # Setup environment
    device = setup_environment(config)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, device)

    # Apply LoRA
    model = apply_lora(model, config)

    # Load data
    train_dataset, eval_dataset = load_nllb_data(tokenizer, config)

    # Create trainer
    trainer, output_dir = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)

    # Train
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")

    trainer.train()

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")

    # Save LoRA adapter
    lora_output_dir = os.path.join(output_dir, 'lora_adapter')
    print(f"Saving LoRA adapter to: {lora_output_dir}")
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print(f"✓ LoRA adapter saved")

    # Merge LoRA weights with base model and save
    print(f"\nMerging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()

    merged_output_dir = os.path.join(output_dir, 'merged_model')
    print(f"Saving merged model to: {merged_output_dir}")
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    print(f"✓ Merged model saved")

    print(f"\n{'='*60}")
    print("Output Summary")
    print(f"{'='*60}")
    print(f"  LoRA adapter: {lora_output_dir}")
    print(f"  Merged model: {merged_output_dir}")
    print(f"\nTo evaluate, use EVALUATION pipeline:")
    print(f"  cd /home/orrz/gpufs/projects/EVALUATION")
    print(f"  # Edit config.yaml to point to: {merged_output_dir}")
    print(f"  python -m src.main --config config.yaml")

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
