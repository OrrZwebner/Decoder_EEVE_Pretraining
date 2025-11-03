#!/usr/bin/env python3
"""
Gemma-3 continual pretraining script with EEVE multi-stage training support
Optimized for RTX 3090 (24GB VRAM) with compilation fixes
Location: /home/orrz/gpufs/projects/gemma3/src/train.py
"""


import yaml
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path # Make sure Path is imported
# import logging
import shutil
import glob
import argparse




# import deepspeed
# from transformers.integrations import HfDeepSpeedConfig


# Get absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()  # /home/orrz/gpufs/projects/gemma3/src
PROJECT_ROOT = SCRIPT_DIR.parent.absolute()    # /home/orrz/gpufs/projects/gemma3
# CONFIG_PATH = PROJECT_ROOT / 'gemma_config.yaml'


# # Initialize config variable
global config

config = None

# Add argument parser for config file
parser = argparse.ArgumentParser(description='Gemma-3 Sanskrit Training')
parser.add_argument('--config', type=str, default='gemma_config.yaml',
                    help='Path to configuration file (default: gemma_config.yaml)')
args = parser.parse_args()

# Load config from specified file
config_path = Path(args.config)

if not config_path.exists():
    print(f"❌ Config file not found: {config_path}")
    sys.exit(1)

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
print(f"✅ Config loaded from: {config_path}")

# Set environment variables IMMEDIATELY after loading config
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['environment']['cuda_visible_devices'])
print(f"Setting CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")

os.environ["TORCH_COMPILE_DISABLE"] = str(config['environment']['torch_compile_disable']).lower()
os.environ["PYTORCH_DISABLE_DYNAMO"] = str(config['environment']['pytorch_disable_dynamo']).lower()
os.environ["HF_HOME"] = config['environment']['hf_home']
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Memory optimization: Enable expandable segments to reduce fragmentation
# This helps prevent OOM errors caused by memory fragmentation during long training runs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug.log')
    ],
    force=True
)
logging.info(f"Setting CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")


# Add src directory to Python path using absolute path
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Now import torch and other libraries after environment is set
import torch


# Verify CUDA sees only the intended GPU
print(f"✅ PyTorch CUDA initialized")
print(f"   Visible GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"   GPU 0 (physical GPU {os.environ['CUDA_VISIBLE_DEVICES']}): {torch.cuda.get_device_name(0)}")
    # Force CUDA initialization to lock in device visibility
    _ = torch.zeros(1).cuda()
    print(f"   CUDA context locked to visible devices")


from pathlib import Path
import wandb

# Import our modular utilities
from data_utils import load_data_pipeline
from model_utils import load_model_pipeline, freeze_layers, get_stage_checkpoint_path

# Import training components
from transformers import (
    TrainingArguments,
    Trainer
)
from training_callbacks import SampleLoggingCallback, DataInspectionCallback, EpochBasedStoppingCallback, MemoryCleanupCallback
from datasets import IterableDataset

from pathlib import Path
from peft import LoraConfig, get_peft_model
import traceback


def create_directories(config):
    """
    Create necessary directories from configuration
    
    Args:
        config (dict): Configuration dictionary containing directory paths
    """
    import os
    
    # Create project directories
    project_config = config['project']
    dirs_to_create = [
        project_config['output_dir'],
        project_config['logs_dir'],
    ]
    
    # Add EEVE stage directory if enabled
    if config.get('eeve', {}).get('enable', False):
        eeve_config = config['eeve']
        stage_dir = eeve_config.get('stage_output_dir', 'outputs/eeve_stages')
        dirs_to_create.append(stage_dir)
    
    # Create all directories
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"📁 Created directory: {dir_path}")
    
    logging.info("✅ All required directories created")


def get_eeve_stage_config(config, stage: int) -> dict:
    """
    Get EEVE configuration for a specific stage
    
    Args:
        config: Configuration dictionary
        stage: Stage number (1-7)
        
    Returns:
        Dictionary with stage configuration
    """
    if not config.get('eeve', {}).get('enable', False):
        return None
    
    # Get stage-specific configuration
    stages_config = config.get('eeve', {}).get('stages', {})
    stage_config = stages_config.get(stage, {})
    
    if not stage_config:
        logging.warning(f"No configuration found for EEVE stage {stage}")
        return None
    
    # Determine train_layers based on vocabulary expansion
    train_layers = stage_config.get('train_layers', 'all')
    
    # Handle stages 4-5 which have different configs based on vocabulary expansion
    if stage in [4, 5]:
        vocab_enabled = config.get('vocabulary', {}).get('use_custom_vocabulary', False)
        
        if not vocab_enabled and 'train_layers_no_vocab' in stage_config:
            train_layers = stage_config['train_layers_no_vocab']
    
    # Get number of added tokens from vocabulary_generation config
    n_added_tokens = get_n_added_tokens(config)
    
    # Handle 2-stage training
    if stage in [1, 2] and config.get('two_stage_training', {}).get('enable', False):
        stage_key = f'stage{stage}'
        stage_cfg = config['two_stage_training'][stage_key]
        return {
            'train_layers': stage_cfg.get('train_layers', 'added_tokens_embeddings' if stage == 1 else 'all'),
            'epochs': stage_cfg.get('epochs', 1),
            'learning_rate': float(stage_cfg.get('learning_rate')),
            'description': stage_cfg.get('description', f'Stage {stage}'),
            'n_added_tokens': n_added_tokens
        }
    return {
        'train_layers': train_layers,
        'epochs': stage_config.get('epochs', 1),
        'learning_rate': stage_config.get('learning_rate', None),
        'description': stage_config.get('description', f'Stage {stage}'),
        'n_added_tokens': n_added_tokens  # Add this for stage processing
    }


def get_n_added_tokens(config) -> int:
    """
    Get the number of added tokens from vocabulary generation configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Number of added tokens
    """
    # Get from vocabulary_generation configuration
    vocab_gen_config = config.get('vocabulary_generation', {})
    n_tokens = vocab_gen_config.get('num_tokens', 0)
    
    if n_tokens > 0:
        logging.info(f"Using {n_tokens} added tokens from vocabulary_generation.num_tokens")
        return n_tokens
    
    # Fallback: check if custom vocabulary is enabled but no generation config
    vocab_config = config.get('vocabulary', {})
    if vocab_config.get('use_custom_vocabulary', False):
        logging.warning("Custom vocabulary enabled but num_tokens not specified in vocabulary_generation config")
        logging.warning("Falling back to analyzing actual vocabulary file...")
        
        # Try to load and count tokens from the actual vocabulary file
        vocab_path = vocab_config.get('vocabulary_full_path')
        if vocab_path and os.path.exists(vocab_path):
            try:
                import pickle
                with open(vocab_path, 'rb') as f:
                    custom_tokens = pickle.load(f)
                    if isinstance(custom_tokens, list):
                        actual_count = len(custom_tokens)
                        logging.info(f"Counted {actual_count} tokens from vocabulary file: {vocab_path}")
                        return actual_count
            except Exception as e:
                logging.error(f"Failed to load vocabulary file {vocab_path}: {e}")
    
    logging.info("No custom vocabulary configured, using 0 added tokens")
    return 0


def get_lora_config_from_stage(config, stage_config: dict = None) -> tuple:
    """
    Determine if LoRA should be enabled and get LoRA configuration.
    
    Args:
        config: Main configuration dictionary
        stage_config: Optional stage-specific configuration
        
    Returns:
        Tuple of (lora_enabled: bool, peft_config: LoraConfig or None)
    """
    lora_cfg = config.get('lora', {})
    
    # Determine if this is stage training or single-stage training
    is_stage_training = stage_config is not None
    
    # Initialize lora_enabled (FIXED: explicit initialization)
    lora_enabled = False
    
    if is_stage_training:
        # For stage training, check stage-level lora_enable
        lora_enabled = stage_config.get('lora_enable', False)
    else:
        # For single-stage training, check single_stage_enable
        lora_enabled = lora_cfg.get('single_stage_enable', False)
    
    # FIXED: Check and return immediately if disabled
    if not lora_enabled:
        if is_stage_training:
            logging.info(f"🚫 LoRA disabled for this stage")
        else:
            logging.info(f"🚫 LoRA disabled for single-stage training")
        return False, None
    
    # Only reach here if LoRA is enabled
    if is_stage_training:
        logging.info(f"✅ LoRA enabled for this stage")
    else:
        logging.info(f"✅ LoRA enabled for single-stage training")
    
    # Build LoRA config from global lora settings
    peft_config = LoraConfig(
        r=lora_cfg.get('r', 64),
        lora_alpha=lora_cfg.get('lora_alpha', 128),
        lora_dropout=lora_cfg.get('lora_dropout', 0.1),
        bias=lora_cfg.get('bias', 'none'),
        target_modules=lora_cfg.get('target_modules', 
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        task_type=lora_cfg.get('task_type', "CAUSAL_LM")
    )
    
    logging.info(f"📊 LoRA configuration:")
    logging.info(f"   r={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")
    logging.info(f"   target_modules={peft_config.target_modules}")
    
    return True, peft_config

def create_training_arguments(config, stage_config: dict = None) -> TrainingArguments:
    """
    Create TrainingArguments from configuration with optional stage overrides
    
    Args:
        config: Configuration dictionary
        stage_config: Optional stage-specific configuration
        
    Returns:
        TrainingArguments instance
    """
    training_config = config['training']
    checkpointing_config = config['checkpointing']
    logging_config = config['logging']
    project_config = config['project']
    
    # Get base parameters
    num_epochs = training_config.get('num_train_epochs', 1)
    learning_rate = training_config.get('learning_rate', 5e-5)
    
    #log learning_rate
    logging.info(f"Base learning rate from config: {learning_rate}")

    # Apply stage overrides if provided
    if stage_config:
        num_epochs = stage_config.get('epochs', num_epochs)
        if stage_config.get('learning_rate'):
            learning_rate = float(stage_config['learning_rate'])
            logging.info(f"Overriding learning rate for stage: {learning_rate}")
    # Ensure we have valid epochs value
    if num_epochs is None or num_epochs <= 0:
        logging.warning(f"Invalid epochs value: {num_epochs}, defaulting to 1")
        num_epochs = 1

        # Handle max_steps for streaming datasets
    max_steps = training_config.get('max_steps', -1)
    data_config = config.get('data', {})
    is_streaming = (
        data_config.get('source_type') == 'huggingface' and 
        data_config.get('hf_streaming', False)
    )

    if is_streaming and max_steps == -1:
        max_steps = 1000000  # Default safety boundary for streaming
        logging.info(f"🛡️ Streaming mode: set max_steps safety boundary to {max_steps:,}")


    # Prepare gradient checkpointing kwargs
    gradient_checkpointing_kwargs = None
    if training_config.get('gradient_checkpointing_kwargs'):
        gradient_checkpointing_kwargs = {
            'use_reentrant': training_config['gradient_checkpointing_kwargs'].get('use_reentrant', False)
        }
    
    # Determine output directory based on stage
    if stage_config and 'stage' in stage_config:
        # Use stage-specific directory (for EEVE or standard)
        output_dir = get_stage_checkpoint_path(config, stage_config.get('stage', 0))
    else:
        # Fallback to default (should not happen with proper setup)
        output_dir = get_stage_checkpoint_path(config, 0)

    

    # Get random seed from config for reproducibility
    random_seed = config.get('data', {}).get('random_seed', 42)

    # Create training arguments
    training_args = TrainingArguments(
        # Output and logging
        output_dir=project_config['output_dir'],
        logging_dir=logging_config.get('logging_dir', project_config['logs_dir']),
        run_name=logging_config.get('wandb_run_name', 'gemma3-sanskrit-training'),

        # Reproducibility: Set seeds for all random operations
        seed=random_seed,           # Global seed for model initialization, dropout, etc.
        data_seed=random_seed,      # Seed for data sampling/shuffling operations

        # Training parameters
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 16),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),

        # Optimization
        learning_rate=float(learning_rate),
        warmup_steps=training_config.get('warmup_steps', 50),
        optim=training_config.get('optimizer', 'adafactor'),
        
        # Memory optimization
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        dataloader_pin_memory=training_config.get('dataloader_pin_memory', False),
        remove_unused_columns=training_config.get('remove_unused_columns', True),
        
        # Distributed training - DeepSpeed or standard
        # deepspeed=deepspeed_config_file if use_deepspeed else None,

        # Precision settings
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', False),
        fp16_full_eval=training_config.get('fp16_full_eval', False),
        # Precision settings - let DeepSpeed handle when enabled
        # fp16=training_config.get('fp16', False) if not use_deepspeed else False,
        # bf16=training_config.get('bf16', False) if not use_deepspeed else False,
        # fp16_full_eval=training_config.get('fp16_full_eval', True) if not use_deepspeed else False,
        tf32=training_config.get('tf32', True),
        
        # Checkpointing and evaluation
        save_strategy=checkpointing_config.get('save_strategy', 'epoch'),
        save_steps=checkpointing_config.get('save_steps', None),
        save_total_limit=checkpointing_config.get('save_total_limit', 0),
        eval_strategy=checkpointing_config.get('eval_strategy', 'epoch'),
        eval_steps=checkpointing_config.get('eval_steps', None),
        load_best_model_at_end=checkpointing_config.get('load_best_model_at_end', False),
        metric_for_best_model=checkpointing_config.get('metric_for_best_model', 'loss'),
        greater_is_better=checkpointing_config.get('greater_is_better', False),
        
        # Logging
        logging_steps=logging_config.get('logging_steps', 10),
        logging_strategy=logging_config.get('logging_strategy', 'steps'),
        logging_first_step=logging_config.get('logging_first_step', True),
        logging_nan_inf_filter=logging_config.get('logging_nan_inf_filter', True),
        report_to=logging_config.get('report_to', 'wandb') if logging_config.get('use_wandb', True) else None,
    )
    
    return training_args


def initialize_wandb(config, stage: int = 0, stage_config: dict = None) -> None:
    """
    Initialize Weights & Biases logging with stage information
    
    Args:
        config: Configuration dictionary
        stage: Current training stage
        stage_config: Stage-specific configuration
    """
    logging_config = config['logging']
    
    if not logging_config.get('use_wandb', True):
        logging.info("Wandb disabled in configuration")
        return
    
    try:
        # Create run name with stage information
        base_name = logging_config.get('wandb_run_name', 'gemma3-sanskrit')
        
        if stage > 0:
            run_name = f"{base_name}_stage{stage}"
            eeve_config = config.get('eeve', {})
            if eeve_config.get('run_name'):
                run_name = f"{eeve_config['run_name']}_stage{stage}"
        else:
            run_name = base_name
        
        # Prepare config for wandb
        wandb_config = config.copy()
        if stage_config:
            wandb_config['current_stage'] = stage
            wandb_config['stage_config'] = stage_config
        
        wandb.init(
            project=logging_config.get('wandb_project', 'gemma3-sanskrit-pretraining'),
            name=run_name,
            config=wandb_config,
            group=config.get('eeve', {}).get('run_name', 'gemma3-training'),
            settings=wandb.Settings(console='off')
        )
        
        logging.info(f"✅ Wandb initialized: {run_name}")
        
    except Exception as e:
        logging.error(f"Failed to initialize wandb: {e}")
        logging.info("Continuing without wandb logging")


def print_training_summary(config, model, train_dataset, eval_dataset, stage: int = 0, stage_config: dict = None) -> None:
    """
    Print training configuration summary with stage information
    
    Args:
        config: Configuration dictionary
        model: Loaded model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        stage: Current training stage
        stage_config: Stage-specific configuration
    """
    training_config = config['training']
    
    print("\n" + "="*60)
    if stage > 0:
        print(f"TRAINING SUMMARY - STAGE {stage}")
        if stage_config:
            print(f"Stage Description: {stage_config.get('description', 'N/A')}")
    else:
        print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"Model: {config['model']['name']}")
    # Handle streaming datasets (no len())
    if isinstance(train_dataset, IterableDataset):
        print(f"Training samples: Unknown (streaming mode)")
        if config.get('debug', {}).get('test_mode', False):
            limit = config.get('debug', {}).get('limit_train_samples', 0)
            if limit > 0:
                print(f"  Debug limit: {limit} samples")
    else:
        print(f"Training samples: {len(train_dataset):,}")
    
    if isinstance(eval_dataset, IterableDataset):
        print(f"Evaluation samples: Unknown (streaming mode)")
        if config.get('debug', {}).get('test_mode', False):
            limit = config.get('debug', {}).get('limit_eval_samples', 0)
            if limit > 0:
                print(f"  Debug limit: {limit} samples")
    else:
        print(f"Evaluation samples: {len(eval_dataset):,}")
    
    
    # Show training parameters
    epochs = stage_config.get('epochs', training_config['num_train_epochs']) if stage_config else training_config['num_train_epochs']
    batch_size = training_config['per_device_train_batch_size']
    effective_batch_size = batch_size * training_config['gradient_accumulation_steps']
    
    print(f"Epochs: {epochs}")
    print(f"Batch size per device: {batch_size}")
    print(f"Gradient accumulation steps: {training_config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")
    
    # Stage-specific information
    if stage > 0 and stage_config:
        print(f"Training layers: {stage_config.get('train_layers', 'N/A')}")
    
    # Memory info
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        print(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    # Debug info
    debug_config = config.get('debug', {})
    if debug_config.get('test_mode', False):
        print("🐛 DEBUG: Test mode enabled")
    if debug_config.get('limit_train_samples', 0) > 0:
        print(f"🐛 DEBUG: Training samples limited to {debug_config['limit_train_samples']}")
    if debug_config.get('limit_eval_samples', 0) > 0:
        print(f"🐛 DEBUG: Evaluation samples limited to {debug_config['limit_eval_samples']}")
    
    print("="*60 + "\n")


def train_single_stage(config, model, tokenizer, train_dataset, eval_dataset, data_collator, 
                      stage: int = 0, stage_config: dict = None) -> str:
    """
    Train a single stage (standard training or EEVE stage)
    
    Args:
        config: Configuration dictionary
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        stage: Stage number (0 for standard, 1-7 for EEVE)
        stage_config: Stage-specific configuration
        
    Returns:
        Path to saved model checkpoint
    """
    
    # Initialize wandb for this stage
    initialize_wandb(config, stage, stage_config)
    
    if stage_config is None:
        stage_config = {}
    stage_config['stage'] = stage

    # Check if LoRA should be applied for this stage
    lora_enabled, peft_config = get_lora_config_from_stage(config, stage_config)
    
    if lora_enabled:
        logging.info(f"🔧 Applying LoRA to model...")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    # Create training arguments
    training_args = create_training_arguments(config, stage_config)
    
    # Print training summary
    print_training_summary(config, model, train_dataset, eval_dataset, stage, stage_config)
    
    # Create callbacks for detailed logging
    callbacks = []

    # Check if using streaming datasets
    is_streaming = isinstance(train_dataset, IterableDataset)

    # For streaming datasets, add epoch-based stopping callback
    if is_streaming:
        epochs = config['training'].get('num_train_epochs', 1)
        if epochs is None or epochs <= 0:
            logging.warning(f"Invalid epochs value: {epochs}, using default: 1")
            epochs = 1
        callbacks.append(EpochBasedStoppingCallback(num_epochs=epochs))
        logging.info(f"✅ Added EpochBasedStoppingCallback (will stop after {epochs} epochs)")

    if config.get('debug', {}).get('verbose_logging', False):
        
        if not is_streaming:
            # Only add data inspection callback for non-streaming datasets
            # (it consumes samples from the stream)
            callbacks.append(DataInspectionCallback(tokenizer, num_samples=3))
            logging.info("Added DataInspectionCallback (non-streaming mode)")
        else:
            logging.info("⚠️ Skipping DataInspectionCallback (streaming mode - would consume data)")
        
        # Add sample logging callback (works with streaming)
        log_interval = config.get('logging', {}).get('sample_log_steps', 50)
        #log how many interval steps
        callbacks.append(SampleLoggingCallback(tokenizer, log_every_n_steps=log_interval))
        logging.info(f"Added SampleLoggingCallback (every {log_interval} steps)")

    # Add memory cleanup callback (ALWAYS enabled for OOM prevention)
    # This runs every 100 steps by default to prevent memory fragmentation
    cleanup_interval = config.get('hardware', {}).get('memory_cleanup_steps', 100)
    callbacks.append(MemoryCleanupCallback(cleanup_every_n_steps=cleanup_interval, log_memory=True))
    logging.info(f"✅ Added MemoryCleanupCallback (every {cleanup_interval} steps)")

    # Initialize trainer
    logging.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Log initial memory usage
    logging.info(f"Model dtype: {next(model.parameters()).dtype}")
    logging.info(f"Embedding dtype: {model.get_input_embeddings().weight.dtype}")

    # # Check for dtype mismatches
    for name, param in model.named_parameters():
        # logging.info(f"Parameter {name} dtype: {param.dtype}")
        if param.dtype != next(model.parameters()).dtype:
            logging.warning(f"Parameter {name} has dtype {param.dtype}, expected {next(model.parameters()).dtype}")
            # logging.warning(f"Parameter {name} has dtype {param.dtype}, not bfloat16!")

    # Start training
    logging.info(f"🚀 Starting training for stage {stage}...")
    print("="*60)
    
    try:
        trainer.train()
        
        logging.info(f"✅ Stage {stage} training completed successfully!")
        
        # Get base directory for this stage/training
        stage_base_dir = get_stage_checkpoint_path(config, stage)
        
        # Save to final_model subdirectory
        save_path = os.path.join(stage_base_dir, "final_model")
        
        # Create directory and save
        os.makedirs(save_path, exist_ok=True)
        
        logging.info(f"💾 Saving final model to {save_path}")
        
        # Check if model is a PEFT model
        # first check if it stage training or single stage with LoRA
        # is_stage_training = config.get('training', {}).get('stage', 0) > 0
        # lora_enabled = config.get('lora', {}).get('enable', False)
        if lora_enabled:
            # For LoRA: save adapters (and embedding layers if resized)
            save_embedding_layers = config.get('lora', {}).get('save_embedding_layers', True)
            model.save_pretrained(
                save_path,
                save_embedding_layers=save_embedding_layers
            )
            logging.info(f"   Saved LoRA adapters")
            if save_embedding_layers:
                logging.info(f"   Saved embedding layers (vocabulary was expanded)")
            
            # Save tokenizer for adapter model
            try:
                tokenizer.save_pretrained(save_path)
                logging.info(f"   Saved tokenizer to adapter directory")
            except Exception as e:
                logging.warning(f"   Failed to save tokenizer: {e}")
                logging.warning(f"   Tokenizer can be loaded from base model during inference")
            
            # Optionally merge and save full model
            if config.get('lora', {}).get('merge_on_save', False):
                logging.info(f"   Merging LoRA adapters into base model...")
                merged_model = model.merge_and_unload()
                merged_save_path = os.path.join(stage_base_dir, "merged_model")
                os.makedirs(merged_save_path, exist_ok=True)
                
                # Save merged model
                merged_model.save_pretrained(merged_save_path)
                logging.info(f"   Saved merged model to {merged_save_path}")
                
                # Save tokenizer for merged model too
                try:
                    tokenizer.save_pretrained(merged_save_path)
                    logging.info(f"   Saved tokenizer to merged model directory")
                except Exception as e:
                    logging.warning(f"   Failed to save tokenizer to merged dir: {e}")
        else:
            # For full model training
            model.save_pretrained(save_path)
            logging.info(f"   Saved full model")
            
            # Save tokenizer
            tokenizer.save_pretrained(save_path)
            logging.info(f"   Saved tokenizer")

        # Clean up checkpoints after saving final model
        checkpoint_pattern = os.path.join(stage_base_dir, "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            logging.info(f"🧹 Cleaning up {len(checkpoints)} checkpoint(s)")
            for checkpoint_dir in checkpoints:
                try:
                    shutil.rmtree(checkpoint_dir)
                    logging.info(f"   Deleted: {checkpoint_dir}")
                except Exception as e:
                    logging.warning(f"   Failed to delete {checkpoint_dir}: {e}")
        
        # Log final memory usage
        if torch.cuda.is_available():
            device = next(model.parameters()).device
            final_memory = torch.cuda.memory_allocated(device) / 1024**3
            logging.info(f"Final GPU memory usage: {final_memory:.2f} GB")
        
        return save_path
        
    except Exception as e:
        logging.error(f"❌ Stage {stage} training failed: {e}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        raise
    
    finally:
        # Finish wandb run for this stage
        if config['logging'].get('use_wandb', True):
            try:
                wandb.finish()
            except:
                pass


def run_eeve_training(config) -> None:
    """
    Run complete EEVE multi-stage training
    
    Args:
        config: Configuration dictionary
    """
    eeve_config = config['eeve']
    start_stage = eeve_config.get('start_stage', 1)
    end_stage = eeve_config.get('end_stage', 7)

    # Set timestamp once for all stages
    config['_training_timestamp'] = datetime.now().strftime("%d-%m-%Y_%H%M")
    
    logging.info(f"🔄 Starting EEVE training: stages {start_stage} to {end_stage}")
    
    # Warn about streaming datasets with EEVE
    if config.get('data', {}).get('source_type') == 'huggingface' and \
       config.get('data', {}).get('hf_streaming', False):
        logging.warning("⚠️ Using streaming datasets with EEVE multi-stage training")
        logging.warning("   Data will be reloaded for each stage (streaming iterators are consumed)")

    # Load data once (shared across all stages)
    logging.info("Loading data pipeline...")
    tokenizer = None  # Will be loaded with model in first stage
    
    for stage in range(start_stage, end_stage + 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"🎯 STARTING EEVE STAGE {stage}")
        logging.info(f"{'='*60}")
        
        # Get stage configuration
        stage_config = get_eeve_stage_config(config, stage)
        if not stage_config:
            logging.error(f"No configuration for stage {stage} - skipping")
            continue
        
        # Load model for this stage
        load_from_previous = stage > start_stage
        model, tokenizer = load_model_pipeline(config, stage, load_from_previous)
        
        # Load data
        logging.info(f"Loading data pipeline for stage {stage}...")
        train_dataset, eval_dataset, data_collator = load_data_pipeline(tokenizer, config)
        
        
        # Train this stage
        save_path = train_single_stage(
            config, model, tokenizer, train_dataset, eval_dataset, data_collator,
            stage, stage_config
        )
        
        logging.info(f"✅ Stage {stage} completed. Model saved to: {save_path}")
        
        # Clear references to datasets (especially important for streaming)
        if isinstance(train_dataset, IterableDataset):
            # For streaming datasets, explicitly delete references
            del train_dataset
            del eval_dataset
            logging.info("Cleared streaming dataset references")

        # Clear GPU memory between stages
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logging.info(f"🎉 EEVE training completed! All stages {start_stage}-{end_stage} finished.")


def run_standard_training(config) -> None:
    """
    Run standard single-stage training
    
    Args:
        config: Configuration dictionary
    """
    logging.info("🔄 Starting standard training...")
    
    # Set timestamp for standard training
    config['_training_timestamp'] = datetime.now().strftime("%d-%m-%Y_%H%M")

    # Load model and data
    model, tokenizer = load_model_pipeline(config)
    train_dataset, eval_dataset, data_collator = load_data_pipeline(tokenizer, config)
    
    # Train with stage_config=None to trigger single_stage_enable check
    save_path = train_single_stage(
        config, model, tokenizer, train_dataset, eval_dataset, data_collator,
        stage=0,           # Stage 0 indicates single-stage training
        stage_config=None  # None triggers single_stage_enable check
    )
    
    logging.info(f"✅ Standard training completed. Model saved to: {save_path}")



def run_2_stage_training(config) -> None:
    """
    2-stage training: embeddings → full model fine-tuning

    Stage 1 is SKIPPED if no custom vocabulary expansion is enabled,
    since there are no new token embeddings to train.
    """
    logging.info("="*80)
    logging.info("🚀 STARTING 2-STAGE TRAINING")
    logging.info("="*80)

    config['_training_timestamp'] = datetime.now().strftime("%d-%m-%Y_%H%M")

    two_stage_config = config.get('two_stage_training', {})
    stage1_config = two_stage_config.get('stage1', {})
    stage2_config = two_stage_config.get('stage2', {})

    # Check if vocabulary expansion is enabled
    vocab_config = config.get('vocabulary', {})
    use_custom_vocab = vocab_config.get('use_custom_vocabulary', False)

    save_path_stage1 = None  # Will be set if Stage 1 runs

    # ===== STAGE 1: Train New Token Embeddings (CONDITIONAL) =====
    if use_custom_vocab:
        logging.info("\n" + "="*80)
        logging.info("📍 STAGE 1: Training New Token Embeddings")
        logging.info("="*80)
        logging.info("Custom vocabulary expansion is enabled")
        logging.info("Stage 1 will train only the new token embeddings")

        model, tokenizer = load_model_pipeline(config, stage=1)

        # Verify that tokens were actually added
        from model_utils import get_n_added_tokens
        n_added_tokens = get_n_added_tokens(config, model, tokenizer)

        if n_added_tokens > 0:
            logging.info(f"✅ Confirmed: {n_added_tokens} tokens added - Stage 1 is necessary")

            train_dataset, eval_dataset, data_collator = load_data_pipeline(tokenizer, config)

            save_path_stage1 = train_single_stage(
                config, model, tokenizer, train_dataset, eval_dataset, data_collator,
                stage=1, stage_config=stage1_config
            )

            logging.info(f"✅ Stage 1 completed: {save_path_stage1}")

            # Clean up before Stage 2
            del model, train_dataset, eval_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logging.warning("⚠️ Custom vocabulary enabled but no tokens were added")
            logging.warning("⚠️ Skipping Stage 1 (no new embeddings to train)")
            logging.info("Proceeding directly to Stage 2...")
            # Don't load model yet, will be loaded in Stage 2
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        logging.info("\n" + "="*80)
        logging.info("⏭️  SKIPPING STAGE 1: No Vocabulary Expansion")
        logging.info("="*80)
        logging.info("Custom vocabulary is disabled (use_custom_vocabulary: false)")
        logging.info("Stage 1 trains new token embeddings, but no new tokens exist")
        logging.info("Proceeding directly to Stage 2 (full model training)...")
        logging.info("="*80)

    # ===== STAGE 2: Full Model Fine-tuning =====
    logging.info("\n" + "="*80)
    logging.info("📍 STAGE 2: Full Model Fine-tuning")
    logging.info("="*80)

    # Load model - either from Stage 1 checkpoint or fresh
    if save_path_stage1:
        logging.info(f"Loading model from Stage 1 checkpoint: {save_path_stage1}")
        model, tokenizer = load_model_pipeline(config, stage=2, load_from_dir=save_path_stage1)
    else:
        logging.info("Loading fresh model (Stage 1 was skipped)")
        model, tokenizer = load_model_pipeline(config, stage=2)

    # Load data and train
    train_dataset, eval_dataset, data_collator = load_data_pipeline(tokenizer, config)

    save_path_stage2 = train_single_stage(
        config, model, tokenizer, train_dataset, eval_dataset, data_collator,
        stage=2, stage_config=stage2_config
    )

    logging.info(f"✅ Stage 2 completed: {save_path_stage2}")
    logging.info("="*80)
    logging.info("🎉 2-STAGE TRAINING COMPLETE!")
    logging.info("="*80)



def create_temporary_sanskrit_config(config) -> str:
    """
    Create a temporary sanskrit_config.yaml file for the Tokenizers project.
    Supports both local files and HuggingFace datasets.
    
    Args:
        config (dict): The main Gemma config dictionary
        
    Returns:
        str: Path to the temporary config file
    """
    
    # Create temporary config content based on Gemma config
    gen_config = config.get('vocabulary_generation', {})
    
    # ===== DATA CONFIGURATION =====
    # Check for new nested structure first, then fall back to old structure
    if 'data' in gen_config:
        # New nested structure: vocabulary_generation.data
        vocab_data_config = gen_config['data']
    else:
        # Old flat structure: backwards compatibility
        vocab_data_config = {
            'source_type': 'local_files',
            'path': gen_config.get('data_path', 
                   config.get('data', {}).get('file_paths', [None])[0] 
                   if config.get('data', {}).get('file_paths') 
                   else '/home/orrz/gpufs/projects/gemma3/sanskrit_data'),
            'debug': 0
        }
    
    # Build the data config for Tokenizers project
    tokenizer_data_config = {}
    
    source_type = vocab_data_config.get('source_type', 'local_files')
    tokenizer_data_config['source_type'] = source_type
    
    if source_type == 'huggingface':
        # HuggingFace dataset configuration
        tokenizer_data_config.update({
            'hf_dataset_name': vocab_data_config.get('hf_dataset_name'),
            'hf_dataset_config': vocab_data_config.get('hf_dataset_config'),
            'hf_text_column': vocab_data_config.get('hf_text_column', 'text'),
            'hf_split': vocab_data_config.get('hf_split', 'train'),
            'hf_streaming': vocab_data_config.get('hf_streaming', False),
            'hf_trust_remote_code': vocab_data_config.get('hf_trust_remote_code', False),
            'debug': vocab_data_config.get('debug', 0)
        })
    else:
        # Local files configuration
        tokenizer_data_config.update({
            'path': vocab_data_config.get('path', '/home/orrz/gpufs/projects/gemma3/sanskrit_data'),
            'debug': vocab_data_config.get('debug', 0)
        })
    
    temp_config = {
        'data': tokenizer_data_config,
        'training': {
            'num_samples': -1,
            'num_new_tokens': gen_config.get('num_tokens', 2048),
            'unigram_max_iterations': 10
        },
        'models': {
            gen_config.get('model_target', 'gemma'): {
                'model_name': config.get('model', {}).get('name', 'google/gemma-3-1b-it'),
                'algorithms': [gen_config.get('algorithm_target', 'sentencepiece_bpe')]
            }
        },
        'output': {
            # 'create_plots': False,  # Disable plots for vocabulary generation
            'create_plots': True,  # Enable plots for vocabulary generation
            'plot_filename': 'sanskrit_tokenizer_comparison.png',
            'save_individual_plots': False,
            'compression_test_samples': 1000
        },
        'logging': {
            'log_dir': '/home/orrz/gpufs/projects/Tokenizers/logs'
        },
        # 'random_seed': 42
        'random_seed': config.get('data', {}).get('random_seed', config.get('random_seed', 42))

    }

    # Create configs directory in Tokenizers project
    tokenizers_configs_dir = "/home/orrz/gpufs/projects/Tokenizers/configs"
    os.makedirs(tokenizers_configs_dir, exist_ok=True)

    # Create temporary file with timestamp
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    temp_config_path = os.path.join(tokenizers_configs_dir, f"sanskrit_config_{timestamp}.yaml")
    
    try:
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f, default_flow_style=False, indent=2)
        
        logging.info(f"📄 Created config file: {temp_config_path}")
        return temp_config_path
        
    except Exception as e:
        logging.error(f"❌ Failed to create temporary config: {e}")
        raise


def run_vocabulary_generation(config):
    """
    Executes the external vocabulary generation script if enabled.
    
    Args:
        config: Configuration dictionary
    """
    gen_config = config.get('vocabulary_generation')
    vocab_config = config['vocabulary']
    str_num_tokens = str(config.get('vocabulary_generation', {}).get('num_tokens', ""))
    vocab_path = vocab_config['vocabulary_full_path'] + str_num_tokens +'.pkl'
    # Check if the feature is enabled
    if not gen_config or not gen_config.get('enable'):
        logging.info("➡️ Vocabulary generation script execution is disabled in config.")
        return
    
    # Check if vocabulary file already exists (optional skip)

    if os.path.exists(vocab_path):
        logging.info(f"📄 Vocabulary file already exists: {vocab_path}")
        logging.info("➡️ Skipping vocabulary generation.")
        return
    
    # Check if we should use existing vocabulary file
    if gen_config.get('use_existing_vocabulary', False):
        # vocab_path = config['vocabulary']['vocabulary_full_path']
        if os.path.exists(vocab_path):
            logging.info(f"📄 Using existing vocabulary file: {vocab_path}")
            return
        else:
            logging.warning(f"⚠️ Existing vocabulary file not found: {vocab_path}")
            logging.info("🔄 Falling back to vocabulary generation...")
    
    # Generate new vocabulary (either because use_existing_vocabulary=false or file not found)
    logging.info("🚀 Executing external vocabulary generation script...")
    
    # Get parameters from the config
    script_path = gen_config.get('script_path')
    model = gen_config.get('model_target')
    algo = gen_config.get('algorithm_target')
    
    # Convert output file path to absolute if needed
    # vocab_path = config['vocabulary']['vocabulary_full_path']
    if not os.path.isabs(vocab_path):
        output_file = str(PROJECT_ROOT / vocab_path)
    else:
        output_file = vocab_path

    if not all([script_path, model, algo, output_file]):
        logging.error("❌ Missing configuration under 'vocabulary_generation' or 'vocabulary' in YAML.")
        raise ValueError("Incomplete configuration for vocabulary generation.")

    # Create temporary config file
    temp_config_path = create_temporary_sanskrit_config(config)
    
    try:
        # Construct the command to run the Tokenizers script
        command = [
            "python", script_path,
            "--config", temp_config_path,  # Use temporary config
            "--generate-for-model", model,
            "--with-algorithm", algo,
            "--output-file", output_file
        ]
        
        logging.info(f"Running command: {' '.join(command)}")
        # Execute the command and wait for it to complete
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info("✅ Vocabulary generation script finished successfully.")
        logging.info(f"--- Script Output ---\n{result.stdout.strip()}\n---------------------")

    except FileNotFoundError:
        logging.error(f"❌ Script not found at '{script_path}'. Please check 'vocabulary_generation.script_path' in your config.")
        raise
    except subprocess.CalledProcessError as e:
        logging.error("❌ Vocabulary generation script FAILED!")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"--- STDOUT ---\n{e.stdout.strip()}")
        logging.error(f"--- STDERR ---\n{e.stderr.strip()}")
        raise RuntimeError("Vocabulary generation failed, stopping training process.")
    

def main():
    """Main training function with EEVE support"""

    

    print("Starting Gemma-3 Sanskrit Continual Pretraining with EEVE Support...")
    print("="*60)

    try:
        # Create necessary directories
        logging.info("Creating required directories...")
        create_directories(config)
        logging.info("✅ Directories created successfully")

        # Run vocabulary generation if enabled
        if config.get('vocabulary_generation', {}).get('enable', False):
            logging.info("🔄 Running vocabulary generation...")
            run_vocabulary_generation(config)
            logging.info("✅ Vocabulary generation completed (if enabled)")
        
        # Determine training mode
        if config.get('two_stage_training', {}).get('enable', False):
            run_2_stage_training(config)
        elif config.get('eeve', {}).get('enable', False):
            run_eeve_training(config)
        else:
            run_standard_training(config)
        
        print(f"\n" + "="*60)
        print("🎉 ALL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        # sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        logging.info("Training interrupted by user")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Training failed: {e}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        
        if torch.cuda.is_available():
            print("GPU memory status at error:")
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print("Consider reducing batch size or enabling more aggressive memory optimizations")
        
        raise e


if __name__ == "__main__":
    # Set up basic logging before loading config
    print(f'{"="*60}')
    print(f' Starting Gemma-3 Sanskrit Training Script at {datetime.now().strftime("%d-%m-%Y %H:%M")}')
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s',
    #     handlers=[
    #         logging.StreamHandler(sys.stdout),  # Force stdoutd
    #         logging.FileHandler('debug.log')     # Also save to file
    #     ],
    #     force=True  # Override any existing configuration
    # )
    
    # Force flush after each log
    for handler in logging.root.handlers:
        handler.flush()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check PyTorch and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        # print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Current CUDA device name: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"GPU memory at start: {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    else:
        print("⚠️  CUDA not available - will use CPU")
    
    # Run main training
    try:
        main()
        print(f'{"="*60}')
        print(f'🎉 Training completed  at {datetime.now().strftime("%d-%m-%Y %H:%M")}')
        print("✅ Script reached the end of main function, exiting cleanly.")
        sys.exit(0)
    except Exception as e:
        # This block will run if any error occurs anywhere in main()
        print(f"❌ AN UNHANDLED EXCEPTION OCCURRED: {e}")
        # traceback.print_exc()  # This prints the full error traceback
        sys.exit(1)          # Exit with a failure code