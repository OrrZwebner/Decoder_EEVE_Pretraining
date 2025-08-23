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
import logging
import shutil
import glob
import argparse
from pathlib import Path

# import deepspeed
# from transformers.integrations import HfDeepSpeedConfig


# Get absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()  # /home/orrz/gpufs/projects/gemma3/src
PROJECT_ROOT = SCRIPT_DIR.parent.absolute()    # /home/orrz/gpufs/projects/gemma3
# CONFIG_PATH = PROJECT_ROOT / 'gemma_config.yaml'

# # Load config using absolute path
# try:
#     with open(CONFIG_PATH, 'r') as file:
#         config = yaml.safe_load(file)
#     print(f"✅ Config loaded from: {CONFIG_PATH}")
# except FileNotFoundError:
#     print(f"❌ Config file not found: {CONFIG_PATH}")
#     print(f"Expected location: {CONFIG_PATH}")
#     print(f"Current working directory: {os.getcwd()}")
#     sys.exit(1)

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
logging.info(f"Setting CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
os.environ["TORCH_COMPILE_DISABLE"] = str(config['environment']['torch_compile_disable']).lower()
os.environ["PYTORCH_DISABLE_DYNAMO"] = str(config['environment']['pytorch_disable_dynamo']).lower()
os.environ["HF_HOME"] = config['environment']['hf_home']
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Add src directory to Python path using absolute path
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Now import torch and other libraries after environment is set
import torch

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
    
    # Apply stage overrides if provided
    if stage_config:
        num_epochs = stage_config.get('epochs', num_epochs)
        if stage_config.get('learning_rate'):
            learning_rate = stage_config['learning_rate']

    # # Simple DeepSpeed check - just look for enable flag
    # use_deepspeed = config.get('deepspeed', {}).get('enable', False)
    # deepspeed_config_file = None

    # if use_deepspeed:
    #     # Use the config file path from YAML or default
    #     deepspeed_config_file = config['deepspeed'].get(
    #         'config_file', 
    #         '/home/orrz/gpufs/projects/gemma3/deepspeed_config.json'
    #     )
    #     logging.info(f"🚀 DeepSpeed ENABLED - using config: {deepspeed_config_file}")
    # else:
    #     logging.info("DeepSpeed disabled - using standard training")

    # Ensure we have valid epochs value
    if num_epochs is None or num_epochs <= 0:
        logging.warning(f"Invalid epochs value: {num_epochs}, defaulting to 1")
        num_epochs = 1

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

    

    # Create training arguments
    training_args = TrainingArguments(
        # Output and logging
        output_dir=project_config['output_dir'],
        logging_dir=logging_config.get('logging_dir', project_config['logs_dir']),
        run_name=logging_config.get('wandb_run_name', 'gemma3-sanskrit-training'),
        
        # Training parameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 16),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 16),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
        
        # Optimization
        learning_rate=learning_rate,
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
        print(f"EEVE TRAINING SUMMARY - STAGE {stage}")
        if stage_config:
            print(f"Stage Description: {stage_config.get('description', 'N/A')}")
    else:
        print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"Model: {config['model']['name']}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
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

    # Create training arguments
    training_args = create_training_arguments(config, stage_config)
    
    # Print training summary
    print_training_summary(config, model, train_dataset, eval_dataset, stage, stage_config)
    
    # Initialize trainer
    logging.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
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
        
        # logging.info(f"✅ Stage {stage} training completed successfully!")
        
        # # Save the model for this stage
        # if stage > 0:
        #     # EEVE stage - save to stage-specific directory
        #     save_path = get_stage_checkpoint_path(config, stage)
        # else:
        #     # Standard training - save to output directory
        #     save_path = os.path.join(config['project']['output_dir'], "final_model")
        
        # # Create directory and save
        # os.makedirs(save_path, exist_ok=True)
        
        # logging.info(f"💾 Saving model to {save_path}")
        # model.save_pretrained(save_path)
        # tokenizer.save_pretrained(save_path)
        logging.info(f"✅ Stage {stage} training completed successfully!")
        
        # Get base directory for this stage/training
        stage_base_dir = get_stage_checkpoint_path(config, stage)
        
        # Save to final_model subdirectory
        save_path = os.path.join(stage_base_dir, "final_model")
        
        # Create directory and save
        os.makedirs(save_path, exist_ok=True)
        
        logging.info(f"💾 Saving final model to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
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
        
        # Load data if not already loaded or if tokenizer changed
        if stage == start_stage or tokenizer is None:
            train_dataset, eval_dataset, data_collator = load_data_pipeline(tokenizer, config)
        
        # Train this stage
        save_path = train_single_stage(
            config, model, tokenizer, train_dataset, eval_dataset, data_collator,
            stage, stage_config
        )
        
        logging.info(f"✅ Stage {stage} completed. Model saved to: {save_path}")
        
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
    
    # Train
    save_path = train_single_stage(
        config, model, tokenizer, train_dataset, eval_dataset, data_collator
    )
    
    logging.info(f"✅ Standard training completed. Model saved to: {save_path}")



def create_temporary_sanskrit_config(config) -> str:
    """
    Create a temporary sanskrit_config.yaml file for the Tokenizers project.
    
    Args:
        config (dict): The main Gemma config dictionary
        
    Returns:
        str: Path to the temporary config file
    """

    
    # Create temporary config content based on Gemma config
    gen_config = config.get('vocabulary_generation', {})
    
    temp_config = {
        'data': {
            'path': gen_config.get('data_path', config.get('data', {}).get('file_paths', [None])[0] if config.get('data', {}).get('file_paths') else '/home/orrz/gpufs/projects/gemma3/sanskrit_data'),
            'debug': 0
        },
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
    
    # finally:
        # Clean up temporary config file
        # try:
        #     if os.path.exists(temp_config_path):
        #         os.remove(temp_config_path)
        #         logging.info(f"🧹 Cleaned up temporary config: {temp_config_path}")
        # except Exception as e:
        #     logging.warning(f"⚠️ Failed to clean up temporary config: {e}")
    

def main():
    """Main training function with EEVE support"""
    
    print("Starting Gemma-3 Sanskrit Continual Pretraining with EEVE Support...")
    print("="*60)
    
    # # Add argument parser for config file
    # parser = argparse.ArgumentParser(description='Gemma-3 Sanskrit Training')
    # parser.add_argument('--config', type=str, default='gemma_config.yaml',
    #                    help='Path to configuration file (default: gemma_config.yaml)')
    # args = parser.parse_args()
    
    # # Load config from specified file
    # global config
    # config_path = Path(args.config)
    # if not config_path.exists():
    #     print(f"❌ Config file not found: {config_path}")
    #     sys.exit(1)
    
    # with open(config_path, 'r') as file:
    #     config = yaml.safe_load(file)
    # print(f"✅ Config loaded from: {config_path}")


    try:
        # Create necessary directories
        logging.info("Creating required directories...")
        create_directories(config)
        logging.info("✅ Directories created successfully")

        # Run vocabulary generation if enabled
        if config.get('vocabulary_generation', {}).get('enable', False):
            logging.info("🔄 Running vocabulary generation...")
            print("🔄 Running vocabulary generation...")
            run_vocabulary_generation(config)
            logging.info("✅ Vocabulary generation completed (if enabled)")
        
        # Determine training mode
        if config.get('eeve', {}).get('enable', False):
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
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Training failed: {e}")
        
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Force stdoutd
            logging.FileHandler('debug.log')     # Also save to file
        ],
        force=True  # Override any existing configuration
    )
    
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
        print(f"Current CUDA device: {torch.cuda.current_device()}")
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