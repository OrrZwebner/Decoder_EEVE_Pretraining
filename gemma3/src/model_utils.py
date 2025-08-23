#!/usr/bin/env python3
"""
Model loading and compatibility utilities for Gemma-3 Sanskrit training
Location: /home/orrz/gpufs/projects/gemma3/src/model_utils.py
"""

import os
import torch
import torch._dynamo
import logging
import functools
import pickle
from pathlib import Path
from typing import Tuple, Optional, Any, List, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from typing import Tuple, Optional, Any, List, Union, Dict  # Add Dict to the existing import

# Import accelerate for compatibility patches
try:
    import accelerate
    from accelerate.utils import extract_model_from_parallel
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    logging.warning("Accelerate not available - skipping compatibility patches")

def get_n_added_tokens(config) -> int:
    """Get number of added tokens from config"""
    vocab_gen_config = config.get('vocabulary_generation', {})
    return vocab_gen_config.get('num_tokens', 0)


def apply_accelerate_compatibility_patch() -> None:
    """
    Apply compatibility patch for accelerate keep_torch_compile issue
    This fixes the compatibility issue mentioned in the original script
    """
    if not ACCELERATE_AVAILABLE:
        logging.warning("Accelerate not available - skipping compatibility patch")
        return
    
    try:
        # Store original unwrap_model method
        original_unwrap = accelerate.Accelerator.unwrap_model

        @functools.wraps(original_unwrap)
        def patched_unwrap_model(self, model, keep_fp32_wrapper=True, **kwargs):
            """Patched unwrap_model that ignores keep_torch_compile parameter"""
            # Remove keep_torch_compile from kwargs if present
            kwargs.pop('keep_torch_compile', None)
            return original_unwrap(self, model, keep_fp32_wrapper)

        # Apply the patch
        accelerate.Accelerator.unwrap_model = patched_unwrap_model
        
        logging.info("✅ Applied accelerate compatibility patch")
        
    except Exception as e:
        logging.error(f"Failed to apply accelerate compatibility patch: {e}")


def apply_torch_compilation_fixes(config) -> None:
    """
    Apply PyTorch compilation and hardware optimization fixes based on configuration.
    
    This function configures PyTorch's internal settings for optimal performance and 
    compatibility, especially important for large language model training on modern 
    hardware like RTX 3090. It handles several critical optimizations:
    
    1. **Dynamo Error Suppression**: PyTorch's dynamo compiler can sometimes fail on 
       complex models like Gemma-3. When enabled, this suppresses compilation errors 
       and falls back to eager execution instead of crashing.
    
    2. **Flash Attention (SDPA)**: Scaled Dot Product Attention optimization that can 
       significantly speed up transformer training and reduce memory usage. However, 
       it may cause compatibility issues with some model architectures.
    
    3. **Matrix Multiplication Precision**: Controls the precision used for float32 
       matrix operations on Ampere GPUs (RTX 30xx/40xx). Options:
       - "highest": Most accurate but slower
       - "high": Good balance (recommended)
       - "medium": Fastest but less accurate
    
    4. **CUDA Cache Management**: Clears GPU memory cache at startup to ensure 
       maximum available memory for training.
    
    Args:
        config (dict): Configuration dictionary containing hardware optimization settings.
                      Expected structure:
                      {
                          'hardware': {
                              'suppress_dynamo_errors': bool,      # Enable dynamo error suppression
                              'enable_flash_sdp': bool,           # Enable Flash Attention SDPA
                              'set_float32_matmul_precision': str, # "highest", "high", or "medium"
                              'empty_cache_on_start': bool        # Clear CUDA cache on startup
                          }
                      }
    
    Returns:
        None
    
    Raises:
        Exception: Logs any errors but continues execution to avoid blocking training.
    
    Example:
        config = {
            'hardware': {
                'suppress_dynamo_errors': True,
                'enable_flash_sdp': False,  # Disable if causing issues
                'set_float32_matmul_precision': 'high',
                'empty_cache_on_start': True
            }
        }
        apply_torch_compilation_fixes(config)
    
    Note:
        This function should be called early in the model loading pipeline, before 
        loading the actual model but after PyTorch imports. It's automatically called 
        by the load_model_pipeline() function.
    """
    try:
        hardware_config = config.get('hardware', {})
        
        # 1. Apply dynamo error suppression
        # PyTorch 2.0+ introduced torch.compile() which uses dynamo for graph compilation.
        # Some complex models like Gemma-3 can cause compilation failures.
        if hardware_config.get('suppress_dynamo_errors', False):
            torch._dynamo.config.suppress_errors = True
            logging.info("✅ Applied dynamo error suppression - compilation errors will be ignored")
            logging.info("   Models will fall back to eager execution if compilation fails")
        
        # 2. Apply Flash Attention (Scaled Dot Product Attention) settings
        # Flash Attention can provide significant speedup for transformer models
        # but may cause compatibility issues with some model configurations
        if 'enable_flash_sdp' in hardware_config:
            enable_flash = hardware_config['enable_flash_sdp']
            torch.backends.cuda.enable_flash_sdp(enable_flash)
            status = "enabled" if enable_flash else "disabled"
            logging.info(f"✅ Flash SDP (Scaled Dot Product Attention) {status}")
            
            if enable_flash:
                logging.info("   Flash Attention will be used for faster transformer computation")
            else:
                logging.info("   Flash Attention disabled - using standard attention implementation")
        
        # 3. Set matrix multiplication precision for Ampere GPUs
        # This affects performance vs accuracy tradeoff for float32 operations
        if 'set_float32_matmul_precision' in hardware_config:
            precision = hardware_config['set_float32_matmul_precision']
            
            # Validate precision setting
            valid_precisions = ['highest', 'high', 'medium']
            if precision not in valid_precisions:
                logging.warning(f"Invalid matmul precision '{precision}'. Using 'high' instead.")
                logging.warning(f"Valid options: {valid_precisions}")
                precision = 'high'
            
            torch.set_float32_matmul_precision(precision)
            logging.info(f"✅ Set float32 matmul precision to '{precision}'")
            
            # Provide context about the setting
            precision_info = {
                'highest': 'Maximum accuracy, slower performance',
                'high': 'Good balance of accuracy and performance (recommended)',
                'medium': 'Faster performance, reduced accuracy'
            }
            logging.info(f"   {precision_info.get(precision, 'Unknown precision level')}")
        
        # 4. Clear CUDA memory cache on startup
        # This ensures maximum available GPU memory for training
        if torch.cuda.is_available() and hardware_config.get('empty_cache_on_start', False):
            # Get memory info before clearing
            device = torch.cuda.current_device()
            memory_before = torch.cuda.memory_reserved(device) / 1024**3
            
            torch.cuda.empty_cache()
            
            # Get memory info after clearing
            memory_after = torch.cuda.memory_reserved(device) / 1024**3
            memory_freed = memory_before - memory_after
            
            logging.info("✅ Cleared CUDA memory cache")
            logging.info(f"   Memory freed: {memory_freed:.2f} GB")
            logging.info(f"   Available GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        
        # Log final hardware configuration summary
        logging.info("🔧 PyTorch hardware optimization configuration applied:")
        logging.info(f"   Dynamo error suppression: {hardware_config.get('suppress_dynamo_errors', False)}")
        logging.info(f"   Flash SDP enabled: {hardware_config.get('enable_flash_sdp', 'not configured')}")
        logging.info(f"   Matmul precision: {hardware_config.get('set_float32_matmul_precision', 'default')}")
        logging.info(f"   Cache cleared on start: {hardware_config.get('empty_cache_on_start', False)}")
            
    except Exception as e:
        logging.error(f"❌ Error applying PyTorch compilation fixes: {e}")
        logging.error("Training will continue but performance may be suboptimal")
        logging.error("Check your hardware configuration settings in the YAML file")


def load_custom_vocabulary(vocab_file_path: str, config) -> List[str]:
    """
    Load custom vocabulary tokens from PKL file
    
    Args:
        vocab_file_path: Full path to the vocabulary PKL file
        config: Configuration dictionary
        
    Returns:
        List of custom tokens to add to tokenizer
        
    Raises:
        ValueError: If vocabulary file cannot be loaded or has invalid format
    """
    vocab_config = config.get('vocabulary', {})
    debug_vocab = vocab_config.get('debug_vocabulary', True)
    
    if debug_vocab:
        logging.info(f"Loading custom vocabulary from: {vocab_file_path}")
    
    try:
        with open(vocab_file_path, 'rb') as f:
            vocabulary_data = pickle.load(f)
        
        # Handle different possible formats of vocabulary data
        if isinstance(vocabulary_data, list):
            # Simple list of tokens
            custom_tokens = vocabulary_data
        elif isinstance(vocabulary_data, dict):
            # Dictionary format - try common keys
            if 'tokens' in vocabulary_data:
                custom_tokens = vocabulary_data['tokens']
            elif 'vocabulary' in vocabulary_data:
                custom_tokens = vocabulary_data['vocabulary']
            elif 'custom_tokens' in vocabulary_data:
                custom_tokens = vocabulary_data['custom_tokens']
            else:
                # If no recognized key, try to extract all string values
                custom_tokens = [v for v in vocabulary_data.values() if isinstance(v, str)]
                if not custom_tokens:
                    raise ValueError(f"No recognized token key found in vocabulary dict. "
                                   f"Expected keys: 'tokens', 'vocabulary', 'custom_tokens'")
        else:
            raise ValueError(f"Invalid vocabulary format. Expected list or dict, got {type(vocabulary_data)}")
        
        # Validate that all tokens are strings
        if not all(isinstance(token, str) for token in custom_tokens):
            raise ValueError("All vocabulary tokens must be strings")
        
        # Remove empty tokens
        custom_tokens = [token for token in custom_tokens if token.strip()]
        
        if not custom_tokens:
            raise ValueError("No valid tokens found in vocabulary file")
        
        if debug_vocab:
            logging.info(f"✅ Loaded {len(custom_tokens)} custom tokens")
            logging.info(f"Sample tokens: {custom_tokens[:5]}{'...' if len(custom_tokens) > 5 else ''}")
        
        return custom_tokens
        
    except pickle.PickleError as e:
        raise ValueError(f"Failed to load vocabulary PKL file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading vocabulary file {vocab_file_path}: {e}")

def load_model_pipeline(config, stage: int = 0, load_from_previous: bool = False):
    """
    Complete model loading pipeline with vocabulary expansion and stage support
    
    Args:
        config (dict): Configuration dictionary
        stage (int): Current EEVE training stage (0 for standard training)
        load_from_previous (bool): Whether to load from previous stage checkpoint
        
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Loaded model and tokenizer
    """
    logging.info(f"Starting model loading pipeline for stage {stage}...")
    
    # Apply compatibility patches first
    compatibility_config = config.get('compatibility', {})
    if compatibility_config.get('apply_accelerate_patch', False):
        apply_accelerate_compatibility_patch()
    
    # Apply PyTorch compilation fixes - for example, dynamo error suppression
    apply_torch_compilation_fixes(config)
    
    # Load tokenizer (potentially with vocabulary expansion)
    tokenizer = load_tokenizer(config)
        
    # Determine how to load the model
    if load_from_previous and stage > 1:
        # Load BOTH model and tokenizer from previous stage checkpoint
        prev_stage_path = get_stage_checkpoint_path(config, stage - 1)
        logging.info(f"Loading model and tokenizer from previous stage: {prev_stage_path}")
        
        model, tokenizer = load_model_from_stage(prev_stage_path, config)
        
        if model is None or tokenizer is None:
            logging.error(f"Failed to load from stage {stage - 1}, loading fresh model instead")
            tokenizer = load_tokenizer(config)
            model = load_model(config)
    else:
        # Load fresh model and tokenizer
        tokenizer = load_tokenizer(config)
        model = load_model(config)

    # Only handle vocabulary expansion for stage 1 or standard training
    if stage <= 1 and not load_from_previous:
        vocab_config = config.get('vocabulary', {})
        if vocab_config.get('use_custom_vocabulary', False):
            # Check if resizing is needed
            model_vocab_size = model.get_input_embeddings().weight.shape[0]
            tokenizer_vocab_size = len(tokenizer)
            
            if tokenizer_vocab_size > model_vocab_size:
                tokens_added = tokenizer_vocab_size - model_vocab_size
                logging.info(f"Resizing model embeddings for expanded vocabulary...")
                model = resize_model_embeddings(model, tokenizer, tokens_added, config)

    # Apply layer freezing AFTER model resizing if this is an EEVE stage
    if stage > 0 and 'eeve' in config and config['eeve'].get('enable', False):
        stages_config = config['eeve'].get('stages', {})
        stage_config = stages_config.get(stage, {})
        if stage_config:
            train_layers = stage_config.get('train_layers', 'all')
            n_added_tokens = get_n_added_tokens(config)
            
            logging.info(f"🎯 Stage {stage}: Applying layer freezing - {train_layers}")
            freeze_layers(model, train_layers, n_added_tokens, tokenizer, config)

            # Verify gradient masking for added_tokens_embeddings mode
            if train_layers == 'added_tokens_embeddings':
                verification = verify_gradient_masking(model, stage)
                
                # Store verification results in model metadata for later use
                model._gradient_masking_verification = verification
    
    # Setup model for training
    model = setup_model_for_training(model, config)
    
    # Validate device compatibility
    validate_model_device_compatibility(model, config)
    
    # Print model info if debug mode
    print_model_info(model, tokenizer, config)
    
    logging.info("✅ Model loading pipeline completed")
    
    return model, tokenizer



def expand_tokenizer_vocabulary(tokenizer: AutoTokenizer, custom_tokens: List[str], config) -> Tuple[AutoTokenizer, int]:
    """
    Expand tokenizer vocabulary with custom tokens
    
    Args:
        tokenizer: Original tokenizer
        custom_tokens: List of custom tokens to add
        config: Configuration dictionary
        
    Returns:
        Tuple of (expanded_tokenizer, number_of_tokens_added)
    """
    vocab_config = config['vocabulary']
    debug_vocab = vocab_config.get('debug_vocabulary', True)
    add_method = vocab_config.get('add_tokens_method', 'add')
    
    original_vocab_size = len(tokenizer)
    
    if debug_vocab:
        logging.info(f"Expanding tokenizer vocabulary using method: {add_method}")
        logging.info(f"Original vocabulary size: {original_vocab_size:,}")
    
    # Filter out tokens that already exist in tokenizer
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in custom_tokens if token not in existing_tokens]
    
    if len(new_tokens) < len(custom_tokens):
        skipped_count = len(custom_tokens) - len(new_tokens)
        if debug_vocab:
            logging.info(f"Skipped {skipped_count} tokens that already exist in tokenizer")
    
    if not new_tokens:
        logging.warning("No new tokens to add - all custom tokens already exist in tokenizer")
        return tokenizer, 0
    
    # Add new tokens based on method
    if add_method == 'add':
        # Add tokens to the end of vocabulary
        num_added = tokenizer.add_tokens(new_tokens)
        
        if debug_vocab:
            logging.info(f"Added {num_added} new tokens to tokenizer")
            logging.info(f"New vocabulary size: {len(tokenizer):,}")
            logging.info(f"Sample new tokens: {new_tokens[:5]}{'...' if len(new_tokens) > 5 else ''}")
        
        return tokenizer, num_added
    
    else:
        raise ValueError(f"Unsupported add_tokens_method: {add_method}")


def resize_model_embeddings(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                          tokens_added: int, config) -> AutoModelForCausalLM:
    """
    Resize model embeddings to accommodate new vocabulary tokens
    
    Args:
        model: Model to resize
        tokenizer: Tokenizer with expanded vocabulary
        tokens_added: Number of tokens that were added
        config: Configuration dictionary
        
    Returns:
        Model with resized embeddings
    """
    if tokens_added == 0:
        logging.info("No tokens were added - skipping embedding resize")
        return model
    
    vocab_config = config['vocabulary']
    debug_vocab = vocab_config.get('debug_vocabulary', True)
    should_resize = vocab_config.get('resize_model_embeddings', True)
    
    if not should_resize:
        logging.warning("resize_model_embeddings is disabled - model may not work with new tokens")
        return model
    
    original_embed_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)
    
    if debug_vocab:
        logging.info(f"Resizing model embeddings:")
        logging.info(f"  Original embedding size: {original_embed_size:,}")
        logging.info(f"  New vocabulary size: {new_vocab_size:,}")
        logging.info(f"  Tokens added: {tokens_added}")
    
    # Resize token embeddings
    model.resize_token_embeddings(new_vocab_size)
    
    # Log memory usage change if CUDA is available
    if torch.cuda.is_available() and debug_vocab:
        device = next(model.parameters()).device
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        logging.info(f"  GPU memory after resize: {memory_allocated:.2f} GB")
    
    if debug_vocab:
        final_embed_size = model.get_input_embeddings().weight.shape[0]
        logging.info(f"✅ Model embeddings resized to {final_embed_size:,}")
    
    return model



def load_authentication_token(config) -> Optional[str]:
    """
    Load HuggingFace authentication token using HF API or file fallback
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Authentication token string or None
    """
    auth_config = config.get('authentication', {})
    
    # Try HF API first (improved method)
    if auth_config.get('use_hf_api', True):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            token = api.token
            if token:
                logging.info("✅ Using HF CLI authentication token")
                return token
        except Exception as e:
            logging.warning(f"HF API authentication failed: {e}")
    
    # Fallback to file-based token
    token_path = auth_config.get('token_path')
    if not token_path:
        token_path = config.get('environment', {}).get('token_path')
    
    if token_path:
        try:
            token_path = Path(token_path)
            if token_path.exists():
                with open(token_path, 'r') as f:
                    token = f.read().strip()
                logging.info("✅ Using file-based authentication token")
                return token
            else:
                logging.warning(f"Token file not found: {token_path}")
        except Exception as e:
            logging.error(f"Error loading authentication token: {e}")
    
    logging.warning("No authentication token found - you may need to run: huggingface-cli login")
    return None


def load_model_from_stage(stage_path: str, config) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    Load model AND tokenizer from a previous EEVE training stage.
    
    Args:
        stage_path: Path to the stage checkpoint directory
        config: Configuration dictionary
        
    Returns:
        Tuple of (loaded_model, loaded_tokenizer) or (None, None) if loading fails
    """
    # Define the model path based on stage
    model_path = os.path.join(stage_path, "final_model")


    if not os.path.exists(model_path):
        logging.error(f"Stage checkpoint not found: {model_path}")
        return None, None
    
    try:
        logging.info(f"Loading model from stage: {model_path}")
        
        # Get device and dtype settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = config.get('model', {})
        torch_dtype = get_torch_dtype(model_config.get('torch_dtype', 'bfloat16'))
        # Load the model from checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_config.get('trust_remote_code', True),
            use_cache=model_config.get('use_cache', False),
            attn_implementation=model_config.get('attn_implementation', 'eager'),
            low_cpu_mem_usage=model_config.get('low_cpu_mem_usage', True),
            local_files_only=True
        ).to(device)
        
        # ALSO load the tokenizer from the checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        logging.info(f"✅ Model loaded with vocab size: {model.get_input_embeddings().weight.shape[0]}")
        logging.info(f"✅ Tokenizer loaded with vocab size: {len(tokenizer)}")
        
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Failed to load model/tokenizer from stage {stage_path}: {e}")
        return None, None


def get_stage_checkpoint_path(config, stage: int, run_name: Optional[str] = None) -> str:
    """
    Get the checkpoint path for a specific EEVE stage
    
    Args:
        config: Configuration dictionary
        stage: Stage number (0 for standard training, 1-7 for EEVE)
        run_name: Optional run name override
        
    Returns:
        Path to stage checkpoint directory
    """
    # Get metadata for directory naming
    n_tokens = config.get('vocabulary_generation', {}).get('num_tokens', 0)
    algorithm = config.get('vocabulary_generation', {}).get('algorithm_target', 'unknown')
    timestamp = config.get('_training_timestamp', 'no_timestamp')
    
    # Construct main directory name
    main_dir = f"{algorithm}_{n_tokens}_{timestamp}"
    
    if stage > 0:
        # EEVE training - use stage-specific subdirectory
        eeve_config = config.get('eeve', {})
        base_dir = eeve_config.get('stage_output_dir', 
                                os.path.join(config['project']['output_dir'], "eeve_stages"))
        stage_path = os.path.join(base_dir, main_dir, f"stage{stage}")
    else:
        # Standard training - use outputs directory
        base_dir = config['project']['output_dir']
        stage_path = os.path.join(base_dir, main_dir)
    
    return stage_path


def load_tokenizer(config) -> AutoTokenizer:
    """
    Load and configure tokenizer with optional vocabulary expansion
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AutoTokenizer (potentially with expanded vocabulary)
    """
    model_config = config['model']
    auth_token = load_authentication_token(config)
    
    logging.info(f"Loading tokenizer: {model_config['name']}")
    
    try:
        # Load base tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            cache_dir=None,
            # local_files_only=getattr(config.compatibility, 'local_files_only', False),
            local_files_only=config.get('compatibility', {}).get('local_files_only', False),
            trust_remote_code=model_config.get('trust_remote_code', True),
            token=auth_token
        )
        
        # Configure pad token if needed
        tokenizer_config = config['tokenizer']
        if tokenizer_config.get('add_pad_token', True) and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info("Added pad token to tokenizer")
        
        logging.info(f"✅ Base tokenizer loaded successfully")
        logging.info(f"Original vocab size: {len(tokenizer):,}")
        
        # Handle custom vocabulary expansion if enabled
        if 'vocabulary' in config:
            vocab_config = config['vocabulary']
            if vocab_config.get('use_custom_vocabulary', False):
                logging.info("Custom vocabulary enabled - expanding tokenizer")
                
                # Load custom tokens from PKL file
                str_num_tokens = str(config.get('vocabulary_generation', {}).get('num_tokens', ""))
                vocab_file_path = vocab_config['vocabulary_full_path'] + str_num_tokens +'.pkl'
                custom_tokens = load_custom_vocabulary(vocab_file_path, config)
                
                # Expand tokenizer vocabulary
                tokenizer, tokens_added = expand_tokenizer_vocabulary(tokenizer, custom_tokens, config)
                
                if tokens_added > 0:
                    logging.info(f"✅ Tokenizer vocabulary expanded with {tokens_added} new tokens")
                    logging.info(f"Final tokenizer vocab size: {len(tokenizer):,}")
                else:
                    logging.info("No new tokens were added to tokenizer")
            else:
                logging.info("Custom vocabulary disabled - using original tokenizer")
        
        return tokenizer
        
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise


def load_model(config) -> AutoModelForCausalLM:
    """
    Load and configure Gemma-3 model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AutoModelForCausalLM
    """
    model_config = config['model']
    auth_token = load_authentication_token(config)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert dtype string to torch dtype
    torch_dtype = get_torch_dtype(model_config.get('torch_dtype', 'bfloat16'))
    
    logging.info(f"Loading model: {model_config['name']}")
    logging.info(f"Device: {device}")
    logging.info(f"Dtype: {torch_dtype}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            torch_dtype=torch_dtype,
            # device_map="auto",
            # load_in_8bit=True,
            cache_dir=None,
            ignore_mismatched_sizes=model_config.get('ignore_mismatched_sizes', True),
            local_files_only=config.get('compatibility', {}).get('local_files_only', False),
            trust_remote_code=model_config.get('trust_remote_code', True),
            use_cache=model_config.get('use_cache', False),
            attn_implementation=model_config.get('attn_implementation', 'eager'),
            low_cpu_mem_usage=model_config.get('low_cpu_mem_usage', True),
            token=auth_token
            
        ).to(device)
        
        logging.info(f"✅ Model loaded successfully")
        
        # Log memory usage if CUDA is available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            logging.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
            
            if config.get('hardware', {}).get('monitor_gpu_memory', False): 
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                logging.info(f"GPU memory reserved: {memory_reserved:.2f} GB")
        
        return model
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


def print_model_info(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config) -> None:
    """
    Print detailed model information
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Configuration dictionary
    """
    if not config.get('debug', {}).get('print_model_info', False):
        return
    
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    # Basic model info
    print(f"Model name: {config['model']['name']}")
    print(f"Model type: {type(model).__name__}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Model size info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Tokenizer info
    print(f"\nTokenizer information:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print(f"  EOS token: {tokenizer.eos_token}")
    print(f"  BOS token: {tokenizer.bos_token}")
    
    # Memory info
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        print(f"\nGPU Memory (Device {device}):")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved: {memory_reserved:.2f} GB")
        print(f"  Total: {memory_total:.2f} GB")
        print(f"  Free: {memory_total - memory_reserved:.2f} GB")
    
    # Configuration info
    print(f"\nModel configuration:")
    print(f"  Use cache: {config['model'].get('use_cache', False)}")
    print(f"  Attention implementation: {config['model'].get('attn_implementation', 'eager')}")
    print(f"  Low CPU memory usage: {config['model'].get('low_cpu_mem_usage', True)}")

    
    print("="*50 + "\n")


def setup_model_for_training(model: AutoModelForCausalLM, config) -> AutoModelForCausalLM:
    """
    Configure model for training
    
    Args:
        model: Loaded model
        config: Configuration dictionary
        
    Returns:
        Configured model
    """
    # Enable gradient checkpointing if configured
    training_config = config['training']
    if training_config.get('gradient_checkpointing', False): 
        model.gradient_checkpointing_enable()
        logging.info("✅ Gradient checkpointing enabled")
    
    # Set model to training mode
    model.train()
    
    return model


def validate_model_device_compatibility(model: AutoModelForCausalLM, config) -> None:
    """
    Validate that model is on the correct device
    
    Args:
        model: Loaded model
        config: Configuration dictionary
    """
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_device = next(model.parameters()).device
    
    if model_device != expected_device:
        logging.warning(f"Model device mismatch: expected {expected_device}, got {model_device}")
    else:
        logging.info(f"✅ Model device validation passed: {model_device}")


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype
    
    Args:
        dtype_str: String representation of dtype
        
    Returns:
        torch.dtype object
    """
    dtype_mapping = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }
    
    return dtype_mapping.get(dtype_str.lower(), torch.bfloat16)



def freeze_layers(model: AutoModelForCausalLM, train_layers: str = "all", 
                  n_added_tokens: int = 0, tokenizer: Optional[AutoTokenizer] = None, 
                  config = None) -> None:
    """
    Freeze or unfreeze layers in the model based on the train_layers parameter.
    Updated for Gemma-3 architecture with working gradient hook approach.
    
    Args:
        model: The PyTorch model (AutoModelForCausalLM - Gemma-3)
        train_layers: Options:
                      - "all": Train all layers
                      - "added_tokens_embeddings": Train only embeddings of added tokens (uses gradient masking)
                      - "added_tokens_lm_head": NOT APPLICABLE for Gemma-3 (tied embeddings)
                      - "embedding": Train entire embedding layer  
                      - "lm_head": NOT APPLICABLE for Gemma-3 (tied embeddings)
                      - "hidden_layers": Train only internal transformer layers
                      - Comma-separated string of specific layers to train
        n_added_tokens: Number of tokens added to the tokenizer
        tokenizer: The tokenizer (needed for added tokens options)
        config: Configuration dictionary for debug logging
    """
    logging.info(f"🔧 Freezing layers based on train_layers='{train_layers}'")
    debug_vocab = False
    if config and 'vocabulary' in config:
        debug_vocab = config['vocabulary'].get('debug_vocabulary', False)
    
    # Handle "all" case first
    if train_layers == "all":
        if debug_vocab:
            logging.info("🔥 Training all layers (no freezing applied)")
        for param in model.parameters():
            param.requires_grad = True
        return
    
    # Freeze all parameters initially for selective training
    if debug_vocab:
        logging.info("🔒 Freezing all parameters for selective training...")
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Find Gemma-3 embedding parameter and detect tied embeddings
    embed_param = None
    embed_name = None
    lm_head_param = None
    embeddings_tied = False
    
    for name, param in model.named_parameters():
        if name == 'model.embed_tokens.weight':
            embed_param = param
            embed_name = name
            if debug_vocab:
                logging.info(f"📥 Found embedding parameter: {name} - {param.shape}")
        elif name.endswith('lm_head.weight'):
            lm_head_param = param
            if debug_vocab:
                logging.info(f"📤 Found LM head parameter: {name} - {param.shape}")
    
    # Determine if embeddings are tied
    if lm_head_param is None:
        embeddings_tied = True
        if debug_vocab:
            logging.info("🔗 TIED EMBEDDINGS detected (Gemma-3 architecture)")
            logging.info(f"   Matrix: {embed_name} serves both input and output functions")
    else:
        if debug_vocab:
            logging.info("🔄 SEPARATE EMBEDDINGS detected")
    
    if embed_param is None:
        logging.error("Could not find model.embed_tokens.weight parameter!")
        return
    
    # Parse training configuration
    layers_to_train = [layer.strip() for layer in train_layers.split(",")]
    
    for layer_option in layers_to_train:
        
        if layer_option == "added_tokens_embeddings":
            if n_added_tokens <= 0:
                logging.warning(f"'{layer_option}' specified but n_added_tokens={n_added_tokens}!")
                continue
            
            vocab_size = embed_param.size(0)
            embed_dim = embed_param.size(1)
            
            if debug_vocab:
                logging.info(f"🎯 Setting up selective training for {n_added_tokens} added tokens")
                logging.info(f"   Target matrix: {embed_name}")
                logging.info(f"   Full matrix dimensions: {embed_param.shape}")
                logging.info(f"   Training region: rows [{vocab_size-n_added_tokens}:{vocab_size}]")
                logging.info(f"   Training dimensions: [{n_added_tokens}, {embed_dim}]")
            
            # WORKING GRADIENT HOOK APPROACH (from successful test)
            # Create mask for added tokens only
            mask = torch.zeros_like(embed_param, dtype=torch.bool, device=embed_param.device)
            mask[-n_added_tokens:] = True  # Only last n_added_tokens rows
            
            if debug_vocab:
                logging.info(f"   📍 Created gradient mask:")
                logging.info(f"      Mask shape: {mask.shape}")
                logging.info(f"      True values: {mask.sum().item()} (should be {n_added_tokens * embed_dim})")
                logging.info(f"      Mask region: rows [{vocab_size-n_added_tokens}:{vocab_size}], all columns")
            
            # MUST enable gradients BEFORE registering hook
            embed_param.requires_grad = True
            
            if debug_vocab:
                logging.info(f"   ✅ Enabled requires_grad=True (required for hook registration)")
            
            # Create gradient hook with dtype preservation (CRITICAL FIX)
            hook_call_count = [0]
            
            def gradient_hook(grad):
                """Apply gradient masking to restrict updates to new tokens only."""
                hook_call_count[0] += 1
                if debug_vocab and hook_call_count[0] <= 3:
                    logging.info(f"   🪝 Gradient hook called (call #{hook_call_count[0]})")
                    if grad is not None:
                        logging.info(f"      Input grad shape: {grad.shape}")
                        logging.info(f"      Input grad dtype: {grad.dtype}")
                        logging.info(f"      Input grad norm: {grad.norm().item():.6f}")
                        logging.info(f"      Non-zero elements: {(grad != 0).sum().item()}")
                
                if grad is not None:
                    # CRITICAL: Preserve dtype by matching gradient's dtype
                    mask_same_dtype = mask.to(dtype=grad.dtype, device=grad.device)
                    masked_grad = grad * mask_same_dtype
                    
                    if debug_vocab and hook_call_count[0] <= 3:
                        logging.info(f"      Mask dtype: {mask_same_dtype.dtype} (matches grad: {grad.dtype})")
                        logging.info(f"      Masked grad dtype: {masked_grad.dtype}")
                        logging.info(f"      Masked grad norm: {masked_grad.norm().item():.6f}")
                        logging.info(f"      Masked non-zero: {(masked_grad != 0).sum().item()}")
                    
                    return masked_grad
                return grad
            
            # Register the hook
            hook_handle = embed_param.register_hook(gradient_hook)
            
            if debug_vocab:
                logging.info(f"   ✅ Registered gradient hook on {embed_name}")
            
            # Store metadata for custom parameter counting
            embed_param._added_tokens_only = True
            embed_param._n_added_tokens = n_added_tokens
            embed_param._embed_dim = embed_dim
            embed_param._gradient_hook_handle = hook_handle
            embed_param._hook_call_count = hook_call_count
            
            if debug_vocab:
                actual_trainable = n_added_tokens * embed_dim
                logging.info(f"   ✅ Setup complete for selective training")
                logging.info(f"   Expected effective parameters: {actual_trainable:,}")
                logging.info(f"   ⚠️  PyTorch will count ALL embedding params as trainable")
                logging.info(f"   ⚠️  But gradient hook will mask updates to original tokens")
        
        elif layer_option in ["added_tokens_lm_head", "lm_head"]:
            if embeddings_tied:
                logging.warning(f"'{layer_option}' is not applicable to Gemma-3 (tied embeddings)")
                logging.info("   Use 'added_tokens_embeddings' for added token training")
                logging.info(f"   The tied matrix {embed_name} serves both input and output functions")
            else:
                logging.warning(f"'{layer_option}' not implemented for separate embeddings")
                logging.info("   This would require separate LM head gradient masking")
            continue
        
        elif layer_option == "embedding":
            if debug_vocab:
                if embeddings_tied:
                    logging.info("🎯 Training entire tied embedding matrix (input + output)")
                    logging.info(f"   Matrix: {embed_name}")
                    logging.info(f"   Dimensions: {embed_param.shape}")
                    logging.info(f"   Function: Input embeddings AND output projection")
                else:
                    logging.info("🎯 Training entire embedding layer (input only)")
                    logging.info(f"   Matrix: {embed_name}")
                    logging.info(f"   Dimensions: {embed_param.shape}")
            
            embed_param.requires_grad = True
            
            if debug_vocab:
                logging.info(f"   Trainable parameters: {embed_param.numel():,}")
        
        elif layer_option == "hidden_layers":
            if debug_vocab:
                logging.info("🎯 Training only hidden/internal transformer layers")
            
            hidden_params = 0
            for name, param in model.named_parameters():
                # Include transformer layers and final norm, exclude embeddings
                if 'model.layers.' in name or name == 'model.norm.weight':
                    param.requires_grad = True
                    hidden_params += param.numel()
                    if debug_vocab and 'model.layers.0.' in name:  # Show sample from first layer
                        component = '.'.join(name.split('.')[3:])
                        logging.info(f"   Enabled: {component} - {param.shape}")
            
            if debug_vocab:
                logging.info(f"   Total hidden layer parameters: {hidden_params:,}")
                logging.info(f"   Components: Transformer layers (0-25) + final norm")
        
        else:
            # Handle specific layer patterns
            if debug_vocab:
                logging.info(f"🎯 Training layers matching pattern: '{layer_option}'")
            
            matched_params = 0
            matched_count = 0
            for name, param in model.named_parameters():
                if layer_option in name:
                    param.requires_grad = True
                    matched_params += param.numel()
                    matched_count += 1
                    if debug_vocab and matched_count <= 3:
                        logging.info(f"   Enabled: {name} - {param.shape}")
            
            if debug_vocab:
                if matched_params > 0:
                    logging.info(f"   Matched {matched_count} parameters: {matched_params:,} trainable")
                    if matched_count > 3:
                        logging.info(f"   ... and {matched_count - 3} more")
                else:
                    logging.warning(f"   No parameters matched pattern: '{layer_option}'")
    
    # Enhanced parameter counting and logging
    pytorch_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate effective trainable parameters (respects gradient masking)
    effective_trainable = 0
    masked_params_info = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if hasattr(param, '_added_tokens_only') and param._added_tokens_only:
                # This parameter uses gradient masking
                effective_count = param._n_added_tokens * param._embed_dim
                effective_trainable += effective_count
                masked_params_info.append({
                    'name': name,
                    'pytorch_count': param.numel(),
                    'effective_count': effective_count,
                    'mask_type': 'added_tokens_only'
                })
            else:
                # Regular parameter - count all
                effective_trainable += param.numel()
    
    # Primary logging (always shown)
    if masked_params_info:
        logging.info(f"🔥 PyTorch reports: {pytorch_trainable:,} trainable parameters ({pytorch_trainable/total_params:.2%}) ⚠️")
        logging.info(f"🎯 Actually training: {effective_trainable:,} parameters ({effective_trainable/total_params:.2%}) ✅")
        logging.info(f"📊 Gradient-masked: {pytorch_trainable - effective_trainable:,} parameters")
    else:
        logging.info(f"🔥 Training {pytorch_trainable:,} out of {total_params:,} parameters ({pytorch_trainable/total_params:.2%})")
    
    # Detailed logging (debug mode)
    if debug_vocab:
        logging.info(f"\n💡 REALITY CHECK:")
        if masked_params_info:
            logging.info(f"   ✅ Gradient hooks will mask {pytorch_trainable - effective_trainable:,} parameters")
            logging.info(f"   ✅ Only {effective_trainable:,} parameters will actually be updated")
            logging.info(f"   ⚠️  Training frameworks will show misleading parameter count")
            logging.info(f"   ⚠️  But the training is mathematically correct!")
            
            logging.info(f"\n🎭 GRADIENT-MASKED PARAMETERS:")
            for info in masked_params_info:
                mask_ratio = info['effective_count'] / info['pytorch_count']
                logging.info(f"   {info['name']}:")
                logging.info(f"      PyTorch count: {info['pytorch_count']:,}")
                logging.info(f"      Effective count: {info['effective_count']:,} ({mask_ratio:.1%})")
                logging.info(f"      Mask type: {info['mask_type']}")
        else:
            logging.info(f"   ✅ No gradient masking applied")
            logging.info(f"   ✅ All {effective_trainable:,} parameters will be trained normally")


def verify_gradient_masking(model, stage: int = 0) -> Dict[str, Any]:
    """
    Verify that gradient masking is working correctly for added tokens.
    
    Args:
        model: The model with potential gradient masking
        stage: Current training stage for logging
        
    Returns:
        Dictionary with verification results and actual trainable parameters
    """
    verification_results = {}
    
    # Find embedding parameter with gradient mask
    embed_param = None
    embed_name = None
    
    for name, param in model.named_parameters():
        if name == 'model.embed_tokens.weight':
            embed_param = param
            embed_name = name
            break
    
    if embed_param is None:
        return {"error": "No embedding parameter found"}
    
    # Check if gradient masking is applied
    if hasattr(embed_param, '_gradient_mask'):
        mask = embed_param._gradient_mask
        n_added_tokens = embed_param._n_added_tokens
        vocab_size, embed_dim = embed_param.shape
        
        # Calculate actual trainable parameters
        mask_true_count = mask.sum().item()
        expected_trainable = n_added_tokens * embed_dim
        
        # Verify mask is correct
        mask_correct = (mask_true_count == expected_trainable)
        
        # Check which rows are masked
        rows_with_gradients = mask.any(dim=1).cpu()  # Which rows have ANY True values
        first_trained_row = rows_with_gradients.nonzero()[0].item() if rows_with_gradients.any() else -1
        last_trained_row = rows_with_gradients.nonzero()[-1].item() if rows_with_gradients.any() else -1
        
        verification_results = {
            "stage": stage,
            "gradient_masking_active": True,
            "total_embedding_params": vocab_size * embed_dim,
            "pytorch_reports_trainable": embed_param.numel() if embed_param.requires_grad else 0,
            "actual_trainable_params": mask_true_count,
            "expected_trainable_params": expected_trainable,
            "mask_is_correct": mask_correct,
            "n_added_tokens": n_added_tokens,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "first_trained_token_idx": first_trained_row,
            "last_trained_token_idx": last_trained_row,
            "trained_token_range": f"[{first_trained_row}:{last_trained_row+1}]"
        }
        
        # Log verification results
        logging.info(f"\n{'='*60}")
        logging.info(f"GRADIENT MASKING VERIFICATION - Stage {stage}")
        logging.info(f"{'='*60}")
        logging.info(f"✅ Gradient masking is ACTIVE")
        logging.info(f"📊 Parameter counting:")
        logging.info(f"   PyTorch reports: {verification_results['pytorch_reports_trainable']:,} (MISLEADING)")
        logging.info(f"   Actually training: {verification_results['actual_trainable_params']:,} (CORRECT)")
        logging.info(f"   Verification: {'PASSED ✅' if mask_correct else 'FAILED ❌'}")
        logging.info(f"📍 Training token range: {verification_results['trained_token_range']}")
        logging.info(f"   Original tokens [0:{first_trained_row}]: FROZEN ❄️")
        logging.info(f"   New tokens [{first_trained_row}:{last_trained_row+1}]: TRAINING 🔥")
        logging.info(f"{'='*60}\n")
    else:
        verification_results = {
            "stage": stage,
            "gradient_masking_active": False,
            "pytorch_reports_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "actual_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    return verification_results


def count_effective_trainable_parameters(model) -> dict:
    """
    Count parameters that will actually be trained (respects gradient masking).
    Use this instead of PyTorch's standard counting for accurate results.
    """
    pytorch_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    effective_trainable = 0
    
    for param in model.parameters():
        if param.requires_grad:
            if hasattr(param, '_added_tokens_only') and param._added_tokens_only:
                # Only count the masked portion
                effective_trainable += param._n_added_tokens * param._embed_dim
            else:
                # Count all parameters
                effective_trainable += param.numel()
    
    return {
        'pytorch_trainable': pytorch_trainable,
        'effective_trainable': effective_trainable,
        'total_params': total_params,
        'pytorch_percentage': pytorch_trainable / total_params * 100,
        'effective_percentage': effective_trainable / total_params * 100,
        'masked_params': pytorch_trainable - effective_trainable
    }
   


if __name__ == "__main__":
    # Test model utilities
    print("Testing model utilities...")
    
    # Test dtype conversion (only if function exists)
    try:
        dtype = get_torch_dtype("bfloat16")
        print(f"Converted dtype: {dtype}")
    except NameError:
        print("get_torch_dtype function not found - skipping test")
    
    # Test compatibility patch
    apply_accelerate_compatibility_patch()
    
    print("Model utilities test completed!")