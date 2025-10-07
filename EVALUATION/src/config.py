"""
Configuration management
"""

import os
import yaml
from pathlib import Path


def load_config(config_path, cli_args=None):
    """
    Load configuration from YAML file and merge with CLI arguments
    
    Args:
        config_path: Path to YAML config file
        cli_args: Argparse namespace with CLI arguments
        
    Returns:
        Configuration dictionary
    """
    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge CLI overrides if provided
    if cli_args is not None:
        if cli_args.num_samples is not None:
            config['dataset']['num_samples'] = cli_args.num_samples
        
        if cli_args.devices is not None:
            config['compute']['devices'] = cli_args.devices
        
        if cli_args.source_lang is not None:
            config['translation']['source_lang'] = cli_args.source_lang
        
        if cli_args.target_lang is not None:
            config['translation']['target_lang'] = cli_args.target_lang
    
    return config


def validate_config(config):
    """
    Validate configuration
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['provider', 'translation', 'dataset', 'paths']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Check provider type
    provider_type = config['provider']['type']
    valid_providers = ['huggingface', 'openai', 'anthropic']
    if provider_type not in valid_providers:
        raise ValueError(f"Invalid provider type: {provider_type}. Must be one of {valid_providers}")
    
    # Check provider-specific config
    if provider_type not in config['provider']:
        raise ValueError(f"Missing configuration for provider: {provider_type}")
    
    # Check translation languages
    if 'source_lang' not in config['translation']:
        raise ValueError("Missing source_lang in translation config")
    if 'target_lang' not in config['translation']:
        raise ValueError("Missing target_lang in translation config")
    
    # Check paths
    required_paths = ['hf_home', 'output_dir']
    for path_key in required_paths:
        if path_key not in config['paths']:
            raise ValueError(f"Missing required path: {path_key}")


def setup_environment(config):
    """
    Setup environment variables
    CRITICAL: Must be called BEFORE any torch/transformers imports
    
    Args:
        config: Configuration dictionary
    """
    # Set CUDA devices FIRST (before any CUDA initialization)
    if 'compute' in config and 'devices' in config['compute']:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['compute']['devices']
        print(f"Set CUDA_VISIBLE_DEVICES={config['compute']['devices']}")
    
    # Set HuggingFace paths
    if 'hf_home' in config['paths']:
        os.environ["HF_HOME"] = config['paths']['hf_home']
    
    if 'hf_token' in config['paths']:
        token_path = config['paths']['hf_token']
        if os.path.isfile(token_path):
            with open(token_path, 'r') as f:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = f.read().strip()
        else:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token_path
    
    # Disable torch compile
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"