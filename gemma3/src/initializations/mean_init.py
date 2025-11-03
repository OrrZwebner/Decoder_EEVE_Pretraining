
import json
import math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size

def instantiate_model_by_mean(
    source_model: AutoModelForCausalLM,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    tie_word_embeddings: bool = False,
    pad_to_multiple_of: int = 8
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initialize new token embeddings using mean of constituent tokens."""
    
    print("Initializing new token embeddings using mean of constituent tokens")
    print(f"Source vocab size: {len(source_tokenizer)}, Target vocab size: {len(target_tokenizer)}")

    # Get device and dtype from source model
    device = source_model.get_input_embeddings().weight.device
    dtype = source_model.get_input_embeddings().weight.dtype
    
    # Get source embeddings (keep as PyTorch tensor on GPU)
    source_embeddings = source_model.get_input_embeddings().weight.detach()
    vocab_size = round_to_nearest_multiple(len(target_tokenizer), pad_to_multiple_of)
    print(f"Padding vocab size to multiple of {pad_to_multiple_of}" + 
          f": {len(target_tokenizer)} -> {vocab_size}")
    # Initialize target embeddings (directly on GPU)
    target_embeddings = torch.zeros(
        (vocab_size, source_embeddings.shape[1]),
        dtype=dtype,
        device=device  # Same device as source
    )
    # Copy existing embeddings
    target_embeddings[:source_embeddings.shape[0]] = source_embeddings
    
    # Handle output embeddings if not tied
    if not tie_word_embeddings:
        print("Initializing output projection separately")
        source_head_embeddings = source_model.get_output_embeddings().weight.detach()
        target_head_embeddings = torch.zeros(
            (vocab_size, source_head_embeddings.shape[1]),
            dtype=dtype,
            device=device
        )
        target_head_embeddings[:source_head_embeddings.shape[0]] = source_head_embeddings
    
    # Initialize new token embeddings using mean
    num_initialized = 0
    num_failed = 0
    
    for i in range(len(source_tokenizer), len(target_tokenizer)):
        token = target_tokenizer.convert_ids_to_tokens(i)
        source_tokens = source_tokenizer.tokenize(token)
        source_ids = source_tokenizer.convert_tokens_to_ids(source_tokens)
        
        if len(source_ids) == 0:
            print(f"Warning: Empty tokenization for token '{token}' (id={i})")
            num_failed += 1
            continue
        
        # Compute mean (all operations stay on GPU)
        target_embeddings[i] = source_embeddings[source_ids].mean(dim=0)  # dim=0, not axis=0
        if not tie_word_embeddings:
            target_head_embeddings[i] = source_head_embeddings[source_ids].mean(dim=0)
        
        num_initialized += 1
    
    print(f"Initialized {num_initialized} new tokens using mean embeddings")
    if num_failed > 0:
        print(f"Failed to initialize {num_failed} tokens (kept zero init)")
    
    # Resize model embeddings
    target_model = source_model
    target_model.resize_token_embeddings(len(target_tokenizer), pad_to_multiple_of=pad_to_multiple_of)
    
    # Set embeddings (already on correct device, no .to() needed)
    target_model.get_input_embeddings().weight.data = target_embeddings
    target_model.config.vocab_size = vocab_size
    
    if not tie_word_embeddings:
        target_model.get_output_embeddings().weight.data = target_head_embeddings
    else:
        target_model.tie_weights()
    
    return target_model, target_tokenizer
