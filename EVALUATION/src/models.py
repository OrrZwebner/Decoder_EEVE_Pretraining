"""
Model/LLM initialization with LangChain
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline


def create_llm(config):
    """
    Create LangChain LLM based on provider configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LangChain LLM instance
    """
    provider_type = config['provider']['type']
    
    if provider_type == 'huggingface':
        return _create_huggingface_llm(config)
    elif provider_type == 'openai':
        return _create_openai_llm(config)
    elif provider_type == 'anthropic':
        return _create_anthropic_llm(config)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


def _create_huggingface_llm(config):
    """
    Create HuggingFace LLM with proper GPU device configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HuggingFacePipeline instance
    """
    hf_config = config['provider']['huggingface']
    model_name = hf_config['model_name']
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Determine dtype
    dtype_str = hf_config.get('torch_dtype', 'float16')
    if dtype_str == 'float16':
        torch_dtype = torch.float16
    elif dtype_str == 'float32':
        torch_dtype = torch.float32
    elif dtype_str == 'bfloat16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    
    # Load model with device_map="auto" to use specified GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    print("Model loaded successfully")
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=hf_config.get('max_new_tokens', 128),
        temperature=hf_config.get('temperature', 0.3),
        top_p=hf_config.get('top_p', 0.9),
        do_sample=hf_config.get('do_sample', True),
        return_full_text=False  # Only return generated part
    )
    
    print("Pipeline created successfully")
    
    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm


def _create_openai_llm(config):
    """
    Create OpenAI LLM (for future use)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ChatOpenAI instance
    """
    from langchain_openai import ChatOpenAI
    
    openai_config = config['provider']['openai']
    
    llm = ChatOpenAI(
        model=openai_config['model_name'],
        temperature=openai_config.get('temperature', 0.3),
        max_tokens=openai_config.get('max_tokens', 128)
    )
    
    return llm


def _create_anthropic_llm(config):
    """
    Create Anthropic LLM (for future use)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ChatAnthropic instance
    """
    from langchain_anthropic import ChatAnthropic
    
    anthropic_config = config['provider']['anthropic']
    
    llm = ChatAnthropic(
        model=anthropic_config['model_name'],
        temperature=anthropic_config.get('temperature', 0.3),
        max_tokens=anthropic_config.get('max_tokens', 128)
    )
    
    return llm