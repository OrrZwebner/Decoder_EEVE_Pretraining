#!/usr/bin/env python3
"""
Model Processors Module

This module handles model-specific tokenizer processing for LLaMA, Gemma, and GPT-2.
It includes token format conversion, vocabulary expansion, and model-specific
adaptations to ensure compatibility between learned Sanskrit tokens and
target model architectures.

Author: Sanskrit NLP Research
Version: 2.0
"""

import logging
import copy
from typing import List, Tuple, Any, Dict
from transformers import AutoTokenizer


class BaseProcessor:
    """
    Base class for model-specific token processors.
    
    Defines the common interface that all model processors must implement
    for consistent token processing and tokenizer expansion.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the processor with model configuration.
        
        Args:
            model_config (Dict[str, Any]): Model configuration from YAML
        """
        self.model_config = model_config
        self.model_name = model_config['model_name']
        self.original_tokenizer = None
        self.logger = logging.getLogger(__name__)
        
        # Load the original tokenizer for this model
        try:
            self.original_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.logger.info(f"Loaded {self.model_name} tokenizer")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer for {self.model_name}: {e}")
            raise
    
    def process_tokens_for_model(self, tokens: List[str], algorithm: str) -> List[str]:
        """
        Process tokens for model-specific format requirements.
        
        This method should be overridden by subclasses to implement
        model-specific token processing logic.
        
        Args:
            tokens (List[str]): Raw tokens from training
            algorithm (str): Algorithm used to generate tokens
        
        Returns:
            List[str]: Processed tokens compatible with target model
        """
        raise NotImplementedError("Subclasses must implement process_tokens_for_model")
    
    def expand_tokenizer(self, learned_tokens: List[str], algorithm_name: str, 
                                max_tokens: int) -> Tuple[Any, int, List[str]]:
            """
            Add learned tokens to the original tokenizer.
            
            Args:
                learned_tokens (List[str]): Tokens learned from Sanskrit training
                algorithm_name (str): Name of algorithm used to learn tokens
                max_tokens (int): Maximum number of tokens to add
            
            Returns:
                Tuple[Any, int, List[str]]: (expanded_tokenizer, num_tokens_added, processed_tokens)
            """
            expanded_tokenizer = copy.deepcopy(self.original_tokenizer)
            original_vocab = expanded_tokenizer.get_vocab()
            
            self.logger.info(f"\n=== EXPANDING {self.model_name.upper()} WITH {algorithm_name} ===")
            # REPLACE: # ... (logging statements) ... WITH:
            self.logger.info(f"Original vocabulary size: {len(original_vocab):,}")
            self.logger.info(f"Raw tokens received: {len(learned_tokens)}")
            
            processed_tokens = self.process_tokens_for_model(learned_tokens, algorithm_name)
            
            if len(processed_tokens) > max_tokens:
                processed_tokens = processed_tokens[:max_tokens]
                self.logger.info(f"Limited to {max_tokens} tokens")
            
            self.logger.info(f"Adding {len(processed_tokens)} processed tokens...")
            
            if processed_tokens:
                # Filter out tokens that already exist in vocabulary
                existing_vocab = set(original_vocab.keys())
                new_tokens = [token for token in processed_tokens if token not in existing_vocab]
                
                if new_tokens:
                    num_added = expanded_tokenizer.add_tokens(new_tokens)
                    # REPLACE: # ... (logging statements) ... WITH:
                    self.logger.info(f"✅ Successfully added {num_added} new tokens")
                    self.logger.info(f"New vocabulary size: {len(expanded_tokenizer.get_vocab()):,}")
                    self.logger.info(f"Sample new tokens: {new_tokens[:5]}{'...' if len(new_tokens) > 5 else ''}")
                    
                    # MODIFIED RETURN STATEMENT
                    return expanded_tokenizer, num_added, processed_tokens
                else:
                    self.logger.warning("⚠️ All processed tokens already exist in vocabulary")
                    return self.original_tokenizer, 0, processed_tokens
            else:
                self.logger.warning("❌ No tokens to add")
                # MODIFIED RETURN STATEMENT
                return self.original_tokenizer, 0, []

class LlamaProcessor(BaseProcessor):
    """
    Processor for LLaMA model family tokenizers.
    
    LLaMA uses a BPE-based tokenizer with specific formatting:
    - Uses Ġ tokens for spaces at word beginnings
    - Requires removal of prefixes from other tokenization schemes
    - Compatible with standard BPE token format
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize LLaMA processor.
        
        Args:
            model_config (Dict[str, Any]): LLaMA model configuration
        """
        super().__init__(model_config)
        self.logger.info("Initialized LLaMA processor")
    
    def process_tokens_for_model(self, tokens: List[str], algorithm: str) -> List[str]:
        """
        Process tokens for LLaMA compatibility.
        
        LLaMA processing rules:
        - Remove ## prefixes from WordPiece tokens
        - Remove ▁ prefixes from Unigram/SentencePiece tokens
        - Convert Ġ tokens to space + text format
        - Keep BPE tokens as-is
        
        Args:
            tokens (List[str]): Raw tokens from training
            algorithm (str): Algorithm used to generate tokens
        
        Returns:
            List[str]: LLaMA-compatible tokens
        """
        processed_tokens = []
    
        # Track for debugging
        duplicates_after_processing = set()
        empty_after_processing = []
        
        for token in tokens:
            processed_token = token
            
            if "wordpiece" in algorithm.lower():
                processed_token = token.replace("##", "")
                
            if processed_token.startswith("Ġ"):
                processed_token = " " + processed_token[1:]
            
            if len(processed_token.strip()) > 0:
                if processed_token in processed_tokens:
                    duplicates_after_processing.add(processed_token)
                processed_tokens.append(processed_token)
            else:
                empty_after_processing.append(token)
        
        # Log issues
        if duplicates_after_processing:
            self.logger.warning(f"Found {len(duplicates_after_processing)} duplicates after processing: {list(duplicates_after_processing)[:10]}")
        if empty_after_processing:
            self.logger.warning(f"Found {len(empty_after_processing)} empty tokens: {empty_after_processing}")
        
        return processed_tokens


class GemmaProcessor(BaseProcessor):
    """
    Processor for Gemma model family tokenizers.
    
    Gemma uses SentencePiece-based tokenization with specific formatting:
    - Uses ▁ tokens for word-initial markers
    - Requires both prefixed and non-prefixed versions for non-SP algorithms
    - Native compatibility with SentencePiece token format
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize Gemma processor.
        
        Args:
            model_config (Dict[str, Any]): Gemma model configuration
        """
        super().__init__(model_config)
        self.logger.info("Initialized Gemma processor")
    
    def process_tokens_for_model(self, tokens: List[str], algorithm: str) -> List[str]:
        """
        Process tokens for Gemma SentencePiece compatibility.
        
        Gemma processing rules:
        - Keep SentencePiece tokens as-is (they're already compatible)
        - For other algorithms: create both ▁prefixed (word-initial) and 
          base (continuation) versions
        - Maintain SentencePiece format conventions
        
        Args:
            tokens (List[str]): Raw tokens from training
            algorithm (str): Algorithm used to generate tokens
        
        Returns:
            List[str]: Gemma-compatible tokens
        """
        processed_tokens = []
        
        for token in tokens:
            if "sentencepiece" in algorithm.lower():
                # SentencePiece tokens are already in the correct format for Gemma
                processed_tokens.append(token)
                
            else:
                # For non-SentencePiece algorithms, create both versions
                # This ensures compatibility with Gemma's SentencePiece tokenizer
                
                # Extract base token by removing existing prefixes
                base_token = token.replace("##", "").replace("▁", "").strip()
                
                if len(base_token) > 0:
                    # Add word-initial version (with ▁ prefix)
                    prefixed_token = f"▁{base_token}"
                    processed_tokens.append(prefixed_token)
                    
                    # Add continuation version (without prefix)
                    # Only add if we have space and it's different from prefixed
                    if base_token != prefixed_token:
                        processed_tokens.append(base_token)
        
        self.logger.info(f"Gemma processing: {len(tokens)} → {len(processed_tokens)} tokens")
        return processed_tokens


class Gpt2Processor(BaseProcessor):
    """
    Processor for GPT-2 model family tokenizers.
    
    GPT-2 uses byte-level BPE tokenization similar to LLaMA:
    - Uses Ġ tokens for spaces at word beginnings
    - Byte-level encoding handles all Unicode characters
    - Compatible with standard BPE token format
    - Similar processing rules to LLaMA
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize GPT-2 processor.
        
        Args:
            model_config (Dict[str, Any]): GPT-2 model configuration
        """
        super().__init__(model_config)
        self.logger.info("Initialized GPT-2 processor")
    
    def process_tokens_for_model(self, tokens: List[str], algorithm: str) -> List[str]:
        """
        Process tokens for GPT-2 byte-level BPE compatibility.
        
        GPT-2 processing rules (similar to LLaMA):
        - Remove ## prefixes from WordPiece tokens
        - Remove ▁ prefixes from Unigram/SentencePiece tokens
        - Convert Ġ tokens to space + text format
        - Keep BPE tokens as-is
        - Handle byte-level encoding properly
        
        Args:
            tokens (List[str]): Raw tokens from training
            algorithm (str): Algorithm used to generate tokens
        
        Returns:
            List[str]: GPT-2-compatible tokens
        """
        processed_tokens = []
        
        # Track for debugging
        duplicates_after_processing = set()
        empty_after_processing = []
        
        for token in tokens:
            processed_token = token
            
            # Remove WordPiece prefixes
            if "wordpiece" in algorithm.lower():
                processed_token = token.replace("##", "")
            
            # Remove SentencePiece prefixes
            if processed_token.startswith("▁"):
                processed_token = processed_token[1:]
                
            # Handle Ġ tokens (convert to space + text)
            if processed_token.startswith("Ġ"):
                processed_token = " " + processed_token[1:]
            
            # Only add non-empty tokens
            if len(processed_token.strip()) > 0:
                if processed_token in processed_tokens:
                    duplicates_after_processing.add(processed_token)
                processed_tokens.append(processed_token)
            else:
                empty_after_processing.append(token)
        
        # Log issues for debugging
        if duplicates_after_processing:
            self.logger.warning(f"GPT-2: Found {len(duplicates_after_processing)} duplicates after processing: {list(duplicates_after_processing)[:10]}")
        if empty_after_processing:
            self.logger.warning(f"GPT-2: Found {len(empty_after_processing)} empty tokens: {empty_after_processing}")
        
        self.logger.info(f"GPT-2 processing: {len(tokens)} → {len(processed_tokens)} tokens")
        return processed_tokens


def get_processor(model_name: str, model_config: Dict[str, Any]) -> BaseProcessor:
    """
    Factory function to get the appropriate processor for a model.
    
    Determines the correct processor class based on model name and
    returns an initialized instance.
    
    Args:
        model_name (str): Name of the model (e.g., 'llama', 'gemma', 'gpt2')
        model_config (Dict[str, Any]): Model configuration dictionary
    
    Returns:
        BaseProcessor: Appropriate processor instance for the model
    
    Raises:
        ValueError: If model name is not supported
    """
    model_name_lower = model_name.lower()
    
    if "llama" in model_name_lower:
        return LlamaProcessor(model_config)
    elif "gemma" in model_name_lower:
        return GemmaProcessor(model_config)
    elif "gpt2" in model_name_lower or "gpt-2" in model_name_lower:
        return Gpt2Processor(model_config)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: llama, gemma, gpt2")


# Utility functions for token processing

def convert_wordpiece_to_bpe(tokens: List[str]) -> List[str]:
    """
    Convert WordPiece tokens to BPE format by removing ## prefixes.
    
    Args:
        tokens (List[str]): WordPiece tokens with ## prefixes
    
    Returns:
        List[str]: BPE-compatible tokens without prefixes
    """
    return [token.replace("##", "") for token in tokens if len(token.replace("##", "").strip()) > 0]


def convert_unigram_to_bpe(tokens: List[str]) -> List[str]:
    """
    Convert Unigram tokens to BPE format by removing ▁ prefixes.
    
    Args:
        tokens (List[str]): Unigram tokens with ▁ prefixes
    
    Returns:
        List[str]: BPE-compatible tokens without prefixes
    """
    return [token.replace("▁", "") for token in tokens if len(token.replace("▁", "").strip()) > 0]


def convert_to_sentencepiece_format(tokens: List[str]) -> List[str]:
    """
    Convert tokens to SentencePiece format with appropriate ▁ prefixes.
    
    Creates both word-initial (▁prefixed) and continuation (non-prefixed) versions
    for comprehensive SentencePiece compatibility.
    
    Args:
        tokens (List[str]): Tokens to convert
    
    Returns:
        List[str]: SentencePiece-compatible tokens
    """
    converted_tokens = []
    
    for token in tokens:
        # Clean the token of existing prefixes
        base_token = token.replace("##", "").replace("▁", "").strip()
        
        if len(base_token) > 0:
            # Add word-initial version
            converted_tokens.append(f"▁{base_token}")
            # Add continuation version
            converted_tokens.append(base_token)
    
    return converted_tokens


def handle_space_tokens(tokens: List[str], target_format: str = "llama") -> List[str]:
    """
    Handle space token representations across different tokenizer formats.
    
    Different tokenizers represent spaces differently:
    - LLaMA: Uses Ġ prefix for word-initial spaces
    - Gemma/SentencePiece: Uses ▁ prefix for word boundaries
    - GPT-2: Uses Ġ prefix for word-initial spaces (similar to LLaMA)
    
    Args:
        tokens (List[str]): Tokens potentially containing space representations
        target_format (str): Target format ('llama', 'gemma', or 'gpt2')
    
    Returns:
        List[str]: Tokens with appropriate space representations
    """
    processed_tokens = []
    
    for token in tokens:
        processed_token = token
        
        if target_format.lower() in ["llama", "gpt2"]:
            # Convert ▁ to Ġ for LLaMA/GPT-2 compatibility
            if token.startswith("▁"):
                processed_token = "Ġ" + token[1:]
            # Convert existing Ġ tokens to space + text
            elif token.startswith("Ġ"):
                processed_token = " " + token[1:]
                
        elif target_format.lower() == "gemma":
            # Convert Ġ to ▁ for Gemma compatibility
            if token.startswith("Ġ"):
                processed_token = "▁" + token[1:]
            # Ensure ▁ prefix for word-initial tokens
            elif not token.startswith("▁") and len(token.strip()) > 0:
                # Add ▁ prefix if it looks like a word-initial token
                if token[0].isalpha() or ord(token[0]) > 127:  # Unicode characters
                    processed_token = "▁" + token
        
        processed_tokens.append(processed_token)
    
    return processed_tokens