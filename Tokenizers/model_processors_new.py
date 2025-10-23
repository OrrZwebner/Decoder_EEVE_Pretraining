#!/usr/bin/env python3
"""
Model Processors Module

This module handles model-specific tokenizer processing for LLaMA, Gemma, and GPT-2.
It includes token format conversion, vocabulary expansion via train_new_from_iterator,
and model-specific adaptations to ensure compatibility between learned Sanskrit tokens
and target model architectures.

Author: Sanskrit NLP Research
Version: 3.0 - Updated to use train_new_from_iterator
"""

import logging
import copy
import json
from typing import List, Tuple, Any, Dict
from transformers import AutoTokenizer
from tqdm import tqdm


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
                        max_tokens: int, training_corpus: List[str] = None) -> Tuple[Any, int, List[str]]:
        """
        Expand tokenizer vocabulary using train_new_from_iterator approach.
        
        This method trains a new tokenizer on a corpus that includes the learned tokens,
        then merges the new vocabulary and merges into the original tokenizer.
        
        Args:
            learned_tokens (List[str]): Tokens learned from Sanskrit training
            algorithm_name (str): Name of algorithm used to learn tokens
            max_tokens (int): Maximum number of tokens to add
            training_corpus (List[str], optional): Corpus to train new tokenizer on.
                If None, creates a synthetic corpus from learned_tokens.
        
        Returns:
            Tuple[Any, int, List[str]]: (expanded_tokenizer, num_tokens_added, processed_tokens)
        """
        expanded_tokenizer = copy.deepcopy(self.original_tokenizer)
        original_vocab = expanded_tokenizer.get_vocab()
        
        self.logger.info(f"\n=== EXPANDING {self.model_name.upper()} WITH {algorithm_name} ===")
        self.logger.info(f"Original vocabulary size: {len(original_vocab):,}")
        self.logger.info(f"Raw tokens received: {len(learned_tokens)}")
        
        # Process tokens for model compatibility
        processed_tokens = self.process_tokens_for_model(learned_tokens, algorithm_name)
        
        if len(processed_tokens) > max_tokens:
            processed_tokens = processed_tokens[:max_tokens]
            self.logger.info(f"Limited to {max_tokens} tokens")
        
        self.logger.info(f"Processing {len(processed_tokens)} tokens for vocabulary expansion...")
        
        if not processed_tokens:
            self.logger.warning("❌ No tokens to add")
            return self.original_tokenizer, 0, []
        
        # Filter out tokens that already exist in vocabulary
        existing_vocab = set(original_vocab.keys())
        new_tokens = [token for token in processed_tokens if token not in existing_vocab]
        
        if not new_tokens:
            self.logger.warning("⚠️ All processed tokens already exist in vocabulary")
            return self.original_tokenizer, 0, processed_tokens
        
        self.logger.info(f"Found {len(new_tokens)} new unique tokens to add")
        
        # Create training corpus if not provided
        if training_corpus is None:
            self.logger.info("No training corpus provided, creating synthetic corpus from tokens...")
            training_corpus = self._create_synthetic_corpus(new_tokens)
        
        # Train new tokenizer on the corpus
        self.logger.info(f"Training new tokenizer on corpus ({len(training_corpus)} samples)...")
        
        try:
            # Create training corpus generator
            def get_training_corpus():
                for start_idx in range(0, len(training_corpus), 1000):
                    samples = training_corpus[start_idx : start_idx + 1000]
                    yield samples
            
            # Train new tokenizer with desired vocab size
            new_tokenizer = expanded_tokenizer.train_new_from_iterator(
                get_training_corpus(), 
                len(new_tokens)
            )
            
            # Get JSON representations of both tokenizers
            base_json = json.loads(expanded_tokenizer.backend_tokenizer.to_str())
            new_json = json.loads(new_tokenizer.backend_tokenizer.to_str())
            
            # Initialize with base vocab and merges
            merged_vocab = base_json['model']['vocab'].copy()
            merged_merges = base_json['model']['merges'].copy()
            
            # Count tokens before merging
            original_vocab_size = len(merged_vocab)
            
            # Merge new vocabulary
            self.logger.info("Merging new vocabulary...")
            tokens_added = 0
            for token, idx in tqdm(new_json['model']['vocab'].items(), desc="Merging vocab"):
                if token not in merged_vocab:
                    merged_vocab[token] = len(merged_vocab)
                    tokens_added += 1
            
            # Merge new merges
            self.logger.info("Merging new merges...")
            merges_added = 0
            for merge in tqdm(new_json['model']['merges'], desc="Merging merges"):
                if merge not in merged_merges:
                    merged_merges.append(merge)
                    merges_added += 1
            
            self.logger.info(f"Original merges: {len(base_json['model']['merges']):,}, "
                           f"New total: {len(merged_merges):,} (+{merges_added})")
            
            # Update the JSON with merged vocab and merges
            new_json['model']['vocab'] = merged_vocab
            new_json['model']['merges'] = merged_merges
            
            # Create the expanded tokenizer from merged JSON
            expanded_tokenizer._tokenizer = expanded_tokenizer.backend_tokenizer.from_str(
                json.dumps(new_json)
            )
            
            self.logger.info(f"✅ Successfully expanded vocabulary")
            self.logger.info(f"   Original size: {original_vocab_size:,}")
            self.logger.info(f"   New size: {len(merged_vocab):,}")
            self.logger.info(f"   Tokens added: {tokens_added:,}")
            self.logger.info(f"   Sample new tokens: {new_tokens[:5]}{'...' if len(new_tokens) > 5 else ''}")
            
            return expanded_tokenizer, tokens_added, processed_tokens
            
        except Exception as e:
            self.logger.error(f"Failed to expand tokenizer: {e}")
            self.logger.warning("Falling back to original tokenizer")
            return self.original_tokenizer, 0, processed_tokens
    
    def _create_synthetic_corpus(self, tokens: List[str], samples_per_token: int = 3) -> List[str]:
        """
        Create a synthetic corpus from tokens for training.
        
        Args:
            tokens (List[str]): Tokens to create corpus from
            samples_per_token (int): Number of samples to create per token
        
        Returns:
            List[str]: Synthetic corpus
        """
        corpus = []
        
        for token in tokens:
            # Create variations by combining tokens
            for _ in range(samples_per_token):
                # Simple approach: repeat token with spaces
                sample = f"{token} {token} {token}"
                corpus.append(sample)
        
        return corpus


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
        - Remove ## prefixes from WordPiece tokens
        - Keep ▁ prefixes for SentencePiece tokens
        - For non-SentencePiece algorithms, create both ▁ and non-▁ versions
        - Handle Ġ tokens by converting to ▁ format
        
        Args:
            tokens (List[str]): Raw tokens from training
            algorithm (str): Algorithm used to generate tokens
        
        Returns:
            List[str]: Gemma-compatible tokens
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
            
            # Convert Ġ to ▁ for SentencePiece compatibility
            if processed_token.startswith("Ġ"):
                processed_token = "▁" + processed_token[1:]
            
            # Only add non-empty tokens
            if len(processed_token.strip()) > 0:
                if processed_token in processed_tokens:
                    duplicates_after_processing.add(processed_token)
                    
                # For Gemma, we might want both versions: with and without ▁
                if not processed_token.startswith("▁") and "sentencepiece" not in algorithm.lower():
                    # Add version with ▁ prefix for word-initial usage
                    prefixed_token = "▁" + processed_token
                    if prefixed_token not in processed_tokens:
                        processed_tokens.append(prefixed_token)
                
                processed_tokens.append(processed_token)
            else:
                empty_after_processing.append(token)
        
        # Log issues for debugging
        if duplicates_after_processing:
            self.logger.warning(f"Gemma: Found {len(duplicates_after_processing)} duplicates after processing: {list(duplicates_after_processing)[:10]}")
        if empty_after_processing:
            self.logger.warning(f"Gemma: Found {len(empty_after_processing)} empty tokens: {empty_after_processing}")
        
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
