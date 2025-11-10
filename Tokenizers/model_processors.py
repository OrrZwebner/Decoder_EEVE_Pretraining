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
import json
import tempfile
import os
from typing import List, Tuple, Any, Dict, Optional
from transformers import AutoTokenizer
from tqdm import tqdm
import sentencepiece as smp


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
    
    def expand_tokenizer(self, algorithm_name: str, max_tokens: int,
                                training_corpus: List[str],
                                enforce_max_tokens: bool = True,
                                vocab_buffer_multiplier: float = 2.0) -> Tuple[Any, int, List[str]]:
            """
            Add learned tokens to the original tokenizer using train_new_from_iterator approach.

            For BPE and SentencePiece BPE algorithms, uses train_new_from_iterator approach with training_corpus.
            When enforce_max_tokens=True, trains with a larger vocabulary buffer to ensure sufficient tokens are available
            for selection, then picks the most important max_tokens from the results.

            Args:
                algorithm_name (str): Name of algorithm used to learn tokens (must be "BPE" or "SENTENCEPIECE_BPE")
                max_tokens (int): Maximum number of tokens to add
                training_corpus (List[str]): Training corpus for tokenizer training (required)
                enforce_max_tokens (bool): If True, strictly enforce max_tokens limit by selecting most important tokens.
                                          If False, add all trained tokens (may exceed max_tokens). Default: True.
                vocab_buffer_multiplier (float): When enforce_max_tokens=True, train with (max_tokens * this_value)
                                                vocabulary to ensure sufficient tokens for selection. Default: 2.0.

            Returns:
                Tuple[Any, int, List[str]]: (expanded_tokenizer, num_tokens_added, processed_tokens)

            Raises:
                ValueError: If algorithm is not BPE or SENTENCEPIECE_BPE, or training_corpus is empty
                Exception: If train_new_from_iterator fails
            """

            expanded_tokenizer = copy.deepcopy(self.original_tokenizer)
            original_vocab = expanded_tokenizer.get_vocab()

            self.logger.info(f"\n=== EXPANDING {self.model_name.upper()} WITH {algorithm_name} ===")
            self.logger.info(f"Original vocabulary size: {len(original_vocab):,}")
            self.logger.info(f"Training corpus size: {len(training_corpus) if training_corpus else 0}")

            # Support BPE and SentencePiece BPE with train_new_from_iterator
            algo_upper = algorithm_name.upper()
            if algo_upper not in ["BPE", "SENTENCEPIECE_BPE"]:
                raise ValueError(f"Only BPE and SENTENCEPIECE_BPE algorithms are supported. Got: {algorithm_name}")

            if training_corpus is None or len(training_corpus) == 0:
                raise ValueError("Training corpus is required for tokenizer expansion")

            self.logger.info(f"Using train_new_from_iterator approach for {algorithm_name} with {len(training_corpus)} corpus samples")

            # Create training corpus generator
            def get_training_corpus():
                for start_idx in range(0, len(training_corpus), 1000):
                    samples = training_corpus[start_idx : start_idx + 1000]
                    yield samples

            # Calculate target vocabulary size
            # When enforcing limits, use a buffer to ensure we get enough tokens for selection
            if enforce_max_tokens:
                # Train with buffer (e.g., 2x) to have enough tokens to select from
                buffered_tokens = int(max_tokens * vocab_buffer_multiplier)
                target_vocab_size = len(original_vocab) + buffered_tokens
                self.logger.info(f"Training with {vocab_buffer_multiplier}x buffer: target vocab size = {target_vocab_size:,} "
                               f"({buffered_tokens} new tokens to select {max_tokens} best from)")
            else:
                # Not enforcing, so just train for max_tokens
                target_vocab_size = len(original_vocab) + max_tokens
                self.logger.info(f"Training without buffer: target vocab size = {target_vocab_size:,}")

            self.logger.info(f"Training new tokenizer from base using corpus...")
            new_tokenizer = expanded_tokenizer.train_new_from_iterator(
                get_training_corpus(),
                target_vocab_size
            )

            # Get JSON representations of both tokenizers
            base_json = json.loads(expanded_tokenizer.backend_tokenizer.to_str())
            new_json = json.loads(new_tokenizer.backend_tokenizer.to_str())

            # Check tokenizer model types (base and after training)
            base_model_type = base_json['model']['type']
            new_model_type = new_json['model']['type']
            self.logger.info(f"Base tokenizer model type: {base_model_type}")
            self.logger.info(f"New tokenizer model type: {new_model_type}")
            
            # Use the new model type for merging (train_new_from_iterator may convert types)
            model_type = new_model_type

            if model_type == 'BPE':
                # For BPE tokenizers: merge vocab and merges
                merged_vocab = base_json['model']['vocab'].copy()
                merged_merges = base_json['model']['merges'].copy()

                original_vocab_size = len(merged_vocab)
                original_merges_size = len(merged_merges)

                # Identify truly new tokens and merges
                new_tokens_dict = {token: idx for token, idx in new_json['model']['vocab'].items()
                                   if token not in merged_vocab}
                new_merges_list = [merge for merge in new_json['model']['merges']
                                   if merge not in merged_merges]

                self.logger.info(f"Found {len(new_tokens_dict)} new tokens and {len(new_merges_list)} new merges")

                if enforce_max_tokens:
                    self.logger.info(f"✓ Enforcing max_tokens={max_tokens} limit (keeping most important by merge order)")
                else:
                    self.logger.info(f"✗ NOT enforcing max_tokens limit - will add all {len(new_tokens_dict)} new tokens")

                # Add new merges and their corresponding tokens
                tokens_added = 0
                merges_added = 0
                skipped_invalid = 0

                for merge in tqdm(new_merges_list, desc="Adding merges & tokens"):
                    # Check limit only if enforcing
                    if enforce_max_tokens and tokens_added >= max_tokens:
                        self.logger.info(f"Reached max_tokens limit ({max_tokens}), stopping merge addition")
                        break

                    # Calculate the resulting token from this merge
                    if isinstance(merge, list):
                        # Merge is a list like ["token1", "token2"]
                        if len(merge) == 2:
                            parts = merge
                            resulting_token = parts[0] + parts[1]
                        else:
                            self.logger.warning(f"Unexpected merge list format: {merge} (expected 2 elements)")
                            skipped_invalid += 1
                            continue
                    elif isinstance(merge, str):
                        # Merge is a string like "token1 token2"
                        parts = merge.split(' ', 1)
                        if len(parts) == 2:
                            resulting_token = parts[0] + parts[1]
                        else:
                            self.logger.warning(f"Unexpected merge string format: '{merge}' (expected 'token1 token2')")
                            skipped_invalid += 1
                            continue
                    else:
                        self.logger.warning(f"Unexpected merge type: {type(merge)}")
                        skipped_invalid += 1
                        continue

                    # Check that component tokens exist in the current vocabulary
                    component1, component2 = parts[0], parts[1]
                    if component1 not in merged_vocab:
                        self.logger.warning(f"Skipping corrupt merge: component '{component1}' not in vocabulary")
                        skipped_invalid += 1
                        continue
                    if component2 not in merged_vocab:
                        self.logger.warning(f"Skipping corrupt merge: component '{component2}' not in vocabulary")
                        skipped_invalid += 1
                        continue

                    # Add merge and token if the token is actually new
                    if resulting_token in new_tokens_dict:
                        # Add the merge rule
                        merged_merges.append(merge)
                        merges_added += 1

                        # Add the corresponding token
                        if resulting_token not in merged_vocab:
                            merged_vocab[resulting_token] = len(merged_vocab)
                            tokens_added += 1
                    else:
                        # Token already exists, but merge rule might still be useful
                        merged_merges.append(merge)
                        merges_added += 1

                if skipped_invalid > 0:
                    self.logger.warning(f"⚠️  Skipped {skipped_invalid} invalid/corrupt merges during expansion")

                self.logger.info(f"Original vocab: {original_vocab_size:,}, New vocab: {len(merged_vocab):,} (+{tokens_added})")
                self.logger.info(f"Original merges: {original_merges_size:,}, New merges: {len(merged_merges):,} (+{merges_added})")

                if enforce_max_tokens:
                    self.logger.info(f"✅ Enforced max_tokens={max_tokens}: added {tokens_added} tokens (with {merges_added} merges)")
                else:
                    self.logger.info(f"✅ Added all available tokens: {tokens_added} tokens (with {merges_added} merges)")

                # Update JSON with merged vocab and merges
                new_json['model']['vocab'] = merged_vocab
                new_json['model']['merges'] = merged_merges

                # Reconstruct tokenizer from merged JSON
                expanded_tokenizer._tokenizer = expanded_tokenizer.backend_tokenizer.from_str(
                    json.dumps(new_json)
                )

            elif model_type == 'Unigram':
                # For SentencePiece/Unigram tokenizers: merge vocab only (no merges)
                base_vocab = base_json['model']['vocab']
                new_vocab = new_json['model']['vocab']

                # Calculate default score for new tokens (low probability for unknown tokens)
                base_scores = [score for score in base_vocab.values() if isinstance(score, (int, float))]
                if base_scores:
                    min_base_score = min(base_scores)
                    # Use a score slightly lower than the minimum to indicate new, rare tokens
                    default_score = min_base_score - 1.0
                    self.logger.info(f"Using default score {default_score:.2f} for new tokens (min base score: {min_base_score:.2f})")
                else:
                    # Fallback if no scores found (unlikely but safe)
                    default_score = -10.0
                    self.logger.warning(f"No scores found in base vocab, using fallback score: {default_score}")

                # Identify truly new tokens
                new_tokens_dict = {token: score for token, score in new_vocab.items()
                                   if token not in base_vocab}

                self.logger.info(f"Found {len(new_tokens_dict)} new tokens")

                if enforce_max_tokens:
                    self.logger.info(f"✓ Enforcing max_tokens={max_tokens} limit (keeping highest scoring tokens)")
                    # Sort new tokens by score (descending - higher scores are better in Unigram)
                    sorted_new_tokens = sorted(new_tokens_dict.items(), key=lambda x: x[1], reverse=True)
                    # Limit to max_tokens
                    tokens_to_add = sorted_new_tokens[:max_tokens]
                else:
                    self.logger.info(f"✗ NOT enforcing max_tokens limit - will add all {len(new_tokens_dict)} new tokens")
                    tokens_to_add = new_tokens_dict.items()

                # Merge vocabularies
                self.logger.info("Merging Unigram vocabularies...")
                merged_vocab = base_vocab.copy()
                tokens_added = 0
                for token, score in tqdm(tokens_to_add, desc="Adding tokens"):
                    if token not in merged_vocab:
                        # Preserve the score from new tokenizer if valid, otherwise use default low score
                        if isinstance(score, (int, float)):
                            merged_vocab[token] = score
                        else:
                            merged_vocab[token] = default_score
                            self.logger.warning(f"Token '{token}' has invalid score type {type(score)}, using default")
                        tokens_added += 1

                original_vocab_size = len(base_vocab)
                self.logger.info(f"Original vocab: {original_vocab_size:,}, New vocab: {len(merged_vocab):,} (+{tokens_added})")

                if enforce_max_tokens:
                    self.logger.info(f"✅ Enforced max_tokens={max_tokens}: added {tokens_added} highest-scoring tokens")
                else:
                    self.logger.info(f"✅ Added all available tokens: {tokens_added} tokens")

                # Update JSON with merged vocab
                new_json['model']['vocab'] = merged_vocab

                # Reconstruct tokenizer from merged JSON
                expanded_tokenizer._tokenizer = expanded_tokenizer.backend_tokenizer.from_str(
                    json.dumps(new_json)
                )

            else:
                raise ValueError(f"Unsupported tokenizer model type: {model_type}. Only BPE and Unigram are supported.")

            # Get list of newly added tokens for return value
            # Calculate actual tokens added based on final vocabulary difference
            final_vocab = expanded_tokenizer.get_vocab()
            new_vocab_set = set(final_vocab.keys())
            original_vocab_set = set(original_vocab.keys())
            newly_added_tokens_list = list(new_vocab_set - original_vocab_set)
            actual_tokens_added = len(newly_added_tokens_list)

            self.logger.info(f"✅ Successfully expanded vocabulary using train_new_from_iterator")
            self.logger.info(f"Original vocabulary size: {len(original_vocab):,}")
            self.logger.info(f"New vocabulary size: {len(final_vocab):,}")
            self.logger.info(f"Actual tokens added: {actual_tokens_added:,}")
            if newly_added_tokens_list:
                self.logger.info(f"Sample new tokens: {newly_added_tokens_list[:10]}{'...' if len(newly_added_tokens_list) > 10 else ''}")

            return expanded_tokenizer, actual_tokens_added, newly_added_tokens_list

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