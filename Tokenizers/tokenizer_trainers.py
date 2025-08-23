#!/usr/bin/env python3
"""
Sanskrit Tokenizer Training Module

This module contains all tokenization training algorithms including BPE, WordPiece,
Unigram, and SentencePiece variants. It handles the training of specialized tokenizers
on Sanskrit text and implements smart duplicate filtering to ensure exactly N unique
tokens are returned for each algorithm.

"""

import tempfile
import os
import logging
from typing import List, Tuple, Any, Dict
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, Metaspace
from tokenizers.decoders import WordPiece as WordPieceDecoder, Metaspace as MetaspaceDecoder
import sentencepiece as smp
import numpy as np
import pandas as pd
from log_score_evaluator import LogScoreEvaluator


def train_bpe_tokenizer(texts: List[str], vocab_size: int) -> Tuple[List[str], bool]:
    """
    Train BPE (Byte Pair Encoding) tokenizer on Sanskrit texts.
    
    BPE learns merge rules by iteratively combining the most frequent
    character pairs. It's deterministic and widely used.
    
    Args:
        texts (List[str]): Sanskrit texts for training
        vocab_size (int): Target vocabulary size for training
    
    Returns:
        Tuple[List[str], bool]: (learned_tokens, success_flag)
    """
    try:
        logging.info(f"Training BPE tokenizer with vocab_size={vocab_size}")
        
        # Initialize BPE tokenizer with whitespace pre-tokenization
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Configure BPE trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,  # Minimum frequency for merges
            special_tokens=["<unk>"]
        )
        
        # Train on Sanskrit texts
        tokenizer.train_from_iterator(texts, trainer=trainer)
        vocab = tokenizer.get_vocab()
        
        # Extract learned tokens (exclude special tokens)
        learned_tokens = []
        for token in vocab.keys():
            if token not in ["<unk>"] and len(token.strip()) > 0:
                learned_tokens.append(token)
        
        logging.info(f"BPE training completed: {len(learned_tokens)} tokens learned")
        return learned_tokens, True
        
    except Exception as e:
        logging.error(f"BPE training failed: {e}")
        return [], False

def train_log_score_tokenizer(texts: List[str], vocab_size: int, original_tokenizer: Tokenizer) -> Tuple[List[str], bool]:
    """
    Train log-score tokenizer on Sanskrit texts.
    
    log-score tokenization starat from a very large vocabulary and then sort according to 
    the "log-score" of the tokens. Slice the best (highest log-score) tokens to get the desired vocabulary size.
    
    Args:
        texts (List[str]): Sanskrit texts for training
        vocab_size (int): Target vocabulary size for training
        original_tokenizer (Tokenizer): Original tokenizer to use for log-score computation
    
    Returns:
        Tuple[List[str], bool]: (learned_tokens, success_flag)
    """
    try:
        logging.info(f"Training log-score tokenizer with vocab_size={vocab_size}")
        
        # Initialize BPE tokenizer with whitespace pre-tokenization
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Configure BPE trainer with a large initial vocabulary size
        trainer = BpeTrainer(
            vocab_size= np.min(vocab_size**2, 100000),  # Start with larger size, maximum 100k
            min_frequency=2,  # Minimum frequency for merges
            special_tokens=["<unk>"]
        )
        
        # Train on Sanskrit texts
        tokenizer.train_from_iterator(texts, trainer=trainer)
        init_vocab = tokenizer.get_vocab()
        
        # Extract learned tokens (exclude special tokens)
        learned_tokens = []
        for token in init_vocab.keys():
            if token not in ["<unk>"] and len(token.strip()) > 0:
                learned_tokens.append(token)
        
        logging.info(f"Initial vocabulary training by BPE completed: {len(learned_tokens)} tokens learned")
        logging.info(f"Computing log-scores for {len(learned_tokens)} tokens")

        # create a DataFrame of the tokens to compute log-scores
        df_tokens = pd.DataFrame(learned_tokens, columns=['token'])
        # Compute log-scores based on token frequencies



        return learned_tokens, True
        
    except Exception as e:
        logging.error(f"BPE training failed: {e}")
        return [], False

def train_wordpiece_tokenizer(texts: List[str], vocab_size: int) -> Tuple[List[str], bool]:
    """
    FIXED for LLaMA: Better parameters to generate more diverse tokens.
    """
    try:
        logging.info(f"Training WordPiece tokenizer with vocab_size={vocab_size}")
        
        # Validate inputs
        if not texts or vocab_size <= 0:
            logging.error("Invalid inputs for WordPiece training")
            return [], False
        
        # Initialize WordPiece tokenizer
        tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = WordPieceDecoder(prefix="##")
        
        # FIXED: Optimized parameters for better token generation
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=1,  # FIXED: Lower threshold to get more tokens
            special_tokens=["<unk>"],
            continuing_subword_prefix="##",
            end_of_word_suffix=""
        )
        
        # Train on Sanskrit texts
        tokenizer.train_from_iterator(texts, trainer=trainer)
        vocab = tokenizer.get_vocab()
        
        # FIXED: Better token extraction with validation
        learned_tokens = []
        
        for token in vocab.keys():
            # Skip special tokens
            if token in ["<unk>", "<pad>", "<s>", "</s>"]:
                continue
                
            # FIXED: Keep all non-special tokens, even if they become empty after ## removal
            # The filtering happens in get_unique_tokens(), not here
            if len(token.strip()) > 0:
                learned_tokens.append(token)
        
        logging.info(f"WordPiece training completed: {len(learned_tokens)} tokens learned from {len(vocab)} total vocab")
        
        # FIXED: Log token distribution for debugging
        regular_tokens = [t for t in learned_tokens if not t.startswith("##")]
        continuation_tokens = [t for t in learned_tokens if t.startswith("##")]
        logging.debug(f"Token breakdown: {len(regular_tokens)} regular, {len(continuation_tokens)} continuation")
        
        return learned_tokens, True
        
    except Exception as e:
        logging.error(f"WordPiece training failed: {e}")
        import traceback
        logging.debug(f"Full error: {traceback.format_exc()}")
        return [], False


def train_unigram_tokenizer(texts: List[str], vocab_size: int) -> Tuple[List[str], bool]:
    """
    Train Unigram tokenizer on Sanskrit texts.
    
    Unigram tokenization is probabilistic and non-deterministic.
    The actual vocabulary size can vary from the requested size.
    
    Args:
        texts (List[str]): Sanskrit texts for training
        vocab_size (int): Target vocabulary size for training
    
    Returns:
        Tuple[List[str], bool]: (learned_tokens, success_flag)
    """
    try:
        logging.info(f"Training Unigram tokenizer with vocab_size={vocab_size}")
        
        # Initialize Unigram tokenizer with Metaspace pre-tokenization
        tokenizer = Tokenizer(Unigram())
        tokenizer.pre_tokenizer = Metaspace()
        tokenizer.decoder = MetaspaceDecoder()
        
        # Configure Unigram trainer
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>"],
            unk_token="<unk>"
        )
        
        # Train on Sanskrit texts
        tokenizer.train_from_iterator(texts, trainer=trainer)
        vocab = tokenizer.get_vocab()
        
        # Extract learned tokens (exclude special tokens and clean ▁ prefix)
        learned_tokens = []
        for token in vocab.keys():
            cleaned_token = token.replace("▁", "").strip()
            if token not in ["<unk>"] and len(cleaned_token) > 0:
                learned_tokens.append(token)
        
        logging.info(f"Unigram training completed: {len(learned_tokens)} tokens learned")
        return learned_tokens, True
        
    except Exception as e:
        logging.error(f"Unigram training failed: {e}")
        return [], False


def train_sentencepiece_tokenizer(texts: List[str], vocab_size: int, 
                                 model_type: str = "bpe") -> Tuple[List[str], bool]:
    """
    Train SentencePiece tokenizer with specified model type.
    
    SentencePiece is Google's tokenization library that supports
    multiple algorithms (BPE, Unigram, Word, Char) in a unified framework.
    
    Args:
        texts (List[str]): Sanskrit texts for training
        vocab_size (int): Target vocabulary size for training
        model_type (str): SentencePiece model type ('bpe', 'unigram', 'word', 'char')
    
    Returns:
        Tuple[List[str], bool]: (learned_tokens, success_flag)
    """
    try:
        logging.info(f"Training SentencePiece ({model_type}) with vocab_size={vocab_size}")
        
        # Create temporary file for training data
        # This eliminates the SentenceIterator warning
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
            temp_input_file = f.name
        
        # Create temporary model prefix for output files
        temp_model_prefix = tempfile.mktemp()
        
        # Configure training parameters based on model type
        train_params = {
            'input': temp_input_file,
            'model_prefix': temp_model_prefix,
            'vocab_size': vocab_size,
            'character_coverage': 0.98,  # Cover 98% of characters
            'model_type': model_type,
            'pad_id': 3, 
            'unk_id': 0,
            'bos_id': 1,
            'eos_id': 2,
            'user_defined_symbols': ["[MASK]"]  # Avoid conflict with <unk>
        }
        
        # Model-specific parameter adjustments
        if model_type == "unigram":
            train_params['character_coverage'] = 0.995  # Higher coverage for unigram
        elif model_type == "word":
            train_params['split_by_whitespace'] = True
            train_params['split_by_unicode_script'] = False
        
        # Train SentencePiece model
        smp.SentencePieceTrainer.train(**train_params)
        
        # Load the trained model
        sp = smp.SentencePieceProcessor()
        sp.load(f"{temp_model_prefix}.model")
        
        # Extract vocabulary (exclude special tokens)
        learned_tokens = []
        for i in range(sp.get_piece_size()):
            token = sp.id_to_piece(i)
            if token not in ['<unk>', '<s>', '</s>', '<pad>', '[MASK]'] and len(token.strip()) > 0:
                learned_tokens.append(token)
        
        logging.info(f"SentencePiece ({model_type}) training completed: {len(learned_tokens)} tokens learned")
        
        # Cleanup temporary files
        try:
            os.unlink(temp_input_file)
            os.unlink(f"{temp_model_prefix}.model")
            os.unlink(f"{temp_model_prefix}.vocab")
        except:
            pass  # Ignore cleanup errors
        
        return learned_tokens, True
        
    except Exception as e:
        logging.error(f"SentencePiece ({model_type}) training failed: {e}")
        return [], False


def get_unique_tokens(learned_tokens: List[str], target_tokenizer: Any, 
                     algorithm: str, model_name: str) -> List[str]:
    """
    FIXED: Enhanced for GPT-2 support and better handling of token processing.
    """
    original_vocab = target_tokenizer.get_vocab()
    unique_tokens = []
    duplicates = 0
    
    logging.debug(f"Filtering {len(learned_tokens)} tokens for {model_name} using {algorithm}")
    
    for token in learned_tokens:
        processed_token = token
        
        # Apply model-specific token processing
        if model_name.upper() in ["LLAMA", "GPT2"]:
            # LLaMA and GPT-2 processing: both use byte-level BPE with similar format
            if "wordpiece" in algorithm.lower():
                # Remove WordPiece ## prefix
                processed_token = token.replace("##", "")
                
                # Skip empty tokens that result from ## removal
                if len(processed_token.strip()) == 0:
                    logging.debug(f"Skipping empty token after ## removal: '{token}'")
                    duplicates += 1
                    continue
                    
            elif "unigram" in algorithm.lower() or "sentencepiece" in algorithm.lower():
                # Remove SentencePiece ▁ prefix
                processed_token = token.replace("▁", "")
                
            # Handle special space tokens (Ġ for both LLaMA and GPT-2)
            if processed_token.startswith("Ġ"):
                processed_token = " " + processed_token[1:]
                logging.debug(f"Processing {model_name} Ġ token: '{token}' -> '{processed_token}'")
                
            # Thorough uniqueness check
            if len(processed_token.strip()) > 0:
                if processed_token not in original_vocab:
                    unique_tokens.append(token)  # Keep original format
                    logging.debug(f"Unique for {model_name}: '{token}' -> '{processed_token}'")
                else:
                    duplicates += 1
                    logging.debug(f"Duplicate for {model_name}: '{token}' -> '{processed_token}' (in vocab)")
            else:
                duplicates += 1
                logging.debug(f"Empty after processing: '{token}' -> '{processed_token}'")
                
        elif model_name.upper() == "GEMMA":
            # Gemma is SentencePiece-based, so ▁ prefixes are native
            if "sentencepiece" in algorithm.lower():
                processed_token = token  # Keep as-is for SentencePiece
            else:
                # For other algorithms, try both prefixed and non-prefixed
                logging.warning(f"Non-SentencePiece algorithm {algorithm} may not work well with Gemma")
                base_token = token.replace("##", "").replace("▁", "")
                if len(base_token.strip()) > 0:
                    # Check if either version exists
                    prefixed_token = f"▁{base_token}"
                    if (base_token not in original_vocab and 
                        prefixed_token not in original_vocab):
                        unique_tokens.append(token)  # Keep original format
                    else:
                        duplicates += 1
                else:
                    duplicates += 1
                continue
                
            # Standard Gemma processing
            if (len(processed_token.strip()) > 0 and 
                processed_token not in original_vocab):
                unique_tokens.append(token)
            else:
                duplicates += 1
        else:
            # Unknown model - use generic processing
            logging.warning(f"Unknown model {model_name}, using generic token processing")
            if len(processed_token.strip()) > 0 and processed_token not in original_vocab:
                unique_tokens.append(token)
            else:
                duplicates += 1

    logging.info(f"Token filtering: {len(learned_tokens)} input → {len(unique_tokens)} unique, {duplicates} duplicates")
    return unique_tokens


def train_with_target_size(texts: List[str], algorithm: str, target_size: int, 
                         model_name: str, model_config: Dict[str, Any], 
                         unigram_max_iterations: int, tokenizer: Tokenizer = None) -> Tuple[List[str], bool]:
    """
    Train tokenizer to get exactly target_size unique tokens after processing.
    
    UPDATED: Now accepts model_config instead of hardcoded model names.
    
    Args:
        texts (List[str]): Sanskrit texts for training
        algorithm (str): Algorithm name ('bpe', 'wordpiece', 'sentencepiece_*')
        target_size (int): Exact number of unique tokens needed after processing
        model_name (str): Target model name ('LLAMA', 'GEMMA', 'GPT2') for compatibility
        model_config (Dict[str, Any]): Model configuration from config file
        unigram_max_iterations (int): Maximum iterations for unigram algorithms
        tokenizer (Tokenizer, optional): Pre-trained tokenizer to use for log-score or other algorithms
    
    Returns:
        Tuple[List[str], bool]: (exactly target_size tokens, success_flag)
    """

    logging.info(f"Training {algorithm} to get exactly {target_size} unique tokens for {model_name}")
    
    # Load target model tokenizer dynamically from config
    try:
        target_tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        logging.info(f"Loaded tokenizer: {model_config['model_name']}")
    except Exception as e:
        logging.error(f"Failed to load tokenizer {model_config['model_name']}: {e}")
        return [], False
    
    # For Unigram algorithms, use iterative training approach
    if "unigram" in algorithm.lower():
        return _train_unigram_iterative(texts, algorithm, target_size, target_tokenizer, model_name, max_iterations=unigram_max_iterations)
    
    # Estimate processing loss rate for WordPiece
    processing_loss_rate = 0.15 if algorithm.lower() == "wordpiece" else 0.0  # 15% loss due to duplicates after ## removal
    
    # Calculate adjusted target accounting for processing losses
    adjusted_target = int(target_size / (1 - processing_loss_rate))
    
    buffer_multipliers = [1.5, 2, 3, 4, 5, 8, 12]  # Extended range
    
    for multiplier in buffer_multipliers:
        training_vocab_size = int(adjusted_target * multiplier)
        logging.info(f"Trying {multiplier}x buffer: training_vocab_size={training_vocab_size}")
        
        try:
            # Train with current buffer size
            if algorithm.lower() == "bpe":
                learned_tokens, success = train_bpe_tokenizer(texts, training_vocab_size)
            elif algorithm.lower() == "wordpiece":
                learned_tokens, success = train_wordpiece_tokenizer(texts, training_vocab_size)
            elif algorithm.lower().startswith("sentencepiece"):
                model_type = algorithm.split("_")[1] if "_" in algorithm else "bpe"
                learned_tokens, success = train_sentencepiece_tokenizer(texts, training_vocab_size, model_type)
            elif algorithm.lower() == "log_score":
                learned_tokens, success = train_log_score_tokenizer(texts, training_vocab_size, tokenizer)
            else:
                logging.error(f"Unknown algorithm: {algorithm}")
                return [], False
            
            if not success:
                logging.warning(f"Training failed with {multiplier}x buffer")
                continue
            
            # Filter to get unique tokens
            unique_tokens = get_unique_tokens(learned_tokens, target_tokenizer, algorithm, model_name)
            
            # For WordPiece, simulate processing to count final usable tokens
            if algorithm.lower() == "wordpiece":
                processed_count = 0
                seen = set()
                for token in unique_tokens:
                    processed = token.replace("##", "")
                    if len(processed.strip()) > 0 and processed not in seen:
                        seen.add(processed)
                        processed_count += 1
                final_usable = processed_count
            else:
                final_usable = len(unique_tokens)
            
            logging.info(f"Buffer {multiplier}x: {len(learned_tokens)} learned → {len(unique_tokens)} unique → {final_usable} final usable")
            
            # Check if we have enough final usable tokens
            if final_usable >= target_size:
                # Take exactly target_size tokens, accounting for processing
                if algorithm.lower() == "wordpiece":
                    # Select tokens that will give us exactly target_size after processing
                    final_tokens = []
                    seen = set()
                    for token in unique_tokens:
                        if len(final_tokens) >= target_size:
                            break
                        processed = token.replace("##", "")
                        if len(processed.strip()) > 0 and processed not in seen:
                            seen.add(processed)
                            final_tokens.append(token)
                else:
                    final_tokens = unique_tokens[:target_size]
                
                logging.info(f"✅ SUCCESS: Got {len(final_tokens)} tokens that will yield exactly {target_size} after processing")
                return final_tokens, True
            else:
                logging.info(f"Insufficient: {final_usable}/{target_size} final usable tokens, trying larger buffer...")
                
        except Exception as e:
            logging.error(f"Error with {multiplier}x buffer: {e}")
            continue
    
    logging.error(f"COMPLETE FAILURE: Could not generate {target_size} usable tokens for {algorithm} + {model_name}")
    return [], False


def _train_unigram_iterative(texts: List[str], algorithm: str, target_size: int,
                            target_tokenizer: Any, model_name: str,
                            vocab_size_init_const=2.0 , max_iterations=10) -> Tuple[List[str], bool]:
    """
    Iterative training for Unigram algorithms to find closest number ≥ target_size.
    
    Args:
        texts (List[str]): Sanskrit texts for training
        algorithm (str): Algorithm name (unigram or sentencepiece_unigram)
        target_size (int): Target number of unique tokens
        target_tokenizer (Any): Target model tokenizer
        model_name (str): Model name
        vocab_size_init_const (float): Initial vocabulary size constant multiplier
        max_iterations (int): Maximum iterations to try
    
    Returns:
        Tuple[List[str], bool]: (unique_tokens, success_flag)
    """
    logging.info(f"Using iterative approach for {algorithm} to find ≥{target_size} unique tokens")
    
    # Start with larger vocabulary size to account for non-determinism and duplicates
    current_vocab_size = int(target_size * vocab_size_init_const)  # Start higher
    best_tokens = []
    best_size = 0
    
    # Track results to find the closest ≥ target_size
    all_results = []
    
    for iteration in range(max_iterations):
        logging.info(f"Iteration {iteration + 1}: trying vocab_size={current_vocab_size}")
        
        try:
            # Train with current vocabulary size
            if algorithm.lower() == "unigram":
                learned_tokens, success = train_unigram_tokenizer(texts, current_vocab_size)
            elif algorithm.lower() == "sentencepiece_unigram":
                learned_tokens, success = train_sentencepiece_tokenizer(texts, current_vocab_size, "unigram")
            else:
                return [], False
            
            if not success:
                logging.warning(f"Training failed at iteration {iteration + 1}")
                current_vocab_size = int(current_vocab_size * 1.2)
                continue
            
            # Filter for unique tokens
            unique_tokens = get_unique_tokens(learned_tokens, target_tokenizer, algorithm, model_name)
            actual_size = len(unique_tokens)
            
            logging.info(f"Got {actual_size} unique tokens (need ≥{target_size})")
            
            # Store this result
            all_results.append({
                'size': actual_size,
                'tokens': unique_tokens.copy(),
                'vocab_size': current_vocab_size
            })
            
            # Keep track of best result that meets criteria (≥ target_size)
            if actual_size >= target_size:
                if best_size == 0 or actual_size < best_size:  # Closest from above
                    best_tokens = unique_tokens.copy()
                    best_size = actual_size
                    logging.info(f"✅ NEW BEST: {actual_size} tokens (≥{target_size})")
                
                # If we found exactly the target size, we can stop
                if actual_size == target_size:
                    logging.info(f"🎯 PERFECT: Found exactly {target_size} tokens!")
                    return unique_tokens[:target_size], True
                
                # If we're close to target, try to find something even closer
                if actual_size <= target_size * 1.1:  # Within 10% above target
                    # Try slightly smaller vocab size to get closer
                    current_vocab_size = int(current_vocab_size * 0.95)
                # within 50% above target, try to reduce vocab size
                elif actual_size <= target_size * 1.5:
                    # Too many tokens, reduce vocab size more aggressively
                    current_vocab_size = int(current_vocab_size * 0.85)
                # if we are more than 50% above target, reduce vocab size significantly
                else:
                    current_vocab_size = int(current_vocab_size * 0.75)
            else:
                # Not enough tokens, increase vocabulary size
                current_vocab_size = int(current_vocab_size * 1.3)
                
        except Exception as e:
            logging.error(f"Error in iteration {iteration + 1}: {e}")
            current_vocab_size = int(current_vocab_size * 1.2)
            continue
    
    # If we found any result ≥ target_size, use the best one
    if best_tokens and best_size >= target_size:
        final_tokens = best_tokens[:target_size]  # Take exactly target_size
        logging.info(f"✅ BEST RESULT: Using {len(final_tokens)} tokens from {best_size} available (≥{target_size})")
        return final_tokens, True
    
    # Fallback: find the largest result we got, even if < target_size
    if all_results:
        best_result = max(all_results, key=lambda x: x['size'])
        fallback_tokens = best_result['tokens'][:target_size] if len(best_result['tokens']) >= target_size else best_result['tokens']
        logging.warning(f"FALLBACK: Using {len(fallback_tokens)} tokens (best we could achieve)")
        return fallback_tokens, True
    
    return [], False


# Factory function to get appropriate trainer
def get_trainer(algorithm: str):
    """
    Factory function to get the appropriate trainer for an algorithm.
    
    Args:
        algorithm (str): Algorithm name
    
    Returns:
        callable: Training function for the algorithm
    """
    if algorithm.lower() == "bpe":
        return train_bpe_tokenizer
    elif algorithm.lower() == "wordpiece":
        return train_wordpiece_tokenizer
    elif algorithm.lower() == "unigram":
        return train_unigram_tokenizer
    elif algorithm.lower().startswith("sentencepiece"):
        # Extract model type and create partial function
        model_type = algorithm.split("_")[1] if "_" in algorithm else "bpe"
        return lambda texts, vocab_size: train_sentencepiece_tokenizer(texts, vocab_size, model_type)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    

def debug_token_loss(learned_tokens: List[str], target_tokenizer, algorithm: str):
    """Find exactly where tokens are lost."""
    print(f"Input: {len(learned_tokens)} tokens")
    
    # Step 1: Process tokens
    processed = []
    for token in learned_tokens:
        new_token = token.replace("##", "") if "wordpiece" in algorithm.lower() else token
        if len(new_token.strip()) > 0:
            processed.append(new_token)
    print(f"After processing: {len(processed)} tokens")
    
    # Step 2: Check duplicates
    unique_processed = list(set(processed))
    print(f"After deduplication: {len(unique_processed)} tokens")
    
    # Step 3: Check existing in vocab
    vocab = target_tokenizer.get_vocab()
    truly_new = [t for t in unique_processed if t not in vocab]
    print(f"Truly new: {len(truly_new)} tokens")
    
    # Step 4: Test actual addition
    import copy
    test_tokenizer = copy.deepcopy(target_tokenizer)
    added = test_tokenizer.add_tokens(unique_processed)
    print(f"Actually added: {added} tokens")
    
    return len(processed) - len(learned_tokens), len(unique_processed) - len(processed), len(truly_new) - len(unique_processed)