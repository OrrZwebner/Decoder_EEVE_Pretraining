#!/usr/bin/env python3
"""
Data loading and processing utilities for Gemma-3 Sanskrit training
Location: /home/orrz/gpufs/projects/gemma3/src/data_utils.py
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset


class LocalSanskritDataset(Dataset):
    """
    PyTorch Dataset for Sanskrit texts with tokenization
    """
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            texts: List of Sanskrit texts
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logging.info(f"Initialized Sanskrit dataset with {len(texts)} texts")
        logging.info(f"Max length: {max_length}")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get tokenized text sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()  # For causal LM
        }


def load_sanskrit_texts_from_file(file_path: str) -> List[str]:
    """
    Load Sanskrit texts from a single file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List of Sanskrit text strings
    """
    texts = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        logging.warning(f"File not found: {file_path}")
        return texts
    
    try:
        if file_path.suffix == '.txt':
            # Load from text file (one text per line)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
            
            logging.info(f"Loaded {len(texts)} texts from {file_path}")
            
        elif file_path.suffix == '.json':
            # Load from JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                texts = [str(item) for item in data if item]
            elif isinstance(data, dict):
                # Try common keys for text data
                for key in ['texts', 'data', 'content', 'sentences']:
                    if key in data and isinstance(data[key], list):
                        texts = [str(item) for item in data[key] if item]
                        break
                
                if not texts:
                    logging.warning(f"No recognized text key found in JSON: {file_path}")
            
            logging.info(f"Loaded {len(texts)} texts from JSON file {file_path}")
            
        else:
            logging.warning(f"Unsupported file format: {file_path}")
    
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
    
    return texts


def get_fallback_sanskrit_texts(multiplier: int = 10) -> List[str]:
    """
    Get fallback Sanskrit texts for testing
    
    Args:
        multiplier: Number of times to repeat the fallback texts
        
    Returns:
        List of fallback Sanskrit texts
    """
    fallback_texts = [
        "रागादि-रोगान् सततानुषक्तान् अ-शेष-काय-प्रसृतान् अ-शेषान् ।",
        "औत्सुक्य-मोहा-रति-दाञ् जघान यो ऽ-पूर्व-वैद्याय नमो ऽस्तु तस्मै ॥",
        "आयुः-कामयमानेन धर्मार्थ-सुख-साधनम् ।",
        "आयुर्-वेदोपदेशेषु विधेयः परम् आदरः ॥",
        "ब्रह्मा स्मृत्वायुषो वेदं प्रजापतिम् अजिग्रहत् ।",
        "सो ऽश्विनौ तौ सहस्राक्षं सो ऽत्रि-पुत्रादिकान् मुनीन् ॥",
        "ते ऽग्निवेशादिकांस् ते तु पृथक् तन्त्राणि तेनिरे ।",
        "तेभ्यो ऽति-विप्रकीर्णेभ्यः प्रायः सार-तरोच्चयः ॥",
        "क्रियते ऽष्टाङ्ग-हृदयं नाति-संक्षेप-विस्तरम् ।",
        "काय-बाल-ग्रहोर्ध्वाङ्ग-शल्य-दंष्ट्रा-जरा-वृषान् ॥",
        "वेदोत्पन्नमयुर्वेदं सशाङ्गमुपबृंहितम् ।",
        "चरकेणोपदिष्टं यन्नास्ति तन्नेह किञ्चन ॥",
        "धात्वग्निमान्द्यं सर्वेषां रोगाणां कारणं स्मृतम् ।",
        "अग्निदीपनपाचना औषधानि विशेषतः ॥",
        "सर्वेषामेव रोगाणाम् आदौ लङ्घनमुच्यते ।",
        "कफपित्तानिलाणां हि दोषाणां जनकं हि तत् ॥"
    ]
    
    # Repeat texts based on multiplier
    repeated_texts = fallback_texts * multiplier
    
    logging.info(f"Generated {len(repeated_texts)} fallback Sanskrit texts")
    return repeated_texts


def apply_debug_data_limiting(texts: List[str], config, dataset_type: str = "train") -> List[str]:
    """
    Apply debug data limiting based on configuration
    
    Args:
        texts: List of text samples
        config: Configuration object
        dataset_type: "train" or "eval" to determine which limit to apply
        
    Returns:
        Limited list of texts if debug limiting is enabled
    """
    debug_config = config.get('debug', {})
    if not debug_config:
        return texts

    if not debug_config.get('test_mode', False):
        # If not in test mode, we don't limit data
        logging.info(f"🐛 DEBUG: Not in test mode, skipping data limiting for {dataset_type} dataset")
        return texts
    else:
        logging.info(f"🐛 DEBUG: Test mode enabled, applying data limiting for {dataset_type} dataset")


    # Determine which limit to apply
    limit_key = f'limit_{dataset_type}_samples'
    if limit_key not in debug_config:
        return texts

    limit = debug_config.get(limit_key, 0)

    
    # Apply limit if specified (> 0)
    if limit > 0 and len(texts) > limit:
        original_count = len(texts)
        limited_texts = texts[:limit]
        
        logging.info(f"🐛 DEBUG: Limited {dataset_type} data from {original_count} to {limit} samples")
        return limited_texts
    
    return texts


def load_sanskrit_dataset(config) -> List[str]:
    """
    Load Sanskrit texts from configured data sources
    
    Args:
        config: Configuration object
        
    Returns:
        List of Sanskrit text strings
    """
    texts = []
    
    # Try to load from configured file paths
    data_config = config['data']
    file_paths = data_config.get('file_paths', [])
    
    for file_path in file_paths:
        file_texts = load_sanskrit_texts_from_file(file_path)
        texts.extend(file_texts)
        
        if file_texts:  # If we found texts in this file, we can stop looking
            logging.info(f"Successfully loaded data from {file_path}")
            break
    
    # Use fallback if no texts found and fallback is enabled
    if not texts and data_config.get('use_fallback', False):
        multiplier = data_config.get('fallback_multiplier', 10)
        texts = get_fallback_sanskrit_texts(multiplier)
        logging.info("Using fallback Sanskrit texts")
    
    # Validate we have some texts
    if not texts:
        raise ValueError("No Sanskrit texts loaded and fallback is disabled")
    
    logging.info(f"Total loaded texts: {len(texts)}")
    
    return texts


def split_train_eval_data(texts: List[str], config) -> Tuple[List[str], List[str]]:
    """
    Split texts into training and evaluation sets with debug limiting
    
    Args:
        texts: List of all texts
        config: Configuration object
        
    Returns:
        Tuple of (train_texts, eval_texts)
    """
    data_config = config['data']

    eval_ratio = data_config.get('eval_split_ratio', 0.1)
    random_seed = data_config.get('random_seed', 42)
    
    if eval_ratio <= 0 or eval_ratio >= 1:
        logging.warning(f"Invalid eval split ratio: {eval_ratio}, using 0.1")
        eval_ratio = 0.1
    
    # Perform the split
    train_texts, eval_texts = train_test_split(
        texts,
        test_size=eval_ratio,
        random_state=random_seed
    )
    
    logging.info(f"Data split - Train: {len(train_texts)}, Eval: {len(eval_texts)}")
    
    # Apply debug limiting after split
    train_texts = apply_debug_data_limiting(train_texts, config, "train")
    eval_texts = apply_debug_data_limiting(eval_texts, config, "eval")
    
    logging.info(f"Final data counts - Train: {len(train_texts)}, Eval: {len(eval_texts)}")
    
    return train_texts, eval_texts


def create_datasets(train_texts: List[str], eval_texts: List[str], 
                   tokenizer, config) -> Tuple[LocalSanskritDataset, LocalSanskritDataset]:
    """
    Create PyTorch datasets from text lists
    
    Args:
        train_texts: Training texts
        eval_texts: Evaluation texts
        tokenizer: HuggingFace tokenizer
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    tokenizer_config = config['tokenizer']
    max_length = tokenizer_config.get('max_length', 256)
    
    # Create datasets
    train_dataset = LocalSanskritDataset(train_texts, tokenizer, max_length)
    eval_dataset = LocalSanskritDataset(eval_texts, tokenizer, max_length)
    
    logging.info(f"Created datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_data_collator(tokenizer, config) -> DataCollatorForLanguageModeling:
    """
    Create data collator for language modeling
    
    Args:
        tokenizer: HuggingFace tokenizer
        config: Configuration object
        
    Returns:
        DataCollatorForLanguageModeling instance
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    logging.info("Created data collator for causal language modeling")
    
    return data_collator


def print_dataset_info(train_dataset: Dataset, eval_dataset: Dataset, 
                      tokenizer, config, stage: int = 0) -> None:
    """
    Print information about the datasets with EEVE stage awareness
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: HuggingFace tokenizer (potentially with expanded vocabulary)
        config: Configuration object
        stage: Current EEVE training stage (0 for standard training)
    """
    if not config.get('debug', {}).get('print_dataset_info', False):
        return
    
    print("\n" + "="*60)
    if stage > 0:
        print(f"DATASET INFORMATION - EEVE STAGE {stage}")
        # Get stage description if available
        if 'eeve' in config and 'stages' in config['eeve']:
            stage_config = config['eeve']['stages'].get(stage, {})
            stage_desc = stage_config.get('description', f'Stage {stage}')
            print(f"Stage: {stage_desc}")
    else:
        print("DATASET INFORMATION")
    print("="*60)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Max sequence length: {train_dataset.max_length}")
    
    # Enhanced tokenizer information with vocabulary expansion details
    print(f"\nTokenizer Information:")
    print(f"  Current vocabulary size: {len(tokenizer):,}")
    
    # Show vocabulary expansion details if custom vocabulary was used
    if 'vocabulary' in config:
        vocab_config = config['vocabulary']
        if vocab_config.get('use_custom_vocabulary', False):
            print(f"  🔤 Custom vocabulary: ENABLED")
            str_num_tokens = str(config.get('vocabulary_generation', {}).get('num_tokens', ""))
            vocab_file_path = vocab_config['vocabulary_full_path'] + str_num_tokens +'.pkl'
            print(f"  Vocabulary file: {vocab_file_path}")
            print(f"  Addition method: {vocab_config.get('add_tokens_method', 'N/A')}")
            
            # Estimate tokens added (simplified calculation)
            # estimated_original_size = 256000  # Approximate Gemma-3 base vocab size
            # current_size = len(tokenizer)
            # estimated_added = max(0, current_size - estimated_original_size)
            # if estimated_added > 0:
            #     print(f"  Estimated tokens added: ~{estimated_added:,}")
        else:
            print(f"  🔤 Custom vocabulary: DISABLED (using original tokenizer)")
    
    # Show EEVE stage information
    if stage > 0:
        print(f"\nEEVE Stage Information:")
        print(f"  Current stage: {stage}")
        
        if 'eeve' in config:
            eeve_config = config['eeve']
            
            # Show stage configuration
            if 'stages' in eeve_config:
                stage_info = eeve_config['stages'].get(stage, {})
                if stage_info:
                    print(f"  Training layers: {stage_info.get('train_layers', 'N/A')}")
                    print(f"  Epochs for this stage: {stage_info.get('epochs', 'N/A')}")
                    
                    # Show if this stage has custom learning rate
                    if 'learning_rate' in stage_info:
                        print(f"  Stage learning rate: {stage_info['learning_rate']}")
            
            # Show overall EEVE progress
            start_stage = eeve_config.get('start_stage', 1)
            end_stage = eeve_config.get('end_stage', 7)
            progress = ((stage - start_stage + 1) / (end_stage - start_stage + 1)) * 100
            print(f"  EEVE progress: {stage}/{end_stage} stages ({progress:.0f}%)")
            
            # Show n_added_tokens if relevant for this stage
            n_added_tokens = eeve_config.get('n_added_tokens', 0)
            if n_added_tokens > 0 and stage <= 3:  # Stages 1-3 use added tokens
                print(f"  Added tokens count: {n_added_tokens:,}")
    
    # Show debug information if applicable
    if 'debug' in config:
        debug_config = config['debug']
        print(f"\nDebug Information:")
        if debug_config.get('limit_train_samples', 0) > 0:
            print(f"  🐛 Training data limited to {debug_config['limit_train_samples']} samples")
        if debug_config.get('limit_eval_samples', 0) > 0:
            print(f"  🐛 Evaluation data limited to {debug_config['limit_eval_samples']} samples")
        if debug_config.get('test_mode', False):
            print(f"  🐛 Test mode enabled")
    
    # Show sample from training dataset
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\nSample Training Data:")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        
        # Decode first few tokens for inspection
        decoded_text = tokenizer.decode(sample['input_ids'][100:150], skip_special_tokens=True)
        print(f"  Decoded text (sampled tokens): {decoded_text}")
        
        # Show vocabulary coverage if custom vocabulary is used
        if 'vocabulary' in config:
            vocab_config = config['vocabulary']
            if vocab_config.get('use_custom_vocabulary', False):
                # Count how many tokens in the sample are from the extended vocabulary
                vocab_size_original = 256000  # Approximate original Gemma-3 vocab size
                current_vocab_size = len(tokenizer)
                
                if current_vocab_size > vocab_size_original:
                    # Count tokens that are likely from custom vocabulary
                    custom_token_ids = sample['input_ids'][sample['input_ids'] >= vocab_size_original]
                    if len(custom_token_ids) > 0:
                        custom_token_count = len(custom_token_ids)
                        total_tokens = len(sample['input_ids'][sample['input_ids'] != tokenizer.pad_token_id])
                        percentage = (custom_token_count / total_tokens) * 100
                        print(f"  Custom vocabulary usage: {custom_token_count}/{total_tokens} tokens ({percentage:.1f}%)")
                    else:
                        print(f"  Custom vocabulary usage: No custom tokens in this sample")
    
    print("="*60 + "\n")


def setup_tokenizer_for_sanskrit(tokenizer, config) -> None:
    """
    Configure tokenizer for Sanskrit text processing
    
    Args:
        tokenizer: HuggingFace tokenizer
        config: Configuration object
    """
    tokenizer_config = config['tokenizer']
    
    # Add pad token if missing and configured
    if tokenizer_config.get('add_pad_token', True) and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Added pad token to tokenizer")
    
    logging.info("Tokenizer configured for Sanskrit processing")


def load_data_pipeline(tokenizer, config) -> Tuple[Dataset, Dataset, DataCollatorForLanguageModeling]:
    """
    Complete data loading pipeline with debug support
    
    Args:
        tokenizer: HuggingFace tokenizer
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, eval_dataset, data_collator)
    """
    logging.info("Starting data loading pipeline...")
    
    # Configure tokenizer
    setup_tokenizer_for_sanskrit(tokenizer, config)
    
    # Load Sanskrit texts
    texts = load_sanskrit_dataset(config)
    
    # Split into train/eval with debug limiting
    train_texts, eval_texts = split_train_eval_data(texts, config)
    
    # Create datasets
    train_dataset, eval_dataset = create_datasets(train_texts, eval_texts, tokenizer, config)
    
    # Create data collator
    data_collator = create_data_collator(tokenizer, config)
    
    # Print dataset info if debug mode
    print_dataset_info(train_dataset, eval_dataset, tokenizer, config)
    
    logging.info("✅ Data loading pipeline completed")
    
    return train_dataset, eval_dataset, data_collator


if __name__ == "__main__":
    # Test data utilities
    print("Testing data utilities...")
    
    # This would require a config object, so just test individual functions
    fallback_texts = get_fallback_sanskrit_texts(2)
    print(f"Generated {len(fallback_texts)} fallback texts")
    print(f"Sample text: {fallback_texts[0]}")
    
    print("Data utilities test completed!")