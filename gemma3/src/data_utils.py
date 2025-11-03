#!/usr/bin/env python3
"""
Data loading and processing utilities for Gemma-3 Sanskrit training
Location: /home/orrz/gpufs/projects/gemma3/src/data_utils.py
"""
import sys; print(f"⚠️  data_utils.py imported! CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}", file=sys.stderr, flush=True)


import os
import json

# DIAGNOSTIC: Print when this module is imported
_cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT_SET')
print(f"🔍 [IMPORT DEBUG] data_utils.py being imported with CUDA_VISIBLE_DEVICES={_cuda_env}", file=sys.stderr)




import logging
from pathlib import Path
# Verify CUDA_VISIBLE_DEVICES is set before allowing torch imports
_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if _cuda_visible is None:
    raise RuntimeError(
        "CRITICAL: data_utils.py imported before CUDA_VISIBLE_DEVICES was set!\n"
        "This will cause GPU device assignment failures.\n"
        "Ensure config loading and environment setup happens BEFORE importing data_utils."
    )

import torch
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset, DatasetDict, IterableDataset, IterableDatasetDict

logging.info(f"✅ data_utils loaded with CUDA_VISIBLE_DEVICES={_cuda_visible}")


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
            logging.info(f"Loading text file: {file_path}")
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

        # handling for JSONL files
        elif file_path.suffix == '.jsonl':
            logging.info(f"Loading JSONL file: {file_path}")
            texts = []  # Initialize empty list
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        # Extract text from your specific format
                        if 'text' in obj and isinstance(obj['text'], str):
                            texts.append(obj['text'].strip())
            
            logging.info(f"Loaded {len(texts)} texts from JSONL file {file_path}")

        else:
            logging.warning(f"Unsupported file format: {file_path}")
    
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
    
    return texts


def load_huggingface_dataset(config) -> Union[HFDataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """
    Load dataset from HuggingFace Hub
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HuggingFace Dataset or DatasetDict
    """
    data_config = config['data']
    
    # Extract HF configuration
    dataset_name = data_config.get('hf_dataset_name')
    dataset_config = data_config.get('hf_dataset_config')
    split = data_config.get('hf_split', 'auto')
    streaming = data_config.get('hf_streaming', False)
    trust_remote_code = data_config.get('hf_trust_remote_code', False)
    auto_use_validation = data_config.get('hf_auto_use_validation', True)  
    
    if not dataset_name:
        raise ValueError("hf_dataset_name must be specified when source_type is 'huggingface'")
    
    logging.info(f"Loading HuggingFace dataset: {dataset_name}")
    if dataset_config:
        logging.info(f"  Config/subset: {dataset_config}")
    logging.info(f"  Split: {split}")
    logging.info(f"  Streaming: {streaming}")
    
    try:
        # If auto_use_validation is True and split is 'train', 
        # load all splits to check for validation
        load_split = split
        
        if auto_use_validation and split == 'train':
            # Load without specifying split to get DatasetDict
            logging.info("  Checking for validation split...")
            load_split = None
        elif split == 'auto':
            load_split = None
        
        # Load dataset from HF Hub
        dataset = load_dataset(
            dataset_name,
            name=dataset_config,
            split=load_split,
            streaming=streaming,
            trust_remote_code=trust_remote_code
        )
        
        logging.info(f"✅ Successfully loaded HuggingFace dataset: {dataset_name}")
        logging.info(f"  Dataset type: {type(dataset).__name__}")
        
        # Handle DatasetDict or IterableDatasetDict - check for validation split
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            logging.info(f"  Available splits: {list(dataset.keys())}")
            
            # If we loaded all splits to check for validation
            if auto_use_validation and split == 'train':
                has_validation = any(key in dataset for key in ['validation', 'val', 'dev', 'test'])
                
                if has_validation:
                    logging.info(f"  ✅ Found validation split, will use existing splits")
                    # Log split info before returning
                    for split_name, split_dataset in dataset.items():
                        if not streaming:
                            logging.info(f"    {split_name}: {len(split_dataset)} samples")
                        else:
                            logging.info(f"    {split_name}: streaming")
                    # Return the DatasetDict/IterableDatasetDict - split_train_eval_data will handle it
                    return dataset
                else:
                    logging.info(f"  ⚠️ No validation split found, will create from train")
                    # Return only train split - will be split later
                    dataset = dataset['train']
                    logging.info(f"  Extracted train split, type: {type(dataset).__name__}")
        else:
            if not streaming:
                logging.info(f"  Loaded {len(dataset)} samples")
            else:
                logging.info(f"  Loaded streaming dataset")
        
        return dataset
        
    except Exception as e:
        logging.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
        raise


def prepare_hf_dataset_for_training(hf_dataset: Union[HFDataset, IterableDataset], tokenizer, config) ->Union[HFDataset, IterableDataset]:
    """
    Tokenize HF dataset using map for efficiency
    Works with both regular Dataset and IterableDataset (streaming)
    
    Args:
        hf_dataset: Raw HF dataset or IterableDataset
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary
        
    Returns:
        Tokenized HF dataset ready for Trainer
    """
    data_config = config['data']
    tokenizer_config = config['tokenizer']
    
    text_column = data_config.get('hf_text_column', 'text')
    max_length = tokenizer_config.get('max_length', 256)
    
    is_streaming = isinstance(hf_dataset, IterableDataset)
    
    logging.info(f"Preparing HF dataset for training (streaming={is_streaming})...")
    logging.info(f"  Text column: {text_column}")
    logging.info(f"  Max length: {max_length}")
    
    # For non-streaming, verify text column exists
    if not is_streaming and text_column not in hf_dataset.column_names:
        available_columns = ", ".join(hf_dataset.column_names)
        raise ValueError(
            f"Text column '{text_column}' not found in dataset. "
            f"Available columns: {available_columns}"
        )
    
    def tokenize_function(examples):
        """Tokenization function for dataset.map()"""
        # Tokenize the texts
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            # padding="max_length",
            padding=False,  # Let DataCollator handle padding
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors (for .map())
        )

        
        return tokenized
    
    # Show sample raw text before tokenization
    if config.get('debug', {}).get('verbose_logging', False):
        logging.info("\n" + "="*80)
        logging.info("SAMPLE RAW TEXT (Before Tokenization)")
        logging.info("="*80)
        
        try:
            if is_streaming:
                # For streaming, peek at first item
                sample_iter = iter(hf_dataset)
                raw_sample = next(sample_iter)
            else:
                raw_sample = hf_dataset[0]
            
            raw_text = raw_sample[text_column]
            logging.info(f"Text length: {len(raw_text)} characters")
            logging.info(f"Text preview (first 50 chars):")
            logging.info(f"  {raw_text[:50]}...")
            logging.info(f"\nText preview (last 20 chars):")
            logging.info(f"  ...{raw_text[-20:]}")
            
        except Exception as e:
            logging.warning(f"Could not show raw text sample: {e}")
        
        logging.info("="*80 + "\n")
    
    # Apply tokenization with batching for efficiency
    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        # remove_columns=hf_dataset.column_names if not is_streaming else [text_column],
        # desc="Tokenizing dataset" if not is_streaming else None
    )
    
    # Set format to PyTorch tensors (only for non-streaming)
    if not is_streaming:
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        logging.info(f"✅ Dataset tokenized and ready for training")
    else:
        logging.info(f"✅ Streaming dataset tokenized and ready for training")

        # Show sample tokenization for debugging
    if config.get('debug', {}).get('verbose_logging', False):
        logging.info("\n" + "="*80)
        logging.info("SAMPLE TOKENIZATION (First Example)")
        logging.info("="*80)
        
        # Get first example from tokenized dataset
        try:
            if is_streaming:
                # For streaming, take first item
                sample_iter = iter(tokenized_dataset)
                sample = next(sample_iter)
            else:
                sample = tokenized_dataset[0]
            
            # Show tokenized data
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            
            logging.info(f"Input IDs (first 50): {input_ids[:50]}")
            logging.info(f"Attention mask (first 50): {attention_mask[:50]}")
            logging.info(f"Total tokens: {len(input_ids)}")
            logging.info(f"Non-padding tokens: {sum(attention_mask)}")
            
            # Decode the tokens
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            decoded_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
            
            logging.info(f"\nDecoded text (first 200 chars):")
            logging.info(f"  {decoded_text[:200]}...")
            
            logging.info(f"\nWith special tokens (first 200 chars):")
            logging.info(f"  {decoded_with_special[:200]}...")
            
            # Show token breakdown
            tokens = tokenizer.convert_ids_to_tokens(input_ids[:20])
            logging.info(f"\nFirst 20 tokens: {tokens}")
            
            # Count special tokens
            pad_count = sum(1 for tid in input_ids if tid == tokenizer.pad_token_id)
            eos_count = sum(1 for tid in input_ids if tid == tokenizer.eos_token_id)
            bos_count = sum(1 for tid in input_ids if tid == tokenizer.bos_token_id) if tokenizer.bos_token_id else 0
            
            logging.info(f"\nSpecial token counts:")
            logging.info(f"  Padding tokens: {pad_count}")
            logging.info(f"  EOS tokens: {eos_count}")
            logging.info(f"  BOS tokens: {bos_count}")
            
        except Exception as e:
            logging.warning(f"Could not show sample tokenization: {e}")
        
        logging.info("="*80 + "\n")
    
    # logging.info(f"✅ Dataset tokenized and ready for training")
    

    return tokenized_dataset




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


def apply_debug_data_limiting(data: Union[List[str], HFDataset, IterableDataset], config, dataset_type: str = "train") -> Union[List[str], HFDataset, IterableDataset]:
    """
    Apply debug data limiting based on configuration
    Supports List[str], HuggingFace Dataset, and IterableDataset (streaming)
    
    Args:
        data: List of text samples, HF Dataset, or IterableDataset
        config: Configuration object
        dataset_type: "train" or "eval" to determine which limit to apply
        
    Returns:
        Limited data if debug limiting is enabled
    """
    debug_config = config.get('debug', {})
    if not debug_config:
        return data

    if not debug_config.get('test_mode', False):
        logging.info(f"🐛 DEBUG: Not in test mode, skipping data limiting for {dataset_type} dataset")
        return data
    else:
        logging.info(f"🐛 DEBUG: Test mode enabled, applying data limiting for {dataset_type} dataset")

    # Determine which limit to apply
    limit_key = f'limit_{dataset_type}_samples'
    if limit_key not in debug_config:
        return data

    limit = debug_config.get(limit_key, 0)
    
    # Apply limit if specified (> 0)
    if limit > 0:
        # Handle IterableDataset (streaming)
        if isinstance(data, IterableDataset):
            logging.info(f"🐛 DEBUG: Limiting {dataset_type} IterableDataset to first {limit} samples (streaming)")
            return data.take(limit)
        
        # Handle regular HF Dataset
        elif isinstance(data, HFDataset):
            if len(data) > limit:
                original_count = len(data)
                limited_data = data.select(range(limit))
                logging.info(f"🐛 DEBUG: Limited {dataset_type} HF dataset from {original_count} to {limit} samples")
                return limited_data
        
        # Handle List[str]
        elif isinstance(data, list):
            if len(data) > limit:
                original_count = len(data)
                limited_data = data[:limit]
                logging.info(f"🐛 DEBUG: Limited {dataset_type} data from {original_count} to {limit} samples")
                return limited_data
    
    return data


def load_sanskrit_dataset(config) -> Union[List[str], HFDataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """
    Load Sanskrit texts from configured data sources
    
    Args:
        config: Configuration object
        
    Returns:
        List[str] for local files, or HF Dataset/DatasetDict for HF sources
    """
    data_config = config['data']
    source_type = data_config.get('source_type', 'local_files')
    
    logging.info(f"Loading dataset with source_type: {source_type}")
    
    # Route based on source type
    if source_type == 'huggingface':
        return load_huggingface_dataset(config)
    
    elif source_type == 'local_files':
        # Existing file loading logic
        texts = []
        file_paths = data_config.get('file_paths', [])
        
        for file_path in file_paths:
            file_texts = load_sanskrit_texts_from_file(file_path)
            texts.extend(file_texts)
            logging.info(f"Successfully loaded data from {file_path}")
        
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
    
    else:
        raise ValueError(f"Unknown source_type: {source_type}. Must be 'local_files' or 'huggingface'")


def split_train_eval_data(data: Union[List[str], HFDataset, DatasetDict, IterableDataset, IterableDatasetDict],
                           config) -> Tuple[Union[List[str], HFDataset, IterableDataset, IterableDatasetDict], Union[List[str], HFDataset, IterableDataset, IterableDatasetDict]]:
    """
    Split data into training and evaluation sets with debug limiting
    Handles List[str], HuggingFace Dataset, DatasetDict, and IterableDataset (streaming)    
    
    Args:
        data: List of all texts or HF Dataset/DatasetDict
        config: Configuration object
        
    Returns:
        Tuple of (train_data, eval_data)
    """
    data_config = config['data']
    eval_ratio = data_config.get('eval_split_ratio', 0.1)
    random_seed = data_config.get('random_seed', 42)
    
    if eval_ratio <= 0 or eval_ratio >= 1:
        logging.warning(f"Invalid eval split ratio: {eval_ratio}, using 0.1")
        eval_ratio = 0.1
    
    # Handle IterableDatasetDict (streaming with multiple splits)
    if isinstance(data, IterableDatasetDict):
        logging.info("Dataset is IterableDatasetDict (streaming with splits)")
        logging.info(f"  Available splits: {list(data.keys())}")
        
        train_data = data.get('train')
        # Check multiple common names for validation split
        eval_data = data.get('validation') or data.get('val') or data.get('dev') or data.get('test')
        
        if train_data is None:
            raise ValueError("IterableDatasetDict must contain a 'train' split")
        
        if eval_data is None:
            logging.info(f"⚠️ No validation split found in IterableDatasetDict, creating from train")
            eval_size = data_config.get('hf_eval_size', None)
            train_data, eval_data = split_streaming_dataset(train_data, eval_size, eval_ratio)
        else:
            logging.info(f"✅ Using existing validation split from IterableDatasetDict")
        
        # Apply debug limiting
        train_data = apply_debug_data_limiting(train_data, config, "train")
        eval_data = apply_debug_data_limiting(eval_data, config, "eval")
        
        logging.info("Streaming splits ready")
        
        return train_data, eval_data
    
    # Handle IterableDataset (streaming, single split - need to create eval)
    elif isinstance(data, IterableDataset):
        logging.info("Dataset is IterableDataset (streaming single split), creating eval split")
        
        eval_size = data_config.get('hf_eval_size', None)
        train_data, eval_data = split_streaming_dataset(data, eval_size, eval_ratio)
        
        train_data = apply_debug_data_limiting(train_data, config, "train")
        eval_data = apply_debug_data_limiting(eval_data, config, "eval")
        
        logging.info("Streaming split completed (exact counts not available until iteration)")
        
        return train_data, eval_data
    
    # Handle HuggingFace DatasetDict (non-streaming with splits)
    elif isinstance(data, DatasetDict):
        logging.info("Dataset is DatasetDict with pre-existing splits")
        logging.info(f"  Available splits: {list(data.keys())}")
        
        train_data = data.get('train')
        # Check multiple common names for validation split
        eval_data = data.get('validation') or data.get('val') or data.get('dev') or data.get('test')
        
        if train_data is None:
            raise ValueError("DatasetDict must contain a 'train' split")
        
        if eval_data is None:
            logging.info(f"⚠️ No validation split found in DatasetDict, creating from train")
            
            # Check if train is streaming
            if isinstance(train_data, IterableDataset):
                eval_size = data_config.get('hf_eval_size', None)
                train_data, eval_data = split_streaming_dataset(train_data, eval_size, eval_ratio)
            else:
                split = train_data.train_test_split(test_size=eval_ratio, seed=random_seed)
                train_data = split['train']
                eval_data = split['test']
        else:
            logging.info(f"✅ Using existing validation split")
        
        logging.info(f"Data split - Train: {len(train_data)}, Eval: {len(eval_data)}")
        
        # Apply debug limiting
        train_data = apply_debug_data_limiting(train_data, config, "train")
        eval_data = apply_debug_data_limiting(eval_data, config, "eval")
        
        logging.info(f"Final data counts - Train: {len(train_data)}, Eval: {len(eval_data)}")
        
        return train_data, eval_data
    
    # Handle HuggingFace Dataset (single split, needs splitting)
    elif isinstance(data, HFDataset):
        logging.info("Dataset is HF Dataset, splitting into train/eval")
        
        split = data.train_test_split(
            test_size=eval_ratio,
            seed=random_seed
        )
        train_data = split['train']
        eval_data = split['test']
        
        logging.info(f"Data split - Train: {len(train_data)}, Eval: {len(eval_data)}")
        
        # Apply debug limiting
        train_data = apply_debug_data_limiting(train_data, config, "train")
        eval_data = apply_debug_data_limiting(eval_data, config, "eval")
        
        logging.info(f"Final data counts - Train: {len(train_data)}, Eval: {len(eval_data)}")
        
        return train_data, eval_data
    
    # Handle List[str] (original behavior)
    else:
        logging.info("Dataset is List[str], using sklearn split")
        
        train_texts, eval_texts = train_test_split(
            data,
            test_size=eval_ratio,
            random_state=random_seed
        )
        
        logging.info(f"Data split - Train: {len(train_texts)}, Eval: {len(eval_texts)}")
        
        # Apply debug limiting after split
        train_texts = apply_debug_data_limiting(train_texts, config, "train")
        eval_texts = apply_debug_data_limiting(eval_texts, config, "eval")
        
        logging.info(f"Final data counts - Train: {len(train_texts)}, Eval: {len(eval_texts)}")
        
        return train_texts, eval_texts


def split_streaming_dataset(dataset: IterableDataset, eval_size: int, eval_ratio: float = 0.1) -> Tuple[IterableDataset, IterableDataset]:
    """
    Split streaming dataset into train and eval sets
    
    Args:
        dataset: Streaming IterableDataset
        eval_size: Number of samples for eval (if > 0, uses this; otherwise uses ratio)
        eval_ratio: Ratio for eval split (used only if eval_size is None or 0)
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    if eval_size and eval_size > 0:
        # Use explicit eval size
        logging.info(f"Creating streaming split with eval_size={eval_size}")
        
        # Take eval samples from the beginning
        eval_dataset = dataset.take(eval_size)
        
        # Skip eval samples for train
        train_dataset = dataset.skip(eval_size)
        
        logging.info(f"Created streaming train/eval split - Eval: {eval_size}, Train: rest of dataset")
        
    else:
        # Fall back to filtering approach with ratio
        logging.warning("⚠️ No eval_size specified, using filtering with eval_ratio")
        
        train_dataset = dataset.filter(lambda example, idx: idx % int(1/eval_ratio) != 0, with_indices=True)
        eval_dataset = dataset.filter(lambda example, idx: idx % int(1/eval_ratio) == 0, with_indices=True)
        
        logging.info(f"Created streaming train/eval split using filtering (ratio={eval_ratio})")
    
    return train_dataset, eval_dataset



def create_datasets(train_data: Union[List[str], HFDataset, IterableDataset], eval_data: Union[List[str], HFDataset, IterableDataset],
 tokenizer, config) -> Tuple[Union[LocalSanskritDataset, HFDataset, IterableDataset], Union[LocalSanskritDataset, HFDataset, IterableDataset]]:
    """
    Create PyTorch datasets from text lists or prepare HF datasets
    Supports streaming datasets
    
    Args:
        train_data: Training texts, HF Dataset, or IterableDataset
        eval_data: Evaluation texts, HF Dataset, or IterableDataset
        tokenizer: HuggingFace tokenizer
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
     # If HF Dataset or IterableDataset, tokenize and return
    if isinstance(train_data, (HFDataset, IterableDataset)):
        is_streaming = isinstance(train_data, IterableDataset)
        logging.info(f"Preparing HuggingFace datasets for training (streaming={is_streaming})")
        
        train_dataset = prepare_hf_dataset_for_training(train_data, tokenizer, config)
        eval_dataset = prepare_hf_dataset_for_training(eval_data, tokenizer, config)
        
        if not is_streaming:
            logging.info(f"Created HF datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        else:
            logging.info(f"Created streaming HF datasets (exact counts not available)")
        
        return train_dataset, eval_dataset
    
    # Handle List[str] with LocalSanskritDataset (original behavior)
    else:
        logging.info("Creating LocalSanskritDataset instances")
        
        tokenizer_config = config['tokenizer']
        max_length = tokenizer_config.get('max_length', 256)
        
        train_dataset = LocalSanskritDataset(train_data, tokenizer, max_length)
        eval_dataset = LocalSanskritDataset(eval_data, tokenizer, max_length)
        
        logging.info(f"Created datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset


def create_data_collator(tokenizer, config) -> DataCollatorForLanguageModeling:
    """
    Create data collator for language modeling
    
    Args:
        tokenizer: HuggingFace tokenizer
        config: Configuration object
        
    The collator handles:
        - Padding sequences to same length within batch
        - Creating labels (copy of input_ids with padding masked)
        - Converting to PyTorch tensors

    Returns:
        DataCollatorForLanguageModeling instance
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8,  # pad to multiple of 8 for GPU efficiency
    )
    
    logging.info("Created data collator for causal language modeling")
    logging.info("  Dynamic padding enabled (pads to longest in batch)")
    logging.info("  Labels: auto-generated with padding masked (-100)")
    
    return data_collator


def print_dataset_info(train_dataset: Union[Dataset, HFDataset, IterableDataset], eval_dataset: Union[Dataset, HFDataset, IterableDataset], tokenizer, config, stage: int = 0) -> None:
    """
    Print information about the datasets with EEVE stage awareness
    Supports LocalSanskritDataset, HuggingFace Dataset, and IterableDataset (streaming)
    
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
        if 'eeve' in config and 'stages' in config['eeve']:
            stage_config = config['eeve']['stages'].get(stage, {})
            stage_desc = stage_config.get('description', f'Stage {stage}')
            print(f"Stage: {stage_desc}")
    else:
        print("DATASET INFORMATION")

    print("="*60)
    
    # Detect dataset type
    is_streaming = isinstance(train_dataset, IterableDataset)
    is_hf_dataset = isinstance(train_dataset, (HFDataset, IterableDataset))
    
    if is_hf_dataset:
        if is_streaming:
            dataset_type = "HuggingFace IterableDataset (Streaming)"
            print(f"Dataset type: {dataset_type}")
            print(f"⚠️ Streaming mode: Sample counts not available until iteration")
            print(f"   Use debug.limit_train_samples and limit_eval_samples to control size")
            
            # Show features by peeking at first sample
                # Only peek if verbose logging is enabled
            if config.get('debug', {}).get('verbose_logging', False):
                try:
                    print(f"\nDataset Features (from first sample):")
                    sample_iter = iter(train_dataset)
                    first_sample = next(sample_iter)
                    
                    for key, value in first_sample.items():
                        # Determine type
                        if isinstance(value, torch.Tensor):
                            value_type = f"Tensor(shape={value.shape}, dtype={value.dtype})"
                        elif isinstance(value, list):
                            value_type = f"List[{type(value[0]).__name__}] (length={len(value)})"
                        elif isinstance(value, (int, float, str)):
                            value_type = type(value).__name__
                        else:
                            value_type = str(type(value))
                        
                        print(f"  {key}: {value_type}")
                    
                    # Show a sample of the data
                    print(f"\nFirst Sample Data:")
                    if 'input_ids' in first_sample:
                        input_ids = first_sample['input_ids']
                        if isinstance(input_ids, torch.Tensor):
                            print(f"  input_ids (first 20): {input_ids[:20].tolist()}")
                        else:
                            print(f"  input_ids (first 20): {input_ids[:20]}")
                    
                    if 'attention_mask' in first_sample:
                        att_mask = first_sample['attention_mask']
                        if isinstance(att_mask, torch.Tensor):
                            non_padding = att_mask.sum().item()
                            total = len(att_mask)
                        else:
                            non_padding = sum(att_mask)
                            total = len(att_mask)
                        print(f"  attention_mask: {non_padding}/{total} non-padding tokens ({non_padding/total*100:.1f}%)")
                    
                    if 'labels' in first_sample:
                        labels = first_sample['labels']
                        if isinstance(labels, torch.Tensor):
                            non_ignored = (labels != -100).sum().item()
                            total = len(labels)
                        else:
                            non_ignored = sum(1 for l in labels if l != -100)
                            total = len(labels)
                        print(f"  labels: {non_ignored}/{total} non-ignored tokens ({non_ignored/total*100:.1f}%)")
                    
                    # Try to decode if possible
                    if 'input_ids' in first_sample:
                        try:
                            input_ids = first_sample['input_ids']
                            if isinstance(input_ids, torch.Tensor):
                                input_ids = input_ids.tolist()
                            decoded = tokenizer.decode(input_ids[:50], skip_special_tokens=True)
                            print(f"\nDecoded sample (first 50 tokens):")
                            print(f"  {decoded[:200]}...")
                        except Exception as e:
                            print(f"\nCould not decode sample: {e}")
                        
                except Exception as e:
                    print(f"\nDataset Features: Could not peek at sample - {e}")
            else:
                print(f"\nDataset Features: Enable verbose_logging to see sample inspection")
        else:
            dataset_type = "HuggingFace Dataset"
            print(f"Dataset type: {dataset_type}")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Evaluation samples: {len(eval_dataset)}")
            
            # Show features for non-streaming
            if hasattr(train_dataset, 'features') and train_dataset.features:
                print(f"\nDataset Features:")
                for feature_name, feature_type in train_dataset.features.items():
                    print(f"  {feature_name}: {feature_type}")
            else:
                # Fallback: show from first sample
                try:
                    print(f"\nDataset Features (from first sample):")
                    first_sample = train_dataset[0]
                    for key, value in first_sample.items():
                        if isinstance(value, torch.Tensor):
                            value_type = f"Tensor(shape={value.shape}, dtype={value.dtype})"
                        elif isinstance(value, list):
                            value_type = f"List (length={len(value)})"
                        else:
                            value_type = type(value).__name__
                        print(f"  {key}: {value_type}")
                except Exception as e:
                    print(f"\nDataset Features: Not available - {e}")
        
    else:
        dataset_type = "PyTorch LocalSanskritDataset"
        print(f"Dataset type: {dataset_type}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")
        print(f"Max sequence length: {train_dataset.max_length}")
    
    # Enhanced tokenizer information
    print(f"\nTokenizer Information:")
    print(f"  Current vocabulary size: {len(tokenizer):,}")
    
    if 'vocabulary' in config:
        vocab_config = config['vocabulary']
        if vocab_config.get('use_custom_vocabulary', False):
            print(f"  🔤 Custom vocabulary: ENABLED for the full training")
            str_num_tokens = str(config.get('vocabulary_generation', {}).get('num_tokens', ""))
            vocab_file_path = vocab_config['vocabulary_full_path'] + str_num_tokens +'.pkl'
            print(f"  Vocabulary file: {vocab_file_path}")
            print(f"  Addition method in stage 1: {vocab_config.get('add_tokens_method', 'N/A')}")
        else:
            print(f"  🔤 Custom vocabulary: DISABLED (using original tokenizer)")
    
    # Show EEVE stage information
    if stage > 0:
        print(f"\nEEVE Stage Information:")
        print(f"  Current stage: {stage}")
        
        if 'eeve' in config:
            eeve_config = config['eeve']
            
            if 'stages' in eeve_config:
                stage_info = eeve_config['stages'].get(stage, {})
                if stage_info:
                    print(f"  Training layers: {stage_info.get('train_layers', 'N/A')}")
                    print(f"  Epochs for this stage: {stage_info.get('epochs', 'N/A')}")
                    
                    if 'learning_rate' in stage_info:
                        print(f"  Stage learning rate: {stage_info['learning_rate']}")
            
            start_stage = eeve_config.get('start_stage', 1)
            end_stage = eeve_config.get('end_stage', 7)
            progress = ((stage - start_stage + 1) / (end_stage - start_stage + 1)) * 100
            print(f"  EEVE progress: {stage}/{end_stage} stages ({progress:.0f}%)")
            
            n_added_tokens = eeve_config.get('n_added_tokens', 0)
            if n_added_tokens > 0 and stage <= 3:
                print(f"  Added tokens count: {n_added_tokens:,}")
    
    # Show debug information
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
    if not is_streaming and len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\nSample Training Data:")
        
        if is_hf_dataset:
            # HF Dataset sample
            print(f"  Input IDs length: {len(sample['input_ids'])}")
            print(f"  Attention mask length: {len(sample['attention_mask'])}")
            print(f"  Labels length: {len(sample['labels'])}")
            
            # Decode sample tokens
            input_ids = sample['input_ids']
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            
            # Sample middle portion
            start_idx = min(100, len(input_ids) // 2)
            end_idx = min(start_idx + 50, len(input_ids))
            decoded_text = tokenizer.decode(input_ids[start_idx:end_idx], skip_special_tokens=True)
            print(f"  Decoded text (sampled tokens): {decoded_text}")
            
        else:
            # LocalSanskritDataset sample
            print(f"  Input IDs shape: {sample['input_ids'].shape}")
            print(f"  Attention mask shape: {sample['attention_mask'].shape}")
            print(f"  Labels shape: {sample['labels'].shape}")
            
            decoded_text = tokenizer.decode(sample['input_ids'][100:150], skip_special_tokens=True)
            print(f"  Decoded text (sampled tokens): {decoded_text}")
            
            # Show vocabulary coverage if custom vocabulary is used
            if 'vocabulary' in config:
                vocab_config = config['vocabulary']
                if vocab_config.get('use_custom_vocabulary', False):
                    vocab_size_original = 256000
                    current_vocab_size = len(tokenizer)
                    
                    if current_vocab_size > vocab_size_original:
                        custom_token_ids = sample['input_ids'][sample['input_ids'] >= vocab_size_original]
                        if len(custom_token_ids) > 0:
                            custom_token_count = len(custom_token_ids)
                            total_tokens = len(sample['input_ids'][sample['input_ids'] != tokenizer.pad_token_id])
                            percentage = (custom_token_count / total_tokens) * 100
                            print(f"  Custom vocabulary usage: {custom_token_count}/{total_tokens} tokens ({percentage:.1f}%)")
                        else:
                            print(f"  Custom vocabulary usage: No custom tokens in this sample")
    elif is_streaming:
        print(f"\nSample Training Data:")
        print(f"  ⚠️ Cannot display sample from streaming dataset (no indexing)")
        print(f"  Samples will be processed during training iteration")

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


def load_data_pipeline(tokenizer, config) -> Tuple[Union[Dataset, HFDataset, IterableDataset], Union[Dataset, HFDataset, IterableDataset], DataCollatorForLanguageModeling]:
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