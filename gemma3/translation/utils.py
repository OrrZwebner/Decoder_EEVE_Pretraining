#!/usr/bin/env python3
"""
Utility functions for translation fine-tuning

Functions:
    load_config: Load YAML configuration file
    load_nllb_data: Load and format NLLB-200 dataset for translation
    format_translation_example: Format a single translation example with prompt
    get_nllb_code: Get NLLB-200 language code from language name
"""

import yaml
import os
from datasets import load_dataset
from typing import Dict, Tuple, Any


# NLLB-200 Language Code Mapping
# Maps language names to NLLB-200 dataset codes
# Source: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANGUAGE_CODES = {
    # Major Languages
    "English": "eng_Latn",
    "Chinese": "zho_Hans",  # Simplified Chinese
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Russian": "rus_Cyrl",
    "Portuguese": "por_Latn",
    "Italian": "ita_Latn",
    "Dutch": "nld_Latn",
    "Polish": "pol_Latn",
    "Turkish": "tur_Latn",
    "Vietnamese": "vie_Latn",
    "Thai": "tha_Thai",
    "Indonesian": "ind_Latn",
    "Malay": "zsm_Latn",

    # Middle Eastern Languages
    "Arabic": "ara_Arab",
    "Hebrew": "heb_Hebr",
    "Persian": "pes_Arab",
    "Urdu": "urd_Arab",

    # South Asian Languages
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Nepali": "npi_Deva",
    "Sinhala": "sin_Sinh",

    # Southeast Asian Languages
    "Burmese": "mya_Mymr",
    "Khmer": "khm_Khmr",
    "Lao": "lao_Laoo",
    "Tagalog": "tgl_Latn",
    "Cebuano": "ceb_Latn",

    # European Languages
    "Ukrainian": "ukr_Cyrl",
    "Czech": "ces_Latn",
    "Swedish": "swe_Latn",
    "Danish": "dan_Latn",
    "Finnish": "fin_Latn",
    "Norwegian": "nob_Latn",
    "Hungarian": "hun_Latn",
    "Greek": "ell_Grek",
    "Romanian": "ron_Latn",
    "Bulgarian": "bul_Cyrl",
    "Serbian": "srp_Cyrl",
    "Croatian": "hrv_Latn",
    "Slovak": "slk_Latn",
    "Lithuanian": "lit_Latn",
    "Latvian": "lvs_Latn",
    "Estonian": "est_Latn",
    "Slovenian": "slv_Latn",
    "Albanian": "als_Latn",
    "Macedonian": "mkd_Cyrl",
    "Belarusian": "bel_Cyrl",
    "Icelandic": "isl_Latn",
    "Irish": "gle_Latn",
    "Welsh": "cym_Latn",
    "Scots Gaelic": "gla_Latn",
    "Maltese": "mlt_Latn",

    # African Languages
    "Swahili": "swh_Latn",
    "Amharic": "amh_Ethi",
    "Hausa": "hau_Latn",
    "Yoruba": "yor_Latn",
    "Igbo": "ibo_Latn",
    "Zulu": "zul_Latn",
    "Xhosa": "xho_Latn",
    "Somali": "som_Latn",
    "Afrikaans": "afr_Latn",

    # Other Languages
    "Esperanto": "epo_Latn",
    "Latin": "lat_Latn",
    "Sanskrit": "san_Deva",
    "Tibetan": "bod_Tibt",
}


def get_nllb_code(language_name: str) -> str:
    """
    Get NLLB-200 language code from language name

    Args:
        language_name: Human-readable language name (e.g., "English", "Hebrew")

    Returns:
        NLLB-200 language code (e.g., "eng_Latn", "heb_Hebr")

    Raises:
        ValueError: If language name is not found in mapping

    Example:
        >>> get_nllb_code("English")
        'eng_Latn'
        >>> get_nllb_code("Hebrew")
        'heb_Hebr'
    """
    if language_name not in NLLB_LANGUAGE_CODES:
        raise ValueError(
            f"Language '{language_name}' not found in NLLB-200 mapping. "
            f"Available languages: {', '.join(sorted(NLLB_LANGUAGE_CODES.keys()))}"
        )
    return NLLB_LANGUAGE_CODES[language_name]


def load_config(config_path: str = 'configs/train_config.yaml') -> Dict[str, Any]:
    """
    Load YAML configuration file and resolve language codes

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration with resolved language codes

    Example:
        >>> config = load_config('configs/train_config.yaml')
        >>> print(config['model']['base_model'])
        'google/gemma-3-1b-it'
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Auto-resolve language codes from language names
    if 'translation' in config:
        trans_config = config['translation']

        # If source_code/target_code not specified, resolve from language names
        if 'source_code' not in trans_config and 'source_lang' in trans_config:
            trans_config['source_code'] = get_nllb_code(trans_config['source_lang'])

        if 'target_code' not in trans_config and 'target_lang' in trans_config:
            trans_config['target_code'] = get_nllb_code(trans_config['target_lang'])

    return config


def format_translation_example(example: Dict, config: Dict, tokenizer) -> Dict:
    """
    Format a single NLLB-200 example for translation training

    This function:
    1. Extracts source and target text from NLLB-200 format
    2. Creates instruction prompt (e.g., "Translate from English to Hebrew: ...")
    3. Tokenizes input and output separately
    4. Returns formatted example ready for training

    Args:
        example: Single example from NLLB-200 dataset with 'translation' field
        config: Configuration dictionary with translation settings
        tokenizer: HuggingFace tokenizer

    Returns:
        Dictionary with tokenized input_ids, attention_mask, and labels

    Example Input (NLLB-200 format):
        {
            'translation': {
                'eng_Latn': 'Hello, how are you?',
                'heb_Hebr': 'שלום, מה שלומך?'
            }
        }

    Example Output:
        {
            'input_ids': [1, 2345, 234, ...],
            'attention_mask': [1, 1, 1, ...],
            'labels': [5678, 890, ...]
        }
    """
    trans_config = config['translation']

    # Extract source and target text
    source_text = example['translation'][trans_config['source_code']]
    target_text = example['translation'][trans_config['target_code']]

    # Filter out examples that are too long or too short
    if len(source_text) > trans_config['max_source_length'] or len(source_text) < trans_config['min_source_length']:
        return None

    # Create instruction prompt using template
    prompt = trans_config['prompt_template'].format(
        source_lang=trans_config['source_lang'],
        target_lang=trans_config['target_lang'],
        text=source_text
    )

    # Tokenize input (prompt with source text)
    input_encoding = tokenizer(
        prompt,
        max_length=trans_config['max_length'],
        truncation=True,
        padding=False,  # Padding handled by data collator
        return_tensors=None
    )

    # Tokenize output (target translation)
    target_encoding = tokenizer(
        target_text,
        max_length=trans_config['max_length'],
        truncation=True,
        padding=False,
        return_tensors=None
    )

    return {
        'input_ids': input_encoding['input_ids'],
        'attention_mask': input_encoding['attention_mask'],
        'labels': target_encoding['input_ids']
    }


def load_local_jsonl(file_path: str, source_code: str, target_code: str) -> list:
    """
    Load translation pairs from local JSONL file

    Args:
        file_path: Path to JSONL file
        source_code: Source language NLLB code (e.g., 'eng_Latn')
        target_code: Target language NLLB code (e.g., 'heb_Hebr')

    Returns:
        List of translation examples in NLLB format
    """
    import json

    examples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())

            # Check if this matches our language pair
            if data.get('source_code') == source_code and data.get('target_code') == target_code:
                # Convert to NLLB format
                example = {
                    'translation': {
                        source_code: data['source'],
                        target_code: data['target']
                    }
                }
                examples.append(example)

    return examples


def load_nllb_data(tokenizer, config: Dict) -> Tuple:
    """
    Load NLLB-200 dataset and prepare for translation training

    This function supports two modes:
    1. Local JSONL files (fast, no download needed)
    2. HuggingFace streaming (fallback if local files not available)

    Process:
    1. Load data from local file or HuggingFace
    2. Filter by language pair
    3. Shuffle and sample specified number of examples
    4. Split into train and eval sets
    5. Format with translation prompts
    6. Tokenize all examples

    Args:
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary

    Returns:
        Tuple of (train_dataset, eval_dataset)

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        >>> config = load_config('configs/train_config.yaml')
        >>> train_data, eval_data = load_nllb_data(tokenizer, config)
        >>> print(f"Train samples: {len(train_data)}")
        Train samples: 1000
    """
    import random
    from datasets import Dataset

    trans_config = config['translation']
    seed = config['environment']['seed']

    print(f"\n{'='*60}")
    print("Loading NLLB-200 Dataset")
    print(f"{'='*60}")
    print(f"  Language pair: {trans_config['source_code']} → {trans_config['target_code']}")
    print(f"  Train samples: {trans_config['train_samples']:,}")
    print(f"  Eval samples: {trans_config['eval_samples']:,}")

    # Check for local dataset first
    local_dataset_path = trans_config.get('local_dataset_path')
    examples = []

    if local_dataset_path and os.path.exists(local_dataset_path):
        print(f"\n  ✓ Using local dataset: {local_dataset_path}")

        # Load from local JSONL file
        examples = load_local_jsonl(
            local_dataset_path,
            trans_config['source_code'],
            trans_config['target_code']
        )

        print(f"  ✓ Loaded {len(examples):,} examples from local file")

    else:
        # Fallback to HuggingFace streaming
        if local_dataset_path:
            print(f"  ⚠ Local dataset not found: {local_dataset_path}")

        print(f"  Falling back to HuggingFace streaming...")

        dataset_name = trans_config['dataset']
        language_pair = f"{trans_config['source_code']}-{trans_config['target_code']}"

        dataset = load_dataset(
            dataset_name,
            language_pair,
            split='train',
            trust_remote_code=True,
            streaming=True
        )

        # Download all needed examples
        total_needed = trans_config['train_samples'] + trans_config['eval_samples']
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)

        print(f"  Downloading {total_needed:,} examples...")
        for i, example in enumerate(dataset):
            if i >= total_needed:
                break
            examples.append(example)

        print(f"  ✓ Downloaded {len(examples):,} examples")

    # Shuffle examples
    print(f"\n  Shuffling with seed {seed}...")
    random.seed(seed)
    random.shuffle(examples)

    # Split into train and eval
    train_end = trans_config['train_samples']
    eval_end = train_end + trans_config['eval_samples']

    train_examples = examples[:min(train_end, len(examples))]
    eval_examples = examples[train_end:min(eval_end, len(examples))]

    print(f"  ✓ Train set: {len(train_examples):,} examples")
    print(f"  ✓ Eval set: {len(eval_examples):,} examples")

    # Convert to HuggingFace datasets
    print(f"\n  Converting to HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    # Format examples with translation prompts
    print(f"  Formatting examples with prompts...")

    def format_fn(example):
        return format_translation_example(example, config, tokenizer)

    train_dataset = train_dataset.map(
        format_fn,
        remove_columns=train_dataset.column_names,
        desc="Formatting train data"
    )

    eval_dataset = eval_dataset.map(
        format_fn,
        remove_columns=eval_dataset.column_names,
        desc="Formatting eval data"
    )

    # Remove None values (filtered examples)
    train_dataset = train_dataset.filter(lambda x: x['input_ids'] is not None)
    eval_dataset = eval_dataset.filter(lambda x: x['input_ids'] is not None)

    print(f"  ✓ Formatted train set: {len(train_dataset):,} examples")
    print(f"  ✓ Formatted eval set: {len(eval_dataset):,} examples")

    # Print sample
    if len(train_dataset) > 0:
        print(f"\n  Sample training example:")
        print(f"    Input IDs length: {len(train_dataset[0]['input_ids'])}")
        print(f"    Labels length: {len(train_dataset[0]['labels'])}")

        # Decode to show actual text
        sample_input = tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=True)
        sample_label = tokenizer.decode(train_dataset[0]['labels'], skip_special_tokens=True)
        print(f"\n    Input: {sample_input[:100]}...")
        print(f"    Label: {sample_label[:100]}...")

    print(f"{'='*60}\n")

    return train_dataset, eval_dataset
