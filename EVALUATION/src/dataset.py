"""
FLORES dataset loading and preparation
"""

from datasets import load_dataset
from src.utils import get_flores_code


def load_flores_dataset(config):
    """
    Load FLORES dataset for source and target languages
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of translation pairs [{'source': str, 'target': str, 'id': int}, ...]
    """
    dataset_config = config['dataset']
    translation_config = config['translation']
    
    # Get FLORES codes
    source_lang_name = translation_config['source_lang']
    target_lang_name = translation_config['target_lang']
    
    source_code = get_flores_code(source_lang_name)
    target_code = get_flores_code(target_lang_name)
    
    print(f"Loading FLORES+ dataset...")
    print(f"  Source: {source_lang_name} ({source_code})")
    print(f"  Target: {target_lang_name} ({target_code})")
    print(f"  Split: {dataset_config['split']}")
    
    # Load source language dataset
    source_dataset = load_dataset(
        dataset_config['name'],
        source_code,
        split=dataset_config['split']
    )
    source_df = source_dataset.to_pandas()
    
    # Load target language dataset
    target_dataset = load_dataset(
        dataset_config['name'],
        target_code,
        split=dataset_config['split']
    )
    target_df = target_dataset.to_pandas()
    
    # Create translation pairs
    pairs = []
    for idx in range(len(source_df)):
        pairs.append({
            'source': source_df.iloc[idx]['text'],
            'target': target_df.iloc[idx]['text'],
            'id': idx
        })
    
    # Limit samples if specified
    num_samples = dataset_config.get('num_samples')
    if num_samples is not None:
        num_samples = int(num_samples) # convert to integer
        pairs = pairs[:num_samples]
        print(f"Limited to {num_samples} samples")
    
    print(f"Loaded {len(pairs)} translation pairs")
    
    # Print first few examples
    print("\nFirst 3 examples:")
    for i, pair in enumerate(pairs[:3]):
        print(f"\n  [{i}] Source: {pair['source']}")
        print(f"      Target: {pair['target']}")
    
    return pairs