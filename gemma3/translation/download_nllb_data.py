#!/usr/bin/env python3
"""
Download NLLB-200 Translation Data

Purpose:
    Download 1M translation pairs from NLLB-200 dataset for specified language pairs
    and save as local JSONL files to avoid streaming overhead during training.

Language Pairs:
    - English ↔ Hebrew
    - English ↔ Sanskrit
    - English ↔ Tibetan

Output:
    Saves to: /home/orrz/gpufs/projects/gemma3/data/nllb_translation/
    Format: JSONL files with {"source": "...", "target": "...", "source_lang": "...", "target_lang": "..."}

Usage:
    python download_nllb_data.py [--samples 1000000]
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


# Language pair configurations
# Note: NLLB-200 doesn't have all language pairs
# Available pairs are documented in the dataset card
LANGUAGE_PAIRS = [
    {
        "source_lang": "English",
        "target_lang": "Hebrew",
        "source_code": "eng_Latn",
        "target_code": "heb_Hebr",
        "output_file": "eng_heb.jsonl"
    },
    # Note: eng_Latn-san_Deva (English-Sanskrit) is NOT available in NLLB-200
    # Note: eng_Latn-bod_Tibt (English-Tibetan) is available but in reverse direction
    {
        "source_lang": "Tibetan",
        "target_lang": "English",
        "source_code": "bod_Tibt",
        "target_code": "eng_Latn",
        "output_file": "bod_eng.jsonl"
    },
    # Additional common pairs you might want:
    {
        "source_lang": "English",
        "target_lang": "Arabic",
        "source_code": "eng_Latn",
        "target_code": "arb_Arab",
        "output_file": "eng_arb.jsonl"
    },
]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download NLLB-200 translation data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=1000000,
        help='Number of samples to download per language pair (default: 1M)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/orrz/gpufs/projects/gemma3/data/nllb_translation',
        help='Output directory for downloaded data'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42)'
    )

    return parser.parse_args()


def download_language_pair(pair_config, num_samples, output_dir, seed):
    """
    Download translation data for a single language pair

    Args:
        pair_config: Dictionary with language pair configuration
        num_samples: Number of samples to download
        output_dir: Directory to save output file
        seed: Random seed for shuffling

    Returns:
        Number of samples actually downloaded
    """
    source_lang = pair_config['source_lang']
    target_lang = pair_config['target_lang']
    source_code = pair_config['source_code']
    target_code = pair_config['target_code']
    output_file = pair_config['output_file']

    print(f"\n{'='*70}")
    print(f"Downloading: {source_lang} ↔ {target_lang}")
    print(f"{'='*70}")
    print(f"  NLLB codes: {source_code} → {target_code}")
    print(f"  Target samples: {num_samples:,}")

    # Construct language pair string
    language_pair = f"{source_code}-{target_code}"

    try:
        # Load dataset with streaming
        print(f"\n  Loading dataset (streaming mode)...")
        dataset = load_dataset(
            "allenai/nllb",
            language_pair,
            split='train',
            streaming=True
        )

        # Shuffle dataset
        print(f"  Shuffling dataset with seed {seed}...")
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)

        # Download samples
        print(f"  Downloading {num_samples:,} samples...")
        output_path = os.path.join(output_dir, output_file)

        downloaded_count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, total=num_samples, desc=f"  {source_lang}→{target_lang}"):
                if downloaded_count >= num_samples:
                    break

                # Extract translation pair
                translation = example.get('translation', {})
                source_text = translation.get(source_code, '')
                target_text = translation.get(target_code, '')

                # Skip empty translations
                if not source_text or not target_text:
                    continue

                # Create JSONL entry
                entry = {
                    'source': source_text,
                    'target': target_text,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'source_code': source_code,
                    'target_code': target_code
                }

                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                downloaded_count += 1

        print(f"  ✓ Downloaded {downloaded_count:,} samples")
        print(f"  ✓ Saved to: {output_path}")

        # Calculate file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  ✓ File size: {file_size:.1f} MB")

        return downloaded_count

    except Exception as e:
        print(f"  ✗ Error downloading {source_lang}→{target_lang}: {e}")
        return 0


def create_combined_file(output_dir):
    """
    Create a combined file with all language pairs

    Args:
        output_dir: Directory containing individual JSONL files
    """
    print(f"\n{'='*70}")
    print("Creating Combined Dataset")
    print(f"{'='*70}")

    combined_path = os.path.join(output_dir, 'all_pairs.jsonl')
    total_lines = 0

    with open(combined_path, 'w', encoding='utf-8') as outfile:
        for pair_config in LANGUAGE_PAIRS:
            input_path = os.path.join(output_dir, pair_config['output_file'])

            if not os.path.exists(input_path):
                print(f"  ⚠ Skipping {pair_config['output_file']} (not found)")
                continue

            print(f"  Adding {pair_config['source_lang']}→{pair_config['target_lang']}...")

            with open(input_path, 'r', encoding='utf-8') as infile:
                lines = 0
                for line in infile:
                    outfile.write(line)
                    lines += 1
                    total_lines += 1

                print(f"    Added {lines:,} samples")

    print(f"\n  ✓ Combined file created: {combined_path}")
    print(f"  ✓ Total samples: {total_lines:,}")

    file_size = os.path.getsize(combined_path) / (1024 * 1024)  # MB
    print(f"  ✓ File size: {file_size:.1f} MB")


def main():
    """Main download function"""
    args = parse_args()

    print(f"\n{'='*80}")
    print("NLLB-200 DATA DOWNLOADER")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Samples per language pair: {args.samples:,}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: {args.seed}")
    print(f"  Language pairs: {len(LANGUAGE_PAIRS)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory ready: {output_dir}")

    # Download each language pair
    total_downloaded = 0
    success_count = 0

    for pair_config in LANGUAGE_PAIRS:
        downloaded = download_language_pair(
            pair_config,
            args.samples,
            args.output_dir,
            args.seed
        )

        if downloaded > 0:
            total_downloaded += downloaded
            success_count += 1

    # Create combined file
    if success_count > 0:
        create_combined_file(args.output_dir)

    # Summary
    print(f"\n{'='*80}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"  Language pairs downloaded: {success_count}/{len(LANGUAGE_PAIRS)}")
    print(f"  Total samples: {total_downloaded:,}")
    print(f"  Output directory: {args.output_dir}")

    # Create a summary file
    summary_path = os.path.join(args.output_dir, 'dataset_info.json')
    summary = {
        'total_samples': total_downloaded,
        'language_pairs': success_count,
        'samples_per_pair': args.samples,
        'seed': args.seed,
        'files': [
            {
                'language_pair': f"{p['source_lang']}→{p['target_lang']}",
                'file': p['output_file'],
                'source_code': p['source_code'],
                'target_code': p['target_code']
            }
            for p in LANGUAGE_PAIRS
        ]
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Dataset info saved: {summary_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()