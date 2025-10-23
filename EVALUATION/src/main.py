#!/usr/bin/env python3
"""
FLORES Translation Evaluation Script

Usage:
  nohup python -m src.main --config config.yaml > logs/eval.log 2>&1 &
  
  With overrides:
  nohup python -m src.main --config config.yaml --num_samples 10 --devices 0,1 > logs/eval.log 2>&1 &
"""

import argparse
import sys
import time
import json
import shutil
from pathlib import Path
import pandas as pd

# Import config FIRST (before torch imports)
from src.config import load_config, validate_config, setup_environment
from src.utils import setup_logging, create_output_directory, get_timestamp


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='FLORES Translation Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --config config.yaml
  python -m src.main --config config.yaml --num_samples 10
  python -m src.main --config config.yaml --devices 0,1 --source_lang English --target_lang French
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (overrides config)'
    )
    
    parser.add_argument(
        '--devices',
        type=str,
        default=None,
        help='CUDA devices, e.g., "0,1" (overrides config)'
    )
    
    parser.add_argument(
        '--source_lang',
        type=str,
        default=None,
        help='Source language (overrides config)'
    )
    
    parser.add_argument(
        '--target_lang',
        type=str,
        default=None,
        help='Target language (overrides config)'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Model name or path (overrides config)'
    )

    return parser.parse_args()


def save_results(data_pairs, predictions, metrics, config, output_dir, logger):
    """
    Save evaluation results to files
    
    Args:
        data_pairs: List of source-target pairs
        predictions: List of predicted translations
        metrics: Dictionary of evaluation metrics
        config: Configuration dictionary
        output_dir: Output directory path
        logger: Logger instance
    """
    # Create results dataframe
    results_df = pd.DataFrame({
        'id': [pair['id'] for pair in data_pairs],
        'source': [pair['source'] for pair in data_pairs],
        'reference': [pair['target'] for pair in data_pairs],
        'prediction': predictions
    })
    
    # Save translations CSV
    csv_path = Path(output_dir) / 'translations.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved translations to: {csv_path}")
    
    # Save metrics JSON
    metrics_path = Path(output_dir) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")
    
    # Save config copy
    config_path = Path(output_dir) / 'config.yaml'
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to: {config_path}")
    
    # Print sample translations
    logger.info("\n" + "="*80)
    logger.info("SAMPLE TRANSLATIONS (First 5)")
    logger.info("="*80)
    for idx in range(min(5, len(results_df))):
        row = results_df.iloc[idx]
        logger.info(f"\n[{row['id']}]")
        logger.info(f"Source:     {row['source']}")
        logger.info(f"Reference:  {row['reference']}")
        logger.info(f"Prediction: {row['prediction']}")


def main():
    """Main execution function"""
    
    # Parse arguments first
    args = parse_args()
    
    # Load config
    try:
        config = load_config(args.config, cli_args=args)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return 1
    
    # Validate config
    try:
        validate_config(config)
    except Exception as e:
        print(f"Invalid configuration: {str(e)}")
        return 1
    
    # Setup environment (CUDA devices, HF paths) - BEFORE torch imports
    setup_environment(config)
    
    # NOW safe to import torch and other modules
    import torch
    from src.models import create_llm
    from src.dataset import load_flores_dataset
    from src.translation import create_translation_chain, translate_batch
    from src.evaluation import calculate_metrics, format_metrics_report
    from src.output_parser import TranslationOutputParser
    
    # Create output directory
    try:
        output_dir = create_output_directory(config)
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return 1
    
    # Setup logging (output to stdout, captured by nohup)
    logger = setup_logging(None)
    
    logger.info("="*80)
    logger.info("FLORES TRANSLATION EVALUATION")
    logger.info("="*80)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    # logger.info(f"Log file: {log_file}")
    
    try:
        # Log configuration
        logger.info("\nConfiguration:")
        logger.info(f"  Provider: {config['provider']['type']}")
        logger.info(f"  Model: {config['provider'][config['provider']['type']]['model_name']}")
        logger.info(f"  Source Language: {config['translation']['source_lang']}")
        logger.info(f"  Target Language: {config['translation']['target_lang']}")
        logger.info(f"  Dataset: {config['dataset']['name']}")
        logger.info(f"  Split: {config['dataset']['split']}")
        if config['dataset']['num_samples']:
            logger.info(f"  Num Samples: {config['dataset']['num_samples']}")
        
        # Check CUDA
        if torch.cuda.is_available():
            logger.info(f"\nCUDA available: True")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        else:
            logger.info(f"\nCUDA available: False (using CPU)")
        
        # Load dataset
        logger.info("\n" + "="*80)
        logger.info("LOADING DATASET")
        logger.info("="*80)
        data_pairs = load_flores_dataset(config)
        logger.info(f"Successfully loaded {len(data_pairs)} translation pairs")
        
        # Load model
        logger.info("\n" + "="*80)
        logger.info("LOADING MODEL")
        logger.info("="*80)
        llm = create_llm(config)
        logger.info("Model loaded successfully")
        
        # Create translation chain
        logger.info("\n" + "="*80)
        logger.info("CREATING TRANSLATION CHAIN")
        logger.info("="*80)
        output_parser = TranslationOutputParser()
        chain = create_translation_chain(llm, config, output_parser)
        logger.info("Translation chain created successfully")
        
    # Run translations
        logger.info("\n" + "="*80)
        logger.info("RUNNING TRANSLATIONS")
        logger.info("="*80)
        start_time = time.time()
        predictions = translate_batch(chain, data_pairs, config, logger=logger)
        translation_time = time.time() - start_time
        logger.info(f"\nTranslation completed in {translation_time:.2f} seconds")
        logger.info(f"Average time per sample: {translation_time/len(data_pairs):.2f} seconds")
        
        # Evaluate
        logger.info("\n" + "="*80)
        logger.info("CALCULATING METRICS")
        logger.info("="*80)
        references = [pair['target'] for pair in data_pairs]
        metrics = calculate_metrics(predictions, references)
        
        # Add timing to metrics
        metrics['translation_time_seconds'] = round(translation_time, 2)
        metrics['avg_time_per_sample_seconds'] = round(translation_time / len(data_pairs), 2)
        
        # Format and log report
        report = format_metrics_report(metrics, config, translation_time)
        logger.info("\n" + report)
        
        # Save results
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)
        save_results(data_pairs, predictions, metrics, config, output_dir, logger)
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)
        logger.info(f"All results saved to: {output_dir}")
        
        return 0  # Success
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("ERROR OCCURRED")
        logger.error("="*80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1  # Error


if __name__ == "__main__":
    sys.exit(main())