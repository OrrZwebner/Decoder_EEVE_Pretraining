#!/usr/bin/env python3
"""
Sanskrit Tokenizer Comparison Tool - Main Orchestrator

This is the main entry point for the Sanskrit tokenizer comparison system.
It handles configuration loading, data preparation, and orchestrates the entire
comparison pipeline across different tokenization algorithms and models.

Author: Sanskrit NLP Research
Version: 2.0
"""

import os
import sys
import yaml
import argparse
import random
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime

# Import our custom modules
from tokenizer_trainers import train_with_target_size
from model_processors import get_processor
from evaluators import CompressionEvaluator, TokenizationTester, Plotter
from log_score_evaluator import LogScoreEvaluator



def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup comprehensive logging for the application.
    
    Creates both file and console loggers with timestamps.
    File logs go to timestamped files in log_dir.
    
    Args:
        log_dir (str): Directory path where log files will be stored
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sanskrit_tokenizer_{timestamp}.log"
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_sanskrit_data(data_path: str, debug: int, seed=42) -> List[str]:
    """
    Load Sanskrit data from a specified absolute path.
    
    Supports reading .jsonl, .txt, and .pkl files from either a single
    file or a directory. Assumes `data_path` is an absolute path.
    
    Args:
        data_path (str): The absolute path to the data file or directory.
        debug (int): If > 0, sample a debug subset of texts.
        seed (int): Random seed for reproducibility.
    
    Returns:
        List[str]: A list of Sanskrit text strings.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading Sanskrit data from absolute path: {data_path}")
    
    # SIMPLIFIED: We now assume an absolute path is provided.
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist")
    
    texts = []
    
    # CORRECTED: This is the full, correct logic for handling files and directories.
    if data_path.is_file():
        if data_path.suffix == '.pkl': # Handle pickle files
            logger.info(f"Loading pickle file: {data_path}")
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                texts = data if isinstance(data, list) else [str(data)]
        elif data_path.suffix == '.txt':
            logger.info(f"Loading text file: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        elif data_path.suffix == '.jsonl':
            logger.info(f"Loading JSONL file: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            text = data.get('text', data.get('content', data.get('sentence', str(data))))
                            if text and len(str(text).strip()) > 0:
                                texts.append(str(text).strip())
                        except json.JSONDecodeError:
                            continue
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    elif data_path.is_dir():
        logger.info(f"Loading all supported files from directory: {data_path}")
        # Process all supported files in the directory
        for file_path in data_path.glob("*.pkl"):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list): texts.extend(data)
                else: texts.append(str(data))
        
        for file_path in data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.extend([line.strip() for line in f if line.strip()])
        
        for file_path in data_path.glob("*.jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            text = data.get('text', data.get('content', data.get('sentence', str(data))))
                            if text and len(str(text).strip()) > 0:
                                texts.append(str(text).strip())
                        except json.JSONDecodeError:
                            continue

    if not texts:
        raise ValueError(f"No texts found in {data_path}")
    
    logger.info(f"Loaded {len(texts):,} from {data_path}")
    # Final filtering and sampling logic
    texts = [str(text).strip() for text in texts if text and len(str(text).strip()) > 0]

    if debug > 0 and len(texts) > debug:
        logger.info(f"Debug mode enabled: sampling {debug} texts for debugging")
        random.seed(seed)
        texts = random.sample(texts, debug)
    
    logger.info(f"✅ Loaded {len(texts):,} Sanskrit texts from {data_path}")
    return texts

def prepare_training_data(texts: List[str], num_samples: Union[int, None]) -> List[str]:
    """
    Prepare training data based on sample size configuration.
    
    Args:
        texts (List[str]): Full list of available texts
        num_samples (Union[int, None]): Number of samples to use (None/-1 for all)
    
    Returns:
        List[str]: Selected texts for training
    """
    # Use full dataset if num_samples is None, -1, or larger than available data
    if num_samples is None or num_samples == -1 or num_samples >= len(texts):
        logging.info(f"Using FULL dataset: {len(texts)} texts")
        return texts
    else:
        # Sample the specified number of texts randomly
        sampled = random.sample(texts, min(num_samples, len(texts)))
        logging.info(f"Using SAMPLED dataset: {len(sampled)} texts (from {len(texts)} total)")
        return sampled


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_summary(results: List[Dict[str, Any]], logger: logging.Logger) -> None:
    """Print comprehensive summary table of all results."""
    logger.info(f"\n{'='*120}") # Increased width
    logger.info("FINAL RESULTS SUMMARY")
    logger.info(f"{'='*120}") # Increased width
    # ADDED 'LOG-SCORE' TO THE HEADER
    logger.info(f"{'Algorithm':<20} {'Model':<8} {'Ratio':<8} {'Improvement%':<12} {'Log-Score':<15} {'Tokens Added':<12} {'Efficiency':<10}")
    logger.info("-" * 120) # Increased width
    
    for result in results:
        model = result['model']
        algorithm = result['algorithm']
        compression_ratio = result.get('compression_ratio', 0)
        improvement_pct = result.get('improvement_pct', 0)
        tokens_added = result.get('tokens_added', 0)
        # GET THE NEW SCORE FROM RESULTS
        log_score = result.get('log_score', 0.0)
        
        efficiency = improvement_pct / tokens_added if tokens_added > 0 else 0
        
        # ADDED LOG-SCORE TO THE OUTPUT ROW
        logger.info(f"{algorithm:<20} {model:<8} {compression_ratio:<8.3f} "
                   f"{improvement_pct:<12.1f} {log_score:<15.4f} {tokens_added:<12} {efficiency:<10.3f}")


def run_comparison_pipeline(config: Dict[str, Any], texts: List[str], logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Run the complete tokenizer comparison pipeline.
    
    This is the main orchestration method that coordinates all components
    to compare different tokenization algorithms across models.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        texts (List[str]): Sanskrit texts for training and evaluation
        logger (logging.Logger): Logger instance
    
    Returns:
        List[Dict[str, Any]]: List of results for each model-algorithm combination
    """
    logger.info("🚀 Starting Sanskrit tokenizer comparison...")
    logger.info(f"📚 Total texts available: {len(texts)}")
    
    # Set random seed for reproducibility if specified
    if 'random_seed' in config:
        random.seed(config['random_seed'])
        logger.info(f"Set random seed to {config['random_seed']}")
    
    # CORRECTED: Added initialization of evaluators and other setup steps
    # Step 1: Prepare training data based on configuration
    training_texts = prepare_training_data(texts, config['training']['num_samples'])
    
    # Step 2: Initialize evaluation components
    compression_evaluator = CompressionEvaluator(config)
    tokenization_tester = TokenizationTester(config)
    plotter = Plotter(config)
    
    # Step 3: Get baseline compression ratios for comparison
    logger.info("📊 Calculating baseline compression ratios...")
    baseline_ratios = compression_evaluator.get_baseline_ratios(texts)
    
    logger.info(f"\n📋 Baseline Compression Ratios:")
    for model_name, ratio in baseline_ratios.items():
        if "llama" in model_name.lower(): emoji = "🔵"
        elif "gpt2" in model_name.lower() or "gpt-2" in model_name.lower(): emoji = "🟢"
        else: emoji = "🔴"
        logger.info(f"{emoji} {model_name}: {ratio:.3f} chars/token")
    
    # Step 4: Run experiments for each model and algorithm combination
    results = []
    
    # Process each model configuration
    for model_name, model_config in config['models'].items():
        logger.info(f"\n🔄 Processing {model_name.upper()} model...")
        
        processor = get_processor(model_name, model_config)
        
        # Test each algorithm for this model
        for algorithm in model_config['algorithms']:
            logger.info(f"\n{'='*60}")
            if "llama" in model_name.lower(): emoji = "🔵"
            elif "gpt2" in model_name.lower() or "gpt-2" in model_name.lower(): emoji = "🟢"
            else: emoji = "🔴"
            logger.info(f"{emoji} TESTING {algorithm.upper()} FOR {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                # Step 4a: Train tokenizer (for BPE, this just returns the corpus)
                sanskrit_tokens_or_corpus, merges, success = train_with_target_size(
                    texts=training_texts, algorithm=algorithm, 
                    target_size=config['training']['num_new_tokens'], model_name=model_name.upper(),
                    model_config=model_config,
                    unigram_max_iterations=config['training'].get('unigram_max_iterations', 10),
                )
                
                if not success:
                    logger.warning(f"⚠️ Skipping {algorithm} due to training failure")
                    continue
                
                # Step 4b: Expand model tokenizer
                # For BPE, sanskrit_tokens_or_corpus is actually the corpus
                # For other algorithms, it's the learned tokens
                if algorithm.upper() == "BPE":
                    # For BPE, pass corpus for training in expand_tokenizer
                    expanded_tokenizer, tokens_added, processed_tokens = processor.expand_tokenizer(
                        [], algorithm.upper(), config['training']['num_new_tokens'], 
                        training_corpus=sanskrit_tokens_or_corpus
                    )
                else:
                    # For non-BPE, pass learned tokens
                    expanded_tokenizer, tokens_added, processed_tokens = processor.expand_tokenizer(
                        sanskrit_tokens_or_corpus, algorithm.upper(), config['training']['num_new_tokens']
                    )
                
                # --- START OF NEW LOG-SCORE CALCULATION ---
                log_score = 0.0
                if tokens_added > 0:
                    logger.info("Calculating log-score...")
                    log_score_evaluator = LogScoreEvaluator(
                        original_tokenizer=processor.original_tokenizer,
                        expanded_tokenizer=expanded_tokenizer,
                        new_tokens=processed_tokens
                    )
                    log_score = log_score_evaluator.calculate_score(texts)
                else:
                    logger.info("Skipping log-score calculation as no tokens were added.")
                # --- END OF NEW LOG-SCORE CALCULATION ---
                
                # Step 4c: Evaluate compression ratio
                new_ratio = compression_evaluator.calculate_compression_ratio(texts, expanded_tokenizer)
                baseline_ratio = baseline_ratios[model_name]
                # CORRECTED: Improvement formula (higher ratio is better)
                improvement_pct = ((new_ratio - baseline_ratio) / baseline_ratio) * 100 if baseline_ratio > 0 else 0
                
                logger.info(f"{emoji} {model_name} + {algorithm.upper()}: {new_ratio:.3f} chars/token ({improvement_pct:+.1f}%)")
                
                # Step 4d: Test tokenization examples
                test_results = tokenization_tester.test_tokenization(
                    processor.original_tokenizer, expanded_tokenizer, model_name, algorithm.upper()
                )
                
                # Step 4e: Store results
                results.append({
                    'model': model_name,
                    'algorithm': algorithm,
                    'compression_ratio': new_ratio,
                    'baseline_ratio': baseline_ratio,
                    'improvement_pct': improvement_pct,
                    'log_score': log_score,
                    'tokens_added': tokens_added,
                    'test_results': test_results
                })
                
            except Exception as e:
                logger.error(f"❌ Error processing {algorithm} for {model_name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
    
    # Step 5: Generate visualizations if requested
    if config['output']['create_plots']:
        logger.info("\n📊 Creating comparison plots...")
        plotter.create_comparison_plots(results, baseline_ratios)
    
    # Step 6: Print final summary
    print_summary(results, logger)
    logger.info("\n🎉 Sanskrit tokenizer comparison completed!")
    
    return results

def save_vocabulary_to_pickle(vocabulary: List[str], output_path: str, logger: logging.Logger):
    """Saves a list of vocabulary tokens to a .pkl file."""
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(vocabulary, f)
        logger.info(f"✅ Vocabulary with {len(vocabulary)} tokens saved to: {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save vocabulary to {output_path}: {e}")
        

def main() -> int:
    """
    Main function that orchestrates the entire tokenizer comparison process.
    
    Handles command line arguments, configuration loading, data loading,
    and error handling for the complete pipeline. Can run in two modes:
    1. Comparison Mode (default): Runs the full comparison pipeline.
    2. Generation Mode: Generates a single vocabulary file for a specific model.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Sanskrit Tokenizer Comparison and Generation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Run full comparison
    python main.py --config custom_config.yaml

    # Generate a single vocabulary file for Gemma
    python main.py --config custom_config.yaml \\
                    --generate-for-model gemma \\
                    --with-algorithm sentencepiece_bpe \\
                    --output-file /path/to/save/gemma_vocab.pkl
            """
    )
    parser.add_argument('--config', type=str, default='sanskrit_config.yaml', 
                       help='Path to configuration file (default: sanskrit_config.yaml)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--use-existing', action='store_true',
                    help='Use existing vocabulary file instead of generating new one')
    
    # NEW: Arguments for single vocabulary generation mode
    parser.add_argument('--generate-for-model', type=str, default=None,
                        help='Run in generation mode for a specific model (e.g., gemma)')
    parser.add_argument('--with-algorithm', type=str, default=None,
                        help='Specify the algorithm for generation mode (e.g., sentencepiece_bpe)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output path for the generated vocabulary .pkl file (overrides config)')
    
    args = parser.parse_args()
    
    # Find and load configuration file
    script_dir = Path(__file__).parent
    config_paths = [
        script_dir / args.config, Path(args.config), script_dir.parent / args.config
    ]
    config_path = next((path for path in config_paths if path.exists()), None)
    
    if config_path is None:
        print(f"❌ Configuration file '{args.config}' not found.")
        return 1
    
    try:
        config = load_config(config_path)
        print(f"✅ Loaded configuration from {config_path}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Setup logging
    log_dir = config.get('logging', {}).get('log_dir', 'logs/tokenizers')
    logger = setup_logging(log_dir, args.log_level)
    
    # Load Sanskrit data
    data_path = config.get('data', {}).get('path')
    debug = config.get('data', {}).get('debug', 0)
    if not data_path:
        logger.error("❌ No data path specified in config file under 'data.path'")
        return 1
    
    try:
        texts = load_sanskrit_data(data_path, debug, seed=config.get('random_seed', 42))
    except Exception as e:
        logger.error(f"❌ Failed to load Sanskrit data: {e}")
        return 1

    # NEW: Main logic block to switch between modes
    if args.generate_for_model and args.with_algorithm:
            # Check if we should use existing vocabulary
            if args.use_existing:
                if args.output_file:
                    output_file = args.output_file
                else:
                    vocab_filename = config.get('vocabulary_generation', {}).get('vocabulary_file', 'sanskrit_custom_tokens.pkl')
                    vocabularies_dir = "/home/orrz/gpufs/projects/Tokenizers/vocabularies"
                    output_file = os.path.join(vocabularies_dir, vocab_filename)
                
                if os.path.exists(output_file):
                    logger.info(f"📄 Using existing vocabulary file: {output_file}")
                    logger.info("🎉 Vocabulary loading completed successfully!")
                    return 0
                else:
                    logger.warning(f"⚠️ Existing vocabulary file not found: {output_file}")
                    logger.info("🔄 Falling back to vocabulary generation...")
            
            # --- GENERATION MODE ---
            logger.info("🚀 Running in single-file vocabulary GENERATION MODE...")
            model_name = args.generate_for_model # e.g., 'gemma'
            algorithm = args.with_algorithm # e.g., 'sentencepiece_bpe'
            
            if model_name not in config['models']:
                logger.error(f"Model '{model_name}' not found in the 'models' section of the config file.")
                return 1
                
            try:
                model_config = config['models'][model_name]
                training_texts = prepare_training_data(texts, config['training']['num_samples'])
            
                # Determine output file path
                if args.output_file:
                    output_file = args.output_file  # Use command line argument
                else:
                    # Use config to determine filename
                    vocab_filename = config.get('vocabulary_generation', {}).get('vocabulary_file', 'sanskrit_custom_tokens.pkl')
                    vocabularies_dir = "/home/orrz/gpufs/projects/Tokenizers/vocabularies"
                    output_file = os.path.join(vocabularies_dir, vocab_filename)
                
                # Step 1: Train to get the raw Sanskrit tokens
                raw_tokens_or_corpus, merges, success = train_with_target_size(
                    texts=training_texts,
                    algorithm=algorithm,
                    target_size=config['training']['num_new_tokens'],
                    model_name=model_name.upper(),
                    model_config=model_config,
                    unigram_max_iterations=config['training'].get('unigram_max_iterations', 10),
                )
                
                if not success:
                    logger.error("❌ Vocabulary training failed during generation.")
                    return 1
                


                processor = get_processor(model_name, model_config)
                
                # Step 2: Expand model tokenizer
                # For BPE, raw_tokens_or_corpus is the corpus
                # For other algorithms, it's the learned tokens
                if algorithm.upper() == "BPE":
                    # For BPE, training happens in expand_tokenizer
                    expanded_tokenizer, tokens_added, processed_tokens = processor.expand_tokenizer(
                        [], algorithm.upper(), config['training']['num_new_tokens'],
                        training_corpus=raw_tokens_or_corpus
                    )
                else:
                    # For non-BPE, process tokens first
                    logger.info(f"Processing {len(raw_tokens_or_corpus)} raw tokens for model '{model_name}'...")
                    processed_tokens = processor.process_tokens_for_model(raw_tokens_or_corpus, algorithm)
                    
                    # Step 2b: Expand model tokenizer for metrics calculation
                    expanded_tokenizer, tokens_added, _ = processor.expand_tokenizer(
                        raw_tokens_or_corpus, algorithm.upper(), config['training']['num_new_tokens']
                    )
                
                # Step 2c: Initialize evaluation components
                compression_evaluator = CompressionEvaluator(config)
                
                # Step 2d: Calculate baseline compression ratio
                logger.info("📊 Calculating baseline compression ratio...")
                baseline_ratios = compression_evaluator.get_baseline_ratios(texts)
                baseline_ratio = baseline_ratios[model_name]
                
                # Step 2e: Calculate new compression ratio
                new_ratio = compression_evaluator.calculate_compression_ratio(texts, expanded_tokenizer)
                improvement_pct = ((new_ratio - baseline_ratio) / baseline_ratio) * 100 if baseline_ratio > 0 else 0
                
                # Step 2f: Calculate log-score
                log_score = 0.0
                if tokens_added > 0:
                    logger.info("Calculating log-score...")
                    log_score_evaluator = LogScoreEvaluator(
                        original_tokenizer=processor.original_tokenizer,
                        expanded_tokenizer=expanded_tokenizer,
                        new_tokens=processed_tokens
                    )
                    log_score = log_score_evaluator.calculate_score(texts)
                else:
                    logger.info("Skipping log-score calculation as no tokens were added.")
                
                # Step 2g: Log metrics
                if "llama" in model_name.lower(): emoji = "🔵"
                elif "gpt2" in model_name.lower() or "gpt-2" in model_name.lower(): emoji = "🟢"
                else: emoji = "🔴"
                
                logger.info(f"\n📊 GENERATION MODE METRICS:")
                logger.info(f"{emoji} Model: {model_name}")
                logger.info(f"{emoji} Algorithm: {algorithm}")
                logger.info(f"{emoji} Baseline ratio: {baseline_ratio:.3f} chars/token")
                logger.info(f"{emoji} New ratio: {new_ratio:.3f} chars/token")
                logger.info(f"{emoji} Improvement: {improvement_pct:+.1f}%")
                logger.info(f"{emoji} Log-score: {log_score:.4f}")
                logger.info(f"{emoji} Tokens added: {tokens_added}")
                
                # Step 3: Save the final, processed tokens to a pickle file
                save_vocabulary_to_pickle(processed_tokens, output_file, logger)
                logger.info("🎉 Vocabulary generation completed successfully!")
                return 0

            except Exception as e:
                logger.error(f"❌ Vocabulary generation failed: {e}", exc_info=True)
                return 1

    else:
        # --- COMPARISON MODE (Original Behavior) ---
        logger.info("🚀 Running in full tokenization COMPARISON MODE...")
        try:
            run_comparison_pipeline(config, texts, logger)
            logger.info("🎉 Comparison completed successfully!")
            return 0
            
        except Exception as e:
            logger.error(f"❌ Comparison pipeline failed: {e}", exc_info=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())