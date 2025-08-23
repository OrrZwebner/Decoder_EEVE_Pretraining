#!/usr/bin/env python3
"""
Evaluators Module

This module contains evaluation and visualization components for the Sanskrit
tokenizer comparison system. It handles compression ratio calculations,
tokenization testing, results analysis, and plot generation.

Author: Sanskrit NLP Research
Version: 2.0
"""

import logging
import random
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


class CompressionEvaluator:
    """
    Evaluates compression ratios for tokenizers.
    
    Compression ratio is measured as characters per token - higher values
    indicate better compression (fewer tokens needed for same text).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the compression evaluator.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.sample_size = config.get('output', {}).get('compression_test_samples', 1000)
        
        # Cache for baseline tokenizers to avoid reloading
        self._baseline_tokenizers = {}
    
    def calculate_compression_ratio(self, texts: List[str], tokenizer: Any, 
                                  sample_size: int = None) -> float:
        """
        Calculate compression ratio (characters per token) for given tokenizer.
        
        Higher ratios indicate better compression (fewer tokens needed for same text).
        
        Args:
            texts (List[str]): List of texts to test
            tokenizer (Any): HuggingFace tokenizer instance
            sample_size (int): Number of texts to sample for testing (None uses default)
        
        Returns:
            float: Compression ratio (characters per token, higher is better)
        """
        if sample_size is None:
            sample_size = self.sample_size
        
        # Sample texts if we have more than sample_size
        if len(texts) > sample_size:
            sample_texts = random.sample(texts, sample_size)
        else:
            sample_texts = texts
        
        total_tokens = 0
        total_chars = 0
        
        # Process each text and accumulate statistics
        for text in sample_texts:
            try:
                # Encode without special tokens to get pure content tokenization
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
                total_chars += len(text)
            except Exception:
                # Skip texts that cause encoding errors
                continue
        
        # Calculate compression ratio (avoid division by zero)
        compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
        
        self.logger.debug(f"Compression ratio: {compression_ratio:.3f} chars/token "
                         f"({total_chars} chars, {total_tokens} tokens)")
        
        return compression_ratio
    
    def get_baseline_ratios(self, texts: List[str]) -> Dict[str, float]:
        """
        Get baseline compression ratios for all configured models.
        
        Args:
            texts (List[str]): Texts to evaluate
        
        Returns:
            Dict[str, float]: Map of model names to baseline compression ratios
        """
        baseline_ratios = {}
        
        # Load and evaluate each model's baseline tokenizer
        for model_name, model_config in self.config['models'].items():
            try:
                # Load tokenizer if not cached
                if model_name not in self._baseline_tokenizers:
                    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
                    self._baseline_tokenizers[model_name] = tokenizer
                else:
                    tokenizer = self._baseline_tokenizers[model_name]
                
                # Calculate baseline compression ratio
                ratio = self.calculate_compression_ratio(texts, tokenizer)
                baseline_ratios[model_name] = ratio
                
                self.logger.info(f"Baseline {model_name}: {ratio:.3f} chars/token")
                
            except Exception as e:
                self.logger.error(f"Failed to get baseline for {model_name}: {e}")
                baseline_ratios[model_name] = 0.0
        
        return baseline_ratios


class TokenizationTester:
    """
    Tests tokenization improvements on sample Sanskrit texts.
    
    Provides detailed analysis of how expanded tokenizers perform
    compared to original tokenizers on representative Sanskrit texts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tokenization tester.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Standard Sanskrit test texts for evaluation
        self.test_texts = [
            "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः",
            "स्वाध्यायप्रवचने च तपस्तप्त्वा", 
            "महाभारत गीता उपनिषद्",
            "संस्कृत भाषा देवनागरी",
            "श्रीभगवानुवाच",
            "अर्जुन विषाद योग"
        ]
    
    def test_tokenization(self, original_tokenizer: Any, expanded_tokenizer: Any, 
                         model_name: str, algorithm_name: str) -> Dict[str, Any]:
        """
        Test tokenization improvements on sample Sanskrit texts.
        
        Args:
            original_tokenizer (Any): Original tokenizer
            expanded_tokenizer (Any): Expanded tokenizer with new tokens
            model_name (str): Model name for logging
            algorithm_name (str): Algorithm name for logging
        
        Returns:
            Dict[str, Any]: Dictionary containing test results and statistics
        """
        self.logger.info(f"\n=== TESTING {model_name} + {algorithm_name} ===")
        
        total_orig_tokens = 0
        total_exp_tokens = 0
        improvements = []
        test_results = []
        
        # Test each text and log detailed results
        for i, text in enumerate(self.test_texts[:4]):  # Test first 4 texts
            orig_tokens = original_tokenizer.tokenize(text)
            exp_tokens = expanded_tokenizer.tokenize(text)
            
            total_orig_tokens += len(orig_tokens)
            total_exp_tokens += len(exp_tokens)
            
            improvement = len(orig_tokens) - len(exp_tokens)
            improvements.append(improvement)
            
            status = "✅ IMPROVED" if improvement > 0 else "⚠️ NO CHANGE" if improvement == 0 else "❌ WORSE"
            
            # Log detailed tokenization for each test text
            self.logger.info(f"Text {i+1}: '{text[:40]}{'...' if len(text) > 40 else ''}'")
            self.logger.info(f"  Original ({len(orig_tokens)}): {orig_tokens}")
            self.logger.info(f"  Expanded ({len(exp_tokens)}): {exp_tokens}")
            self.logger.info(f"  Result: {len(orig_tokens)} → {len(exp_tokens)} ({status})")
            
            # Store individual test result
            test_results.append({
                'text': text,
                'original_tokens': len(orig_tokens),
                'expanded_tokens': len(exp_tokens),
                'improvement': improvement,
                'status': status
            })
        
        # Calculate overall improvement statistics
        overall_improvement = total_orig_tokens - total_exp_tokens
        improvement_pct = (overall_improvement / total_orig_tokens) * 100 if total_orig_tokens > 0 else 0
        
        self.logger.info(f"\n📊 SUMMARY ({model_name} + {algorithm_name}):")
        self.logger.info(f"  Total tokens: {total_orig_tokens} → {total_exp_tokens}")
        self.logger.info(f"  Overall improvement: {improvement_pct:+.1f}% ({overall_improvement:+d} tokens)")
        
        # Verify new tokens are being used
        new_token_usage = self._check_new_token_usage(
            original_tokenizer, expanded_tokenizer, self.test_texts[0]
        )
        
        # Return comprehensive test results
        return {
            'individual_tests': test_results,
            'total_original_tokens': total_orig_tokens,
            'total_expanded_tokens': total_exp_tokens,
            'overall_improvement': overall_improvement,
            'improvement_percentage': improvement_pct,
            'new_token_usage': new_token_usage
        }
    
    def _check_new_token_usage(self, original_tokenizer: Any, expanded_tokenizer: Any, 
                              sample_text: str) -> Dict[str, Any]:
        """
        Verify that new tokens are actually being used in tokenization.
        
        Args:
            original_tokenizer (Any): Original tokenizer
            expanded_tokenizer (Any): Expanded tokenizer
            sample_text (str): Text to test
        
        Returns:
            Dict[str, Any]: Information about new token usage
        """
        try:
            # Get token IDs for sample text
            exp_token_ids = expanded_tokenizer.encode(sample_text, add_special_tokens=False)
            orig_vocab_size = len(original_tokenizer.get_vocab())
            
            # Find token IDs that are in the new range (beyond original vocabulary)
            new_token_ids = [tid for tid in exp_token_ids if tid >= orig_vocab_size]
            
            if new_token_ids:
                # Convert new token IDs back to token text
                new_token_texts = expanded_tokenizer.convert_ids_to_tokens(new_token_ids[:3])
                self.logger.info(f"✅ New tokens in use: {new_token_texts}")
                return {
                    'new_tokens_detected': True,
                    'new_token_ids': new_token_ids,
                    'sample_new_tokens': new_token_texts
                }
            else:
                self.logger.warning(f"❌ No new tokens detected in sample")
                return {
                    'new_tokens_detected': False,
                    'new_token_ids': [],
                    'sample_new_tokens': []
                }
                
        except Exception as e:
            self.logger.error(f"Error checking new token usage: {e}")
            return {
                'new_tokens_detected': False,
                'error': str(e)
            }


class Plotter:
    """
    Creates visualizations and plots for tokenizer comparison results.
    
    Generates comprehensive comparison plots showing compression ratios,
    improvements, tokens added, and efficiency metrics across different
    algorithms and models including GPT-2 support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the plotter.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_config = config.get('output', {})
    
    def _get_model_color(self, model_name: str) -> str:
        """
        Get consistent color for each model type.
        
        Args:
            model_name (str): Model name
            
        Returns:
            str: Color string for the model
        """
        model_lower = model_name.lower()
        if "llama" in model_lower:
            return 'blue'
        elif "gpt2" in model_lower or "gpt-2" in model_lower:
            return 'green'  # Green for GPT-2
        elif "gemma" in model_lower:
            return 'red'
        else:
            return 'gray'  # Default for unknown models
    
    def _organize_results_for_plotting(self, results: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Organize results into a format suitable for plotting.
        """
        organized = {
            'algorithms': [],
            'compression_ratios': [],
            'improvement_percentages': [],
            'tokens_added': [],
            'models': []
        }
        
        for result in results:
            # Create algorithm label WITHOUT model prefix for cleaner x-axis
            algorithm_label = result['algorithm'].upper()
            
            organized['algorithms'].append(algorithm_label)
            organized['compression_ratios'].append(result.get('compression_ratio', 0))
            organized['improvement_percentages'].append(result.get('improvement_pct', 0))
            organized['tokens_added'].append(result.get('tokens_added', 0))
            organized['models'].append(result['model'].upper())
        
        return organized

    def create_comparison_plots(self, results: List[Dict[str, Any]], 
                            baseline_ratios: Dict[str, float]) -> None:
        """Create comprehensive comparison plots with GPT-2 support."""
        if not results:
            self.logger.warning("No results to plot")
            return
        
        self.logger.info("Creating comparison plots...")
        
        # Organize results
        organized_results = self._organize_results_for_plotting(results)
        
        if not organized_results['algorithms']:
            self.logger.warning("No organized results for plotting")
            return
        
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        algorithms = organized_results['algorithms']
        models = organized_results['models']
        x_pos = np.arange(len(algorithms))
        
        # Create colors based on model using the new color mapping
        colors = [self._get_model_color(model) for model in models]
        
        # Plot 1: Compression Ratios
        bars1 = ax1.bar(x_pos, organized_results['compression_ratios'], 
                        color=colors, alpha=0.8, width=0.6)
        
        # Add baseline lines with model-specific colors
        for model, baseline in baseline_ratios.items():
            color = self._get_model_color(model)
            ax1.axhline(y=baseline, color=color, linestyle='--', alpha=0.7, 
                    label=f'{model.upper()} Original ({baseline:.3f})')
        
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Compression Ratio (chars/token)')
        ax1.set_title('Compression Ratios by Algorithm')
        ax1.legend()
        
        # Plot 2: Improvement Percentages
        bars2 = ax2.bar(x_pos, organized_results['improvement_percentages'], 
                        color=colors, alpha=0.8, width=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Compression Ratio Improvement')
        
        # Plot 3: Tokens Added
        bars3 = ax3.bar(x_pos, organized_results['tokens_added'], 
                        color=colors, alpha=0.8, width=0.6)
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Tokens Added')
        ax3.set_title('Number of Tokens Added')
        
        # Plot 4: Efficiency
        efficiency = []
        for i in range(len(algorithms)):
            improvement = organized_results['improvement_percentages'][i]
            tokens_added = organized_results['tokens_added'][i]
            eff = improvement / tokens_added if tokens_added > 0 else 0
            efficiency.append(eff)
        
        bars4 = ax4.bar(x_pos, efficiency, color=colors, alpha=0.8, width=0.6)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Improvement % per Token')
        ax4.set_title('Efficiency: Improvement per Token Added')
        
        # Format all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(algorithms, rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Add comprehensive legend for model colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='LLaMA'),
            Patch(facecolor='green', label='GPT-2'),
            Patch(facecolor='red', label='Gemma')
        ]
        fig.legend(legend_elements, ['LLaMA', 'GPT-2', 'Gemma'], 
                   loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        self._save_plots(fig)
        plt.close(fig)
    
    def _save_plots(self, fig) -> None:
        """
        Save plots to file with appropriate path handling.
        
        Args:
            fig: Matplotlib figure to save
        """
        plot_filename = self.output_config.get('plot_filename', 'sanskrit_tokenizer_comparison.png')
        
        # Determine save path
        if not os.path.isabs(plot_filename):
            # If relative path, try to save in logs directory
            log_dir = self.config.get('logging', {}).get('log_dir', 'logs/tokenizers')
            log_dir_path = Path(log_dir)
            if log_dir_path.exists():
                plot_path = log_dir_path / plot_filename
            else:
                plot_path = Path(plot_filename)
        else:
            plot_path = Path(plot_filename)
        
        # Create directory if needed
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save with high quality settings
            fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.info(f"📊 Plots saved to: {plot_path}")
            
            # Save individual plots if requested
            if self.output_config.get('save_individual_plots', False):
                self._save_individual_plots(fig, plot_path)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save plots: {e}")
    
    def _save_individual_plots(self, fig, base_path: Path) -> None:
        """
        Save individual subplot images.
        
        Args:
            fig: Matplotlib figure containing subplots
            base_path (Path): Base path for saving individual plots
        """
        try:
            subplot_names = ['compression_ratios', 'improvements', 'tokens_added', 'efficiency']
            base_name = base_path.stem
            base_dir = base_path.parent
            
            for i, name in enumerate(subplot_names):
                # Create individual figure for each subplot
                individual_fig, individual_ax = plt.subplots(1, 1, figsize=(8, 6))
                
                # Copy subplot content (this is a simplified approach)
                individual_path = base_dir / f"{base_name}_{name}.png"
                
                # Save individual plot
                individual_fig.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(individual_fig)
            
            self.logger.info(f"📊 Individual plots saved in: {base_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save individual plots: {e}")


# Utility functions for result analysis

def calculate_statistical_significance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistical significance of improvements across algorithms.
    
    Args:
        results (List[Dict[str, Any]]): Results from experiments
    
    Returns:
        Dict[str, Any]: Statistical analysis of results
    """
    improvements = [r.get('improvement_pct', 0) for r in results]
    
    if not improvements:
        return {'error': 'No improvements to analyze'}
    
    return {
        'mean_improvement': np.mean(improvements),
        'std_improvement': np.std(improvements),
        'max_improvement': max(improvements),
        'min_improvement': min(improvements),
        'num_positive_improvements': sum(1 for imp in improvements if imp > 0),
        'num_algorithms_tested': len(improvements)
    }


def generate_summary_report(results: List[Dict[str, Any]], baseline_ratios: Dict[str, float]) -> str:
    """
    Generate a text summary report of all results.
    
    Args:
        results (List[Dict[str, Any]]): Results from all experiments
        baseline_ratios (Dict[str, float]): Baseline compression ratios
    
    Returns:
        str: Formatted summary report
    """
    if not results:
        return "No results to summarize."
    
    report_lines = [
        "SANSKRIT TOKENIZER COMPARISON SUMMARY",
        "=" * 50,
        "",
        "BASELINE COMPRESSION RATIOS:",
    ]
    
    # Add baseline ratios
    for model, ratio in baseline_ratios.items():
        report_lines.append(f"  {model.upper()}: {ratio:.3f} chars/token")
    
    report_lines.extend(["", "RESULTS BY ALGORITHM:", ""])
    
    # Add results for each algorithm
    for result in results:
        model = result['model']
        algorithm = result['algorithm']
        compression_ratio = result.get('compression_ratio', 0)
        improvement_pct = result.get('improvement_pct', 0)
        tokens_added = result.get('tokens_added', 0)
        
        report_lines.append(f"{model.upper()} + {algorithm.upper()}:")
        report_lines.append(f"  Compression Ratio: {compression_ratio:.3f} chars/token")
        report_lines.append(f"  Improvement: {improvement_pct:+.1f}%")
        report_lines.append(f"  Tokens Added: {tokens_added}")
        report_lines.append("")
    
    # Add statistical summary
    stats = calculate_statistical_significance(results)
    if 'error' not in stats:
        report_lines.extend([
            "STATISTICAL SUMMARY:",
            f"  Mean Improvement: {stats['mean_improvement']:.1f}%",
            f"  Best Result: {stats['max_improvement']:.1f}%",
            f"  Algorithms with Positive Results: {stats['num_positive_improvements']}/{stats['num_algorithms_tested']}",
        ])
    
    return "\n".join(report_lines)