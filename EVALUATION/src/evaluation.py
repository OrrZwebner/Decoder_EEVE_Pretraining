"""
Evaluation metrics calculation
"""

from sacrebleu import CHRF, BLEU
import evaluate


def calculate_metrics(predictions, references):
    """
    Calculate evaluation metrics
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        
    Returns:
        Dictionary with metric scores
    """
    # Initialize metrics
    chrf = CHRF(word_order=2)  # chrF++ with word bi-grams
    bleu = BLEU()
    
    # Calculate scores
    chrf_score = chrf.corpus_score(predictions, [references])
    bleu_score = bleu.corpus_score(predictions, [references])
    
    metrics = {
        'chrF++': round(chrf_score.score, 2),
        'BLEU': round(bleu_score.score, 2),
        'num_samples': len(predictions)
    }
    
    # Calculate BLEURT (optional, skip if not installed)
    try:
        print("Loading BLEURT metric...")
        bleurt = evaluate.load("bleurt", "BLEURT-20")
        bleurt_results = bleurt.compute(predictions=predictions, references=references)
        bleurt_score = bleurt_results['scores']
        bleurt_mean = sum(bleurt_score) / len(bleurt_score)
        metrics['BLEURT'] = round(bleurt_mean, 2)
        print("BLEURT calculated successfully")
    except Exception as e:
        print(f"Warning: BLEURT not available ({str(e)})")
        print("Install with: pip install git+https://github.com/google-research/bleurt.git")
        metrics['BLEURT'] = None
    
    return metrics


def format_metrics_report(metrics, config, translation_time):
    """
    Format metrics into a readable report
    
    Args:
        metrics: Dictionary of metric scores
        config: Configuration dictionary
        translation_time: Time taken for translation in seconds
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("EVALUATION RESULTS")
    report.append("=" * 80)
    report.append(f"Model: {config['provider'][config['provider']['type']]['model_name']}")
    report.append(f"Source Language: {config['translation']['source_lang']}")
    report.append(f"Target Language: {config['translation']['target_lang']}")
    report.append(f"Number of Samples: {metrics['num_samples']}")
    report.append(f"Translation Time: {translation_time:.2f}s")
    report.append(f"Avg Time per Sample: {translation_time/metrics['num_samples']:.2f}s")
    report.append("")
    report.append("Metrics:")
    report.append(f"  chrF++: {metrics['chrF++']:.2f}")
    report.append(f"  BLEU:   {metrics['BLEU']:.2f}")
    if metrics.get('BLEURT') is not None:
        report.append(f"  BLEURT: {metrics['BLEURT']:.2f}")
    else:
        report.append(f"  BLEURT: Not available")
    report.append("=" * 80)
    
    return "\n".join(report)