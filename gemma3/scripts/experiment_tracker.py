#!/usr/bin/env python3
"""
Experiment Tracking System for Gemma-3 Training

This script:
1. Parses training logs to extract key metrics
2. Exports WandB runs to CSV format
3. Maintains a centralized experiment tracking spreadsheet

Usage:
    # Parse a log file and add to tracking CSV
    python experiment_tracker.py --log /path/to/train.log

    # Export from WandB
    python experiment_tracker.py --wandb-project gemma3-sanskrit-pretraining

    # View experiment history
    python experiment_tracker.py --list
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Try to import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. WandB export will not be available.")


class ExperimentTracker:
    """Track and manage training experiments"""

    # CSV columns for experiment tracking
    COLUMNS = [
        'run_id',              # Unique identifier
        'date',                # Date of experiment
        'model',               # Model name (e.g., gemma-3-4b)
        'language',            # Target language (hebrew, sanskrit, etc.)
        'vocab_tokens_added',  # Number of custom tokens added
        'num_train_samples',   # Number of training samples
        'num_eval_samples',    # Number of eval samples
        'total_tokens_trained', # Total number of tokens seen during training
        'learning_rate',       # Learning rate
        'warmup_steps',        # Warmup steps
        'lr_scheduler',        # LR scheduler type
        'grad_clip',           # Gradient clipping value
        'batch_size',          # Effective batch size
        'seq_length',          # Sequence length
        'num_epochs',          # Number of epochs
        'total_steps',         # Total training steps
        'train_loss_start',    # Initial training loss
        'train_loss_final',    # Final training loss
        'eval_loss_start',     # Initial eval loss
        'eval_loss_final',     # Final eval loss
        'perplexity_start',    # Initial perplexity
        'perplexity_final',    # Final perplexity
        'custom_token_usage',  # % of custom tokens used
        'gpu_memory_gb',       # Peak GPU memory usage
        'training_time_hrs',   # Total training time
        'wandb_run_id',        # WandB run ID (if applicable)
        'status',              # completed, failed, running
        'notes',               # Free-form notes
    ]

    def __init__(self, csv_path: str = None):
        if csv_path is None:
            # Default to project root
            csv_path = Path(__file__).parent.parent / "experiments.csv"

        self.csv_path = Path(csv_path)

        # Create CSV if it doesn't exist
        if not self.csv_path.exists():
            self._create_csv()

    def _create_csv(self):
        """Create new experiment tracking CSV"""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()

        print(f"Created experiment tracking CSV: {self.csv_path}")

    def add_experiment(self, experiment_data: Dict) -> None:
        """Add a new experiment to the tracking CSV"""
        # Generate run_id if not provided
        if 'run_id' not in experiment_data:
            experiment_data['run_id'] = self._generate_run_id()

        # Add date if not provided
        if 'date' not in experiment_data:
            experiment_data['date'] = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Ensure all columns exist (fill missing with empty string)
        row = {col: experiment_data.get(col, '') for col in self.COLUMNS}

        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)

        print(f"Added experiment: {row['run_id']}")

    def parse_log_file(self, log_path: str) -> Dict:
        """
        Parse a training log file and extract experiment data

        Args:
            log_path: Path to log file

        Returns:
            Dictionary with extracted experiment data
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        print(f"Parsing log file: {log_path}")

        experiment_data = {
            'run_id': log_path.stem,  # Use log filename as run_id
            'date': datetime.fromtimestamp(log_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
            'status': 'completed',  # Assume completed if log exists
        }

        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()

        # Extract model name
        model_match = re.search(r'Model name: ([\w\-/]+)', log_content)
        if model_match:
            experiment_data['model'] = model_match.group(1)

        # Extract vocabulary info
        vocab_match = re.search(r'Added (\d+) new tokens', log_content)
        if vocab_match:
            experiment_data['vocab_tokens_added'] = vocab_match.group(1)

        # Extract training hyperparameters
        lr_match = re.search(r'learning[_ ]rate[:\s]+([0-9.e\-]+)', log_content, re.IGNORECASE)
        if lr_match:
            # Format learning rate in scientific notation to avoid truncation
            try:
                lr_value = float(lr_match.group(1))
                experiment_data['learning_rate'] = f"{lr_value:.2e}" if lr_value != 0 else "0"
            except ValueError:
                experiment_data['learning_rate'] = lr_match.group(1)

        warmup_match = re.search(r'warmup[_ ]steps[:\s]+(\d+)', log_content, re.IGNORECASE)
        if warmup_match:
            experiment_data['warmup_steps'] = warmup_match.group(1)

        batch_match = re.search(r'Effective batch size[:\s]+(\d+)', log_content, re.IGNORECASE)
        if batch_match:
            experiment_data['batch_size'] = batch_match.group(1)

        # Extract loss values
        # Look for initial loss (first logged loss)
        first_loss_match = re.search(r"'loss':\s+([0-9.]+)", log_content)
        if first_loss_match:
            experiment_data['train_loss_start'] = first_loss_match.group(1)

        # Look for final loss (last logged loss)
        all_losses = re.findall(r"'loss':\s+([0-9.]+)", log_content)
        if all_losses:
            experiment_data['train_loss_final'] = all_losses[-1]

        # Extract eval loss
        all_eval_losses = re.findall(r"'eval_loss':\s+([0-9.]+)", log_content)
        if all_eval_losses:
            experiment_data['eval_loss_start'] = all_eval_losses[0]
            experiment_data['eval_loss_final'] = all_eval_losses[-1]

        # Extract perplexity
        perp_matches = re.findall(r'Perplexity:\s+([0-9.]+)', log_content)
        if perp_matches:
            experiment_data['perplexity_start'] = perp_matches[0]
            experiment_data['perplexity_final'] = perp_matches[-1]

        # Extract custom token usage
        token_usage_match = re.search(r'Custom tokens seen:.*?\(([0-9.]+)%\)', log_content)
        if token_usage_match:
            experiment_data['custom_token_usage'] = token_usage_match.group(1)

        # Extract GPU memory
        gpu_mem_matches = re.findall(r'GPU memory.*?([0-9.]+) GB', log_content)
        if gpu_mem_matches:
            experiment_data['gpu_memory_gb'] = max(float(m) for m in gpu_mem_matches)

        # Extract training steps
        steps_match = re.search(r'max_steps[:\s]+(\d+)', log_content)
        if steps_match:
            experiment_data['total_steps'] = steps_match.group(1)

        # Extract epochs
        epochs_match = re.search(r'num_train_epochs[:\s]+(\d+)', log_content)
        if epochs_match:
            experiment_data['num_epochs'] = epochs_match.group(1)

        # Extract number of training samples
        train_samples_match = re.search(r'limit_train_samples[:\s]+(\d+)', log_content)
        if train_samples_match:
            experiment_data['num_train_samples'] = train_samples_match.group(1)
        else:
            # Try to extract from dataset info
            dataset_match = re.search(r'len\(train_dataset\)[:\s=]+(\d+)', log_content)
            if dataset_match:
                experiment_data['num_train_samples'] = dataset_match.group(1)

        # Extract number of eval samples
        eval_samples_match = re.search(r'limit_eval_samples[:\s]+(\d+)', log_content)
        if eval_samples_match:
            experiment_data['num_eval_samples'] = eval_samples_match.group(1)
        else:
            # Try to extract from dataset info
            eval_dataset_match = re.search(r'len\(eval_dataset\)[:\s=]+(\d+)', log_content)
            if eval_dataset_match:
                experiment_data['num_eval_samples'] = eval_dataset_match.group(1)

        # Extract total tokens trained (from token usage callback)
        total_tokens_match = re.search(r'Total tokens processed:\s+([0-9,]+)', log_content)
        if total_tokens_match:
            experiment_data['total_tokens_trained'] = total_tokens_match.group(1).replace(',', '')

        # Extract sequence length
        seq_len_match = re.search(r'max_length[:\s]+(\d+)', log_content)
        if seq_len_match:
            experiment_data['seq_length'] = seq_len_match.group(1)

        # Extract LR scheduler type
        scheduler_match = re.search(r'lr_scheduler_type[:\s]+["\']?(\w+)["\']?', log_content)
        if scheduler_match:
            experiment_data['lr_scheduler'] = scheduler_match.group(1)

        # Extract gradient clipping
        grad_clip_match = re.search(r'max_grad_norm[:\s]+([0-9.]+)', log_content)
        if grad_clip_match:
            experiment_data['grad_clip'] = grad_clip_match.group(1)

        print(f"Extracted {len([v for v in experiment_data.values() if v])} fields from log")

        return experiment_data

    def export_from_wandb(self, project_name: str, entity: Optional[str] = None) -> List[Dict]:
        """
        Export experiments from WandB project

        Args:
            project_name: WandB project name
            entity: WandB entity (username/team), if None uses default

        Returns:
            List of experiment dictionaries
        """
        if not WANDB_AVAILABLE:
            print("Error: wandb not installed. Run: pip install wandb")
            return []

        print(f"Fetching runs from WandB project: {project_name}")

        api = wandb.Api()

        # Get all runs from project
        if entity:
            runs = api.runs(f"{entity}/{project_name}")
        else:
            runs = api.runs(project_name)

        experiments = []

        for run in runs:
            experiment_data = {
                'run_id': run.name,
                'date': run.created_at,
                'wandb_run_id': run.id,
                'status': run.state,
            }

            # Extract config
            config = run.config
            experiment_data['model'] = config.get('model', {}).get('name', '')

            # Format learning rate in scientific notation
            lr = config.get('training', {}).get('learning_rate', '')
            if lr:
                try:
                    lr_value = float(lr)
                    experiment_data['learning_rate'] = f"{lr_value:.2e}" if lr_value != 0 else "0"
                except (ValueError, TypeError):
                    experiment_data['learning_rate'] = str(lr)
            else:
                experiment_data['learning_rate'] = ''

            experiment_data['warmup_steps'] = str(config.get('training', {}).get('warmup_steps', ''))
            experiment_data['batch_size'] = str(config.get('training', {}).get('per_device_train_batch_size', ''))
            experiment_data['num_epochs'] = str(config.get('training', {}).get('num_train_epochs', ''))

            # Extract summary metrics
            summary = run.summary
            experiment_data['train_loss_final'] = str(summary.get('train/loss', ''))
            experiment_data['eval_loss_final'] = str(summary.get('eval/loss', ''))

            # Calculate perplexity
            if 'eval/loss' in summary:
                try:
                    import math
                    experiment_data['perplexity_final'] = str(math.exp(summary['eval/loss']))
                except:
                    pass

            experiments.append(experiment_data)

        print(f"Fetched {len(experiments)} runs from WandB")

        return experiments

    def list_experiments(self, limit: int = 20) -> None:
        """Print recent experiments from tracking CSV"""
        if not self.csv_path.exists():
            print("No experiments tracked yet.")
            return

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            experiments = list(reader)

        if not experiments:
            print("No experiments tracked yet.")
            return

        # Show most recent first
        experiments = experiments[-limit:][::-1]

        print(f"\nRecent Experiments ({len(experiments)}):")
        print("=" * 120)

        # Print header
        print(f"{'Run ID':<25} {'Date':<18} {'Model':<20} {'LR':<10} {'Final Loss':<12} {'Perp':<8} {'Status':<10}")
        print("-" * 120)

        # Print each experiment
        for exp in experiments:
            print(f"{exp.get('run_id', ''):<25} "
                  f"{exp.get('date', ''):<18} "
                  f"{exp.get('model', '')[:19]:<20} "
                  f"{exp.get('learning_rate', ''):<10} "
                  f"{exp.get('train_loss_final', ''):<12} "
                  f"{exp.get('perplexity_final', ''):<8} "
                  f"{exp.get('status', ''):<10}")

        print("=" * 120)

    def _generate_run_id(self) -> str:
        """Generate a unique run ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"run_{timestamp}"


def main():
    parser = argparse.ArgumentParser(description='Experiment Tracking System')
    parser.add_argument('--log', type=str, help='Path to training log file to parse')
    parser.add_argument('--wandb-project', type=str, help='WandB project name to export from')
    parser.add_argument('--wandb-entity', type=str, help='WandB entity (username/team)')
    parser.add_argument('--csv', type=str, help='Path to experiment tracking CSV (default: experiments.csv)')
    parser.add_argument('--list', action='store_true', help='List recent experiments')
    parser.add_argument('--limit', type=int, default=20, help='Number of experiments to list (default: 20)')

    args = parser.parse_args()

    # Initialize tracker
    tracker = ExperimentTracker(csv_path=args.csv)

    if args.list:
        # List experiments
        tracker.list_experiments(limit=args.limit)

    elif args.log:
        # Parse log file and add to tracker
        experiment_data = tracker.parse_log_file(args.log)
        tracker.add_experiment(experiment_data)
        print(f"\n✅ Experiment added to {tracker.csv_path}")

    elif args.wandb_project:
        # Export from WandB
        experiments = tracker.export_from_wandb(args.wandb_project, entity=args.wandb_entity)

        # Add all to tracker
        for exp in experiments:
            tracker.add_experiment(exp)

        print(f"\n✅ Added {len(experiments)} experiments to {tracker.csv_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()