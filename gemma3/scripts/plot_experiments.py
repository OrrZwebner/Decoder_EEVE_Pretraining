#!/usr/bin/env python3
"""
Loss Plotting and Visualization for Training Experiments

This script:
1. Extracts loss curves from training logs
2. Plots train/eval loss over time
3. Compares multiple experiments
4. Saves plots to plots/ directory

Usage:
    # Plot losses from a single log file
    python plot_experiments.py --log logs/train.log

    # Plot all logs in a directory
    python plot_experiments.py --log-dir logs/pre_train_4b/

    # Compare multiple runs
    python plot_experiments.py --compare run1.log run2.log run3.log

    # Plot from CSV tracker
    python plot_experiments.py --from-csv experiments.csv --run-ids run1,run2,run3
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
import csv

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Run: pip install matplotlib")


class LossPlotter:
    """Extract and plot loss curves from training logs"""

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting. Run: pip install matplotlib")

    def parse_loss_from_log(self, log_path: str) -> Tuple[List[int], List[float], List[float]]:
        """
        Extract step numbers and loss values from log file

        Returns:
            Tuple of (steps, train_losses, eval_losses)
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        print(f"Parsing losses from: {log_path}")

        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()

        # Extract training loss with steps
        train_pattern = r"'loss':\s+([0-9.]+).*?'step':\s+(\d+)"
        train_matches = re.findall(train_pattern, log_content)

        # Also try alternate format: Step N - Loss: X.XX
        alt_train_pattern = r'Step\s+(\d+).*?Loss:\s+([0-9.]+)'
        alt_train_matches = re.findall(alt_train_pattern, log_content)

        # Extract eval loss with steps
        eval_pattern = r"'eval_loss':\s+([0-9.]+).*?'step':\s+(\d+)"
        eval_matches = re.findall(eval_pattern, log_content)

        # Also try alternate format: Step N - Eval Loss: X.XX
        alt_eval_pattern = r'Step\s+(\d+).*?Eval Loss:\s+([0-9.]+)'
        alt_eval_matches = re.findall(alt_eval_pattern, log_content)

        # Combine and sort by step
        train_data = {}
        for loss, step in train_matches:
            train_data[int(step)] = float(loss)
        for step, loss in alt_train_matches:
            if int(step) not in train_data:
                train_data[int(step)] = float(loss)

        eval_data = {}
        for loss, step in eval_matches:
            eval_data[int(step)] = float(loss)
        for step, loss in alt_eval_matches:
            if int(step) not in eval_data:
                eval_data[int(step)] = float(loss)

        # Convert to sorted lists
        train_steps = sorted(train_data.keys())
        train_losses = [train_data[s] for s in train_steps]

        eval_steps = sorted(eval_data.keys())
        eval_losses = [eval_data[s] for s in eval_steps]

        print(f"  Found {len(train_losses)} training loss points")
        print(f"  Found {len(eval_losses)} eval loss points")

        return train_steps, train_losses, eval_steps, eval_losses

    def plot_single_experiment(self, log_path: str, title: str = None, save_name: str = None):
        """
        Plot train and eval loss for a single experiment

        Args:
            log_path: Path to log file
            title: Plot title (defaults to log filename)
            save_name: Output filename (defaults to log stem + .png)
        """
        log_path = Path(log_path)
        train_steps, train_losses, eval_steps, eval_losses = self.parse_loss_from_log(log_path)

        if not train_losses and not eval_losses:
            print(f"Warning: No loss data found in {log_path}")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot train loss
        if train_losses:
            ax.plot(train_steps, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)

        # Plot eval loss
        if eval_losses:
            ax.plot(eval_steps, eval_losses, 'r-', label='Eval Loss', linewidth=2, alpha=0.7)

        # Formatting
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title or f"Training Loss - {log_path.stem}", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add final loss values to legend
        if train_losses:
            final_train = train_losses[-1]
            ax.text(0.02, 0.98, f'Final Train Loss: {final_train:.4f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if eval_losses:
            final_eval = eval_losses[-1]
            ax.text(0.02, 0.90, f'Final Eval Loss: {final_eval:.4f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save plot
        if save_name is None:
            save_name = f"{log_path.stem}_loss.png"

        output_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Plot saved to: {output_path}")

    def plot_comparison(self, log_paths: List[str], labels: List[str] = None,
                       title: str = "Loss Comparison", save_name: str = "comparison.png"):
        """
        Plot multiple experiments on the same graph for comparison

        Args:
            log_paths: List of log file paths
            labels: List of labels for each experiment (defaults to filenames)
            title: Plot title
            save_name: Output filename
        """
        if labels is None:
            labels = [Path(p).stem for p in log_paths]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        colors = plt.cm.tab10(range(len(log_paths)))

        for i, (log_path, label) in enumerate(zip(log_paths, labels)):
            try:
                train_steps, train_losses, eval_steps, eval_losses = self.parse_loss_from_log(log_path)

                # Plot train loss
                if train_losses:
                    ax1.plot(train_steps, train_losses, color=colors[i],
                            label=label, linewidth=2, alpha=0.7)

                # Plot eval loss
                if eval_losses:
                    ax2.plot(eval_steps, eval_losses, color=colors[i],
                            label=label, linewidth=2, alpha=0.7)

            except Exception as e:
                print(f"Warning: Failed to plot {log_path}: {e}")

        # Format train loss subplot
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Format eval loss subplot
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Evaluation Loss', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Save plot
        output_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Comparison plot saved to: {output_path}")

    def plot_from_directory(self, log_dir: str, pattern: str = "*.log"):
        """
        Plot all logs in a directory

        Args:
            log_dir: Directory containing log files
            pattern: Glob pattern for log files
        """
        log_dir = Path(log_dir)
        log_files = sorted(log_dir.glob(pattern))

        if not log_files:
            print(f"No log files found in {log_dir} matching pattern '{pattern}'")
            return

        print(f"Found {len(log_files)} log files")

        # Plot each individually
        for log_file in log_files:
            try:
                self.plot_single_experiment(log_file)
            except Exception as e:
                print(f"Warning: Failed to plot {log_file}: {e}")

        # Also create a comparison plot
        if len(log_files) > 1 and len(log_files) <= 10:  # Don't compare too many
            try:
                self.plot_comparison(
                    [str(f) for f in log_files],
                    title=f"All Experiments in {log_dir.name}",
                    save_name=f"{log_dir.name}_comparison.png"
                )
            except Exception as e:
                print(f"Warning: Failed to create comparison plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Plot Training Losses')
    parser.add_argument('--log', type=str, help='Single log file to plot')
    parser.add_argument('--log-dir', type=str, help='Directory containing multiple logs')
    parser.add_argument('--compare', nargs='+', help='Multiple log files to compare')
    parser.add_argument('--labels', nargs='+', help='Labels for comparison plots')
    parser.add_argument('--output-dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--title', type=str, help='Plot title')
    parser.add_argument('--save-name', type=str, help='Output filename')

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib not installed. Run: pip install matplotlib")
        return

    # Initialize plotter
    plotter = LossPlotter(output_dir=args.output_dir)

    if args.log:
        # Plot single log
        plotter.plot_single_experiment(
            args.log,
            title=args.title,
            save_name=args.save_name
        )

    elif args.log_dir:
        # Plot all logs in directory
        plotter.plot_from_directory(args.log_dir)

    elif args.compare:
        # Compare multiple logs
        plotter.plot_comparison(
            args.compare,
            labels=args.labels,
            title=args.title or "Loss Comparison",
            save_name=args.save_name or "comparison.png"
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
