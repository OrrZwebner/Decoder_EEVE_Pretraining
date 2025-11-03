#!/usr/bin/env python3
"""
Fix experiments.csv formatting issues:
1. Convert learning_rate to scientific notation
2. Replace empty fields with 'N/A' for better readability
3. Format perplexity values with more decimal places

Usage:
    python fix_experiments_csv.py

This will:
- Backup the original experiments.csv
- Create a new experiments.csv with fixed formatting
"""

import csv
import shutil
from pathlib import Path
from datetime import datetime


def format_value(value, field_name):
    """Format a value based on the field name"""

    # Handle empty values
    if not value or value.strip() == '':
        return 'N/A'

    # Format learning rate in scientific notation
    if field_name == 'learning_rate':
        try:
            lr_value = float(value)
            if lr_value == 0:
                return '0.00e+00'
            return f"{lr_value:.2e}"
        except (ValueError, TypeError):
            return value

    # Format perplexity with more precision
    if 'perplexity' in field_name:
        try:
            perp_value = float(value)
            return f"{perp_value:.6f}"
        except (ValueError, TypeError):
            return value

    # Format floating point numbers with consistent precision
    if field_name in ['train_loss_start', 'train_loss_final', 'eval_loss_start', 'eval_loss_final']:
        try:
            loss_value = float(value)
            return f"{loss_value:.6f}"
        except (ValueError, TypeError):
            return value

    # Format GPU memory consistently
    if field_name == 'gpu_memory_gb':
        try:
            mem_value = float(value)
            return f"{mem_value:.2f}"
        except (ValueError, TypeError):
            return value

    # Return as-is for other fields
    return value


def fix_experiments_csv(csv_path):
    """Fix formatting issues in experiments.csv"""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    # Create backup
    backup_path = csv_path.parent / f"experiments_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    shutil.copy(csv_path, backup_path)
    print(f"✅ Created backup: {backup_path}")

    # Read existing data
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"📊 Processing {len(rows)} experiments...")

    # Fix each row
    fixed_rows = []
    for row in rows:
        fixed_row = {}
        for field in fieldnames:
            fixed_row[field] = format_value(row.get(field, ''), field)
        fixed_rows.append(fixed_row)

    # Write fixed data
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fixed_rows)

    print(f"✅ Fixed {len(fixed_rows)} experiments")
    print(f"✅ Updated: {csv_path}")

    # Print summary of changes
    print("\n📈 Summary of fixes:")

    # Count non-N/A values per field
    field_stats = {field: 0 for field in fieldnames}
    for row in fixed_rows:
        for field, value in row.items():
            if value != 'N/A':
                field_stats[field] += 1

    print("\nField coverage (non-N/A values):")
    for field, count in sorted(field_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(fixed_rows)) * 100 if fixed_rows else 0
        print(f"  {field:25s}: {count:3d}/{len(fixed_rows)} ({percentage:5.1f}%)")


def main():
    # Default path
    csv_path = Path(__file__).parent.parent / "experiments.csv"

    print("🔧 Fixing experiments.csv formatting issues...")
    print(f"📁 CSV path: {csv_path}")
    print()

    fix_experiments_csv(csv_path)

    print("\n✨ Done! Your experiments.csv has been updated with:")
    print("  • Learning rates in scientific notation (e.g., 5.00e-05)")
    print("  • Missing values marked as 'N/A'")
    print("  • Consistent decimal precision for losses and perplexity")
    print()
    print("💡 Tip: Open experiments.csv in Excel/LibreOffice to verify the changes")


if __name__ == "__main__":
    main()
