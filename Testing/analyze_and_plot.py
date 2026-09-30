"""
Analyze & Plot -- Convenience wrapper for analyze_results.py

Run from the project root:
    python Testing/analyze_and_plot.py --csv results/benchmark_spellcheck_20261001.csv
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.analyze_results import analyze


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and generate paper-ready charts"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to benchmark CSV file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for output charts (default: same as CSV)")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    analyze(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
