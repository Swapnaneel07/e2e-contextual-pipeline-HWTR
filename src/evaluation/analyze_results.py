"""
Analyze benchmark results and generate comparison charts for the paper.

Reads the CSV output from benchmark.py and produces:
  1. Summary statistics table
  2. CER comparison bar chart
  3. Per-sample CER scatter plot
  4. CER improvement distribution histogram

Usage:
    python -m src.evaluation.analyze_results --csv results/benchmark_spellcheck_ollama_20261001.csv
"""

import os
import csv
import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(csv_path):
    """Load benchmark results from CSV."""
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key.endswith(("_cer", "_wer", "_delta", "_time")):
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = None
                elif key == "num_regions":
                    try:
                        row[key] = int(row[key])
                    except (ValueError, TypeError):
                        row[key] = 0
            results.append(row)
    return results


def detect_backends(results):
    """Auto-detect which correction backends are in the results."""
    if not results:
        return []
    backends = []
    for key in results[0]:
        if key.endswith("_cer") and key != "raw_cer":
            backend = key.replace("_cer", "")
            backends.append(backend)
    return sorted(backends)


def compute_summary(results, backends):
    """Compute aggregate statistics."""
    summary = {"raw": {"cer": [], "wer": []}}
    for b in backends:
        summary[b] = {"cer": [], "wer": [], "cer_delta": [], "time": []}

    for r in results:
        # Raw
        if r.get("raw_cer") is not None and r["raw_cer"] >= 0:
            summary["raw"]["cer"].append(r["raw_cer"])
        if r.get("raw_wer") is not None and r["raw_wer"] >= 0:
            summary["raw"]["wer"].append(r["raw_wer"])

        # Per backend
        for b in backends:
            if r.get(f"{b}_cer") is not None and r[f"{b}_cer"] >= 0:
                summary[b]["cer"].append(r[f"{b}_cer"])
            if r.get(f"{b}_wer") is not None and r[f"{b}_wer"] >= 0:
                summary[b]["wer"].append(r[f"{b}_wer"])
            if r.get(f"{b}_cer_delta") is not None:
                summary[b]["cer_delta"].append(r[f"{b}_cer_delta"])
            if r.get(f"{b}_time") is not None and r[f"{b}_time"] >= 0:
                summary[b]["time"].append(r[f"{b}_time"])

    # Compute aggregates
    stats = {}
    for method, data in summary.items():
        stats[method] = {}
        for metric, values in data.items():
            if values:
                stats[method][metric] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                }
            else:
                stats[method][metric] = {"mean": 0, "count": 0}

    return stats


def print_summary_table(stats, backends):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'Avg CER':>10} {'Med CER':>10} {'Avg WER':>10} {'CER D':>10} {'Avg Time':>10}")
    print("-" * 80)

    # Raw
    raw = stats["raw"]
    print(f"{'No Correction':<20} {raw['cer'].get('mean', 0):>10.4f} "
          f"{raw['cer'].get('median', 0):>10.4f} "
          f"{raw['wer'].get('mean', 0):>10.4f} "
          f"{'baseline':>10} {'--':>10}")

    # Backends
    for b in backends:
        s = stats[b]
        delta = s['cer_delta'].get('mean', 0)
        delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
        time_str = f"{s['time'].get('mean', 0):.2f}s" if s['time'].get('count', 0) > 0 else "--"

        print(f"{b:<20} {s['cer'].get('mean', 0):>10.4f} "
              f"{s['cer'].get('median', 0):>10.4f} "
              f"{s['wer'].get('mean', 0):>10.4f} "
              f"{delta_str:>10} {time_str:>10}")

    print("=" * 80)
    print(f"  Samples: {raw['cer'].get('count', 0)}")
    print(f"  Positive CER D = correction IMPROVED accuracy")


def plot_cer_comparison(stats, backends, output_path):
    """Bar chart comparing average CER across methods."""
    methods = ["No Correction"] + [b.title() for b in backends]
    cer_values = [stats["raw"]["cer"].get("mean", 0)]
    for b in backends:
        cer_values.append(stats[b]["cer"].get("mean", 0))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][:len(methods)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, cer_values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, cer_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Character Error Rate (CER)", fontsize=12)
    ax.set_title("Average CER by Correction Method", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(cer_values) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Chart] CER comparison saved to: {output_path}")


def plot_per_sample_scatter(results, backends, output_path):
    """Scatter plot showing CER per sample for raw vs corrected."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter valid samples
    valid = [r for r in results if r.get("raw_cer") is not None and r["raw_cer"] >= 0]
    if not valid:
        print("[Chart] No valid samples for scatter plot.")
        return

    x = range(len(valid))
    raw_cers = [r["raw_cer"] for r in valid]

    ax.scatter(x, raw_cers, label="No Correction", alpha=0.6, s=30, marker="o", color="#e74c3c")

    markers = ["s", "^", "D"]
    colors = ["#3498db", "#2ecc71", "#f39c12"]
    for i, b in enumerate(backends):
        cor_cers = [r.get(f"{b}_cer", r["raw_cer"]) for r in valid]
        ax.scatter(x, cor_cers, label=b.title(), alpha=0.6, s=30,
                   marker=markers[i % len(markers)], color=colors[i % len(colors)])

    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Character Error Rate (CER)", fontsize=12)
    ax.set_title("Per-Sample CER Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Chart] Per-sample scatter saved to: {output_path}")


def plot_cer_delta_histogram(results, backends, output_path):
    """Histogram showing distribution of CER improvement (delta) per backend."""
    fig, axes = plt.subplots(1, max(len(backends), 1), figsize=(6 * len(backends), 5), squeeze=False)

    for i, b in enumerate(backends):
        ax = axes[0][i]
        deltas = [r.get(f"{b}_cer_delta", 0) for r in results
                  if r.get(f"{b}_cer_delta") is not None]

        if not deltas:
            continue

        improved = sum(1 for d in deltas if d > 0)
        degraded = sum(1 for d in deltas if d < 0)
        unchanged = sum(1 for d in deltas if d == 0)

        ax.hist(deltas, bins=30, edgecolor="black", linewidth=0.5, color="#3498db", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1, label="No change")
        ax.set_xlabel("CER Delta (positive = improved)", fontsize=10)
        ax.set_ylabel("Number of Samples", fontsize=10)
        ax.set_title(f"{b.title()}\n^{improved} improved, v{degraded} degraded, ={unchanged} same",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Chart] CER delta histogram saved to: {output_path}")


def save_summary_json(stats, backends, output_path):
    """Save summary statistics as JSON for programmatic access."""
    # Convert numpy types to Python types for JSON serialization
    def to_python(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        return obj

    serializable = {}
    for method, data in stats.items():
        serializable[method] = {}
        for metric, values in data.items():
            serializable[method][metric] = {k: to_python(v) for k, v in values.items()}

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[Summary] JSON saved to: {output_path}")


def analyze(csv_path, output_dir=None):
    """Main analysis function."""
    results = load_results(csv_path)
    backends = detect_backends(results)

    print(f"\nAnalyzing {len(results)} samples with backends: {backends}")

    # Output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path) or "results"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Compute and print summary
    stats = compute_summary(results, backends)
    print_summary_table(stats, backends)

    # Generate charts
    plot_cer_comparison(stats, backends,
                        os.path.join(output_dir, f"{base_name}_cer_comparison.png"))
    plot_per_sample_scatter(results, backends,
                           os.path.join(output_dir, f"{base_name}_per_sample.png"))
    plot_cer_delta_histogram(results, backends,
                            os.path.join(output_dir, f"{base_name}_cer_delta.png"))

    # Save summary JSON
    save_summary_json(stats, backends,
                      os.path.join(output_dir, f"{base_name}_summary.json"))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results and generate charts")
    parser.add_argument("--csv", type=str, required=True, help="Path to benchmark CSV")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for charts (default: same as CSV)")
    args = parser.parse_args()

    analyze(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
