"""
Benchmark Runner -- Convenience script for common benchmark configurations.

Run from the project root:
    python Testing/run_benchmark.py --mode quick_test
    python Testing/run_benchmark.py --mode spellcheck_only
    python Testing/run_benchmark.py --mode full
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.benchmark import BenchmarkEngine


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

DATASET_DIR = os.path.join(PROJECT_ROOT, "input", "gnhk_dataset", "test_data", "test")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Fine-tuned model path (falls back to pretrained if not found)
FINETUNED_MODEL = os.path.join(PROJECT_ROOT, "trocr_handwritten", "checkpoint-6770")
PRETRAINED_MODEL = "microsoft/trocr-small-handwritten"

# LLM model for Ollama
OLLAMA_MODEL = "phi4-mini"


# --------------------------------------------------------------------------- #
#  Preset modes
# --------------------------------------------------------------------------- #

MODES = {
    "quick_test": {
        "description": "Quick test: 3 samples, spellcheck only, CPU",
        "backends": ["spellcheck"],
        "max_samples": 3,
        "device": "cpu",
    },
    "spellcheck_only": {
        "description": "Full dataset with spellcheck correction only",
        "backends": ["spellcheck"],
        "max_samples": None,
        "device": None,  # auto-detect
    },
    "ollama_only": {
        "description": "Full dataset with Ollama (local LLM) correction only",
        "backends": ["ollama"],
        "max_samples": None,
        "device": None,
    },
    "spellcheck_ollama": {
        "description": "Full dataset with SpellCheck + Ollama comparison",
        "backends": ["spellcheck", "ollama"],
        "max_samples": None,
        "device": None,
    },
    "full": {
        "description": "Full dataset with all available backends",
        "backends": ["spellcheck", "ollama", "gemini"],
        "max_samples": None,
        "device": None,
    },
    "raw_only": {
        "description": "Full dataset, no correction (baseline CER/WER only)",
        "backends": [],
        "max_samples": None,
        "device": None,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks with preset configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available modes:\n" + "\n".join(
            f"  {k:<20} {v['description']}" for k, v in MODES.items()
        )
    )
    parser.add_argument("--mode", type=str, required=True, choices=list(MODES.keys()),
                        help="Benchmark mode to run")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu/cuda)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Override max samples")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="Use pretrained model instead of fine-tuned")

    args = parser.parse_args()

    mode = MODES[args.mode]
    print(f"\n{'='*60}")
    print(f"  Mode: {args.mode}")
    print(f"  {mode['description']}")
    print(f"{'='*60}")

    # Select model
    if args.use_pretrained or not os.path.exists(FINETUNED_MODEL):
        model_path = PRETRAINED_MODEL
        if not args.use_pretrained:
            print(f"  [INFO] Fine-tuned model not found at {FINETUNED_MODEL}")
            print(f"         Using pretrained model: {PRETRAINED_MODEL}")
    else:
        model_path = FINETUNED_MODEL

    print(f"  Model: {model_path}")

    # Check dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"\n  [ERROR] Dataset not found at: {DATASET_DIR}")
        print(f"  Please download the GNHK test dataset first.")
        sys.exit(1)

    # Resolve parameters
    device = args.device or mode["device"]
    max_samples = args.max_samples or mode["max_samples"]
    backends = mode["backends"]

    # Generate output path
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backends_str = "_".join(backends) if backends else "raw"
    output_csv = os.path.join(RESULTS_DIR, f"benchmark_{backends_str}_{timestamp}.csv")

    # Run benchmark
    engine = BenchmarkEngine(
        model_path=model_path,
        device=device,
        backends=backends,
        llm_model=OLLAMA_MODEL,
    )

    results = engine.run_dataset(
        dataset_dir=DATASET_DIR,
        output_csv=output_csv,
        max_samples=max_samples,
    )

    print(f"\n  Results saved to: {output_csv}")
    print(f"  To analyze: python Testing/analyze_and_plot.py --csv {output_csv}")


if __name__ == "__main__":
    main()
