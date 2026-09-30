"""
Benchmark Engine for the End-to-End Contextual OCR Pipeline.

Runs the full pipeline across a dataset with multiple correction backends,
computes CER/WER metrics per sample, and saves results to CSV.

Usage:
    python -m src.evaluation.benchmark --help
"""

import os
import csv
import json
import time
import argparse
import traceback
from datetime import datetime

from src.detection.craft_detector import TextDetector
from src.recognition.trocr_recognizer import TrOCRRecognizer
from src.correction.llm_corrector import LLMContextualCorrector
from src.evaluation.metrics import MetricsEvaluator
from src.evaluation.gnhk_loader import list_dataset_samples, load_ground_truth


class BenchmarkEngine:
    """
    Runs OCR pipeline over a dataset and benchmarks multiple correction backends.

    The engine initializes the detector and recognizer once, then runs each
    image through detection -> recognition -> correction(s), comparing against
    ground truth.
    """

    def __init__(
        self,
        model_path="microsoft/trocr-small-handwritten",
        device=None,
        backends=None,
        llm_model=None,
    ):
        """
        :param model_path: TrOCR model path or HuggingFace ID
        :param device: "cuda" or "cpu"
        :param backends: List of correction backends to test
                         e.g. ["spellcheck", "ollama", "gemini"]
        :param llm_model: Model name for LLM backends (e.g. "phi4-mini")
        """
        self.device = device
        self.backends = backends or []
        self.llm_model = llm_model

        print("=" * 60)
        print("  BENCHMARK ENGINE -- Initializing")
        print("=" * 60)

        # Stage 1 & 2: Shared detector + recognizer
        self.detector = TextDetector(cuda=(device == "cuda"))
        self.recognizer = TrOCRRecognizer(model_path=model_path, device=device)

        # Stage 3: Initialize correctors
        self.correctors = {}
        for backend in self.backends:
            try:
                kwargs = {}
                if llm_model and backend in ("ollama",):
                    kwargs["model_name"] = llm_model
                self.correctors[backend] = LLMContextualCorrector(backend=backend, **kwargs)
                print(f"  [OK] Corrector '{backend}' initialized")
            except Exception as e:
                print(f"  [FAIL] Corrector '{backend}' FAILED: {e}")

        # Metrics evaluator
        self.metrics = MetricsEvaluator()

        print(f"\n  Active backends: {list(self.correctors.keys())}")
        print("=" * 60)

    def run_single_sample(self, image_path, ground_truth_text):
        """
        Run all correction backends on a single image.

        :param image_path: Path to the input image
        :param ground_truth_text: Ground truth text for evaluation
        :return: dict with per-backend results
        """
        result = {
            "image": os.path.basename(image_path),
            "ground_truth": ground_truth_text,
        }

        # ---- Stage 1: Detection ----
        try:
            t0 = time.time()
            text_regions = self.detector.detect_and_crop(image_path)
            result["detection_time"] = round(time.time() - t0, 2)
            result["num_regions"] = len(text_regions)
        except Exception as e:
            print(f"  [ERROR] Detection failed: {e}")
            result["detection_time"] = -1
            result["num_regions"] = 0
            result["raw_ocr"] = ""
            result["raw_cer"] = -1
            result["raw_wer"] = -1
            return result

        if not text_regions:
            result["raw_ocr"] = ""
            result["raw_cer"] = 1.0
            result["raw_wer"] = 1.0
            result["detection_time"] = result.get("detection_time", 0)
            return result

        # ---- Stage 2: Recognition ----
        try:
            t0 = time.time()
            raw_text = self.recognizer.recognize_regions(text_regions)
            result["recognition_time"] = round(time.time() - t0, 2)
            result["raw_ocr"] = raw_text
        except Exception as e:
            print(f"  [ERROR] Recognition failed: {e}")
            result["raw_ocr"] = ""
            result["recognition_time"] = -1

        # ---- Compute raw metrics ----
        if ground_truth_text.strip() and result["raw_ocr"].strip():
            raw_metrics = self.metrics.compute_single(
                result["raw_ocr"].replace("\n", " "),
                ground_truth_text.replace("\n", " ")
            )
            result["raw_cer"] = round(raw_metrics["cer"], 4)
            result["raw_wer"] = round(raw_metrics["wer"], 4)
        else:
            result["raw_cer"] = 1.0 if ground_truth_text.strip() else 0.0
            result["raw_wer"] = 1.0 if ground_truth_text.strip() else 0.0

        # ---- Stage 3: Correction (each backend) ----
        for backend_name, corrector in self.correctors.items():
            try:
                t0 = time.time()
                corrected_text = corrector.correct_text(raw_text)
                correction_time = round(time.time() - t0, 2)

                result[f"{backend_name}_text"] = corrected_text
                result[f"{backend_name}_time"] = correction_time

                # Compute corrected metrics
                if ground_truth_text.strip() and corrected_text.strip():
                    cor_metrics = self.metrics.compute_single(
                        corrected_text.replace("\n", " "),
                        ground_truth_text.replace("\n", " ")
                    )
                    result[f"{backend_name}_cer"] = round(cor_metrics["cer"], 4)
                    result[f"{backend_name}_wer"] = round(cor_metrics["wer"], 4)
                else:
                    result[f"{backend_name}_cer"] = 1.0 if ground_truth_text.strip() else 0.0
                    result[f"{backend_name}_wer"] = 1.0 if ground_truth_text.strip() else 0.0

                # Compute CER improvement delta
                if result["raw_cer"] > 0:
                    result[f"{backend_name}_cer_delta"] = round(
                        result["raw_cer"] - result[f"{backend_name}_cer"], 4
                    )
                else:
                    result[f"{backend_name}_cer_delta"] = 0.0

            except Exception as e:
                print(f"  [ERROR] {backend_name} correction failed: {e}")
                result[f"{backend_name}_text"] = raw_text
                result[f"{backend_name}_time"] = -1
                result[f"{backend_name}_cer"] = result["raw_cer"]
                result[f"{backend_name}_wer"] = result["raw_wer"]
                result[f"{backend_name}_cer_delta"] = 0.0

        return result

    def run_dataset(self, dataset_dir, output_csv, max_samples=None):
        """
        Run the full benchmark over a GNHK dataset directory.

        :param dataset_dir: Path to directory with .jpg/.json pairs
        :param output_csv: Path to save the results CSV
        :param max_samples: Limit number of samples (for testing)
        :return: List of per-sample result dicts
        """
        samples = list_dataset_samples(dataset_dir)
        if max_samples:
            samples = samples[:max_samples]

        print(f"\n{'='*60}")
        print(f"  BENCHMARK: {len(samples)} samples from {dataset_dir}")
        print(f"  Backends: {list(self.correctors.keys())}")
        print(f"  Output: {output_csv}")
        print(f"{'='*60}\n")

        all_results = []
        start_time = time.time()

        for i, sample in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] {sample['sample_id']}")

            # Load ground truth
            gt = load_ground_truth(sample["json_path"])
            gt_text = gt["full_text"]

            if not gt_text.strip():
                print("  Skipping -- empty ground truth")
                continue

            # Run pipeline
            result = self.run_single_sample(sample["image_path"], gt_text)
            result["sample_id"] = sample["sample_id"]
            all_results.append(result)

            # Progress summary
            print(f"  Raw CER: {result.get('raw_cer', 'N/A')}")
            for backend in self.correctors:
                cer = result.get(f"{backend}_cer", "N/A")
                delta = result.get(f"{backend}_cer_delta", "N/A")
                print(f"  {backend} CER: {cer} (D={delta})")

        total_time = time.time() - start_time

        # Save results
        self._save_csv(all_results, output_csv)

        # Print summary
        self._print_summary(all_results, total_time)

        return all_results

    def _save_csv(self, results, output_path):
        """Save benchmark results to CSV."""
        if not results:
            print("[Benchmark] No results to save.")
            return

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Determine fieldnames from the first result (plus any extra from backends)
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())

        # Order fields logically
        ordered_fields = ["sample_id", "image", "num_regions",
                          "detection_time", "recognition_time",
                          "ground_truth", "raw_ocr", "raw_cer", "raw_wer"]

        for backend in self.correctors:
            ordered_fields.extend([
                f"{backend}_text", f"{backend}_time",
                f"{backend}_cer", f"{backend}_wer", f"{backend}_cer_delta"
            ])

        # Add any remaining keys not yet ordered
        remaining = sorted(all_keys - set(ordered_fields))
        fieldnames = ordered_fields + remaining

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        print(f"\n[Benchmark] Results saved to: {output_path}")

    def _print_summary(self, results, total_time):
        """Print aggregate benchmark summary."""
        if not results:
            return

        print(f"\n{'='*60}")
        print(f"  BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"  Samples processed: {len(results)}")
        print(f"  Total time: {total_time:.1f}s ({total_time/len(results):.1f}s/sample)")

        # Aggregate raw metrics
        valid_raw = [r for r in results if r.get("raw_cer", -1) >= 0]
        if valid_raw:
            avg_raw_cer = sum(r["raw_cer"] for r in valid_raw) / len(valid_raw)
            avg_raw_wer = sum(r["raw_wer"] for r in valid_raw) / len(valid_raw)
            print(f"\n  Raw OCR:")
            print(f"    Avg CER: {avg_raw_cer:.4f}")
            print(f"    Avg WER: {avg_raw_wer:.4f}")

        # Per-backend metrics
        for backend in self.correctors:
            valid = [r for r in results if r.get(f"{backend}_cer", -1) >= 0]
            if valid:
                avg_cer = sum(r[f"{backend}_cer"] for r in valid) / len(valid)
                avg_wer = sum(r[f"{backend}_wer"] for r in valid) / len(valid)
                avg_delta = sum(r.get(f"{backend}_cer_delta", 0) for r in valid) / len(valid)
                avg_time = sum(r.get(f"{backend}_time", 0) for r in valid) / len(valid)

                print(f"\n  {backend}:")
                print(f"    Avg CER: {avg_cer:.4f} (D from raw: {avg_delta:+.4f})")
                print(f"    Avg WER: {avg_wer:.4f}")
                print(f"    Avg correction time: {avg_time:.2f}s")

        print(f"\n{'='*60}")


# --------------------------------------------------------------------------- #
#  CLI entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the OCR pipeline across a GNHK dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (3 samples, spellcheck only)
  python -m src.evaluation.benchmark --dataset input/gnhk_dataset/test_data/test --max_samples 3 --backends spellcheck

  # Full benchmark with all backends
  python -m src.evaluation.benchmark --dataset input/gnhk_dataset/test_data/test --backends spellcheck ollama gemini

  # Use fine-tuned model + local LLM
  python -m src.evaluation.benchmark --dataset input/gnhk_dataset/test_data/test --model_path trocr_handwritten/checkpoint-6770 --backends spellcheck ollama --llm_model phi4-mini
        """
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to GNHK dataset directory (e.g. input/gnhk_dataset/test_data/test)")
    parser.add_argument("--model_path", type=str, default="microsoft/trocr-small-handwritten",
                        help="TrOCR model path or HuggingFace ID")
    parser.add_argument("--backends", nargs="+", default=["spellcheck"],
                        choices=["spellcheck", "ollama", "gemini"],
                        help="Correction backends to benchmark")
    parser.add_argument("--llm_model", type=str, default="phi4-mini",
                        help="Model name for Ollama backend")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: auto-generated in results/)")

    args = parser.parse_args()

    # Auto-generate output path
    if args.output is None:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backends_str = "_".join(args.backends)
        args.output = f"results/benchmark_{backends_str}_{timestamp}.csv"

    # Run benchmark
    engine = BenchmarkEngine(
        model_path=args.model_path,
        device=args.device,
        backends=args.backends,
        llm_model=args.llm_model,
    )

    results = engine.run_dataset(
        dataset_dir=args.dataset,
        output_csv=args.output,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
