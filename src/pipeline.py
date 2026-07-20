"""
End-to-End Contextual OCR Pipeline

Three-stage pipeline:
  1. Detection  (CRAFT)     → Finds text regions on a raw page image
  2. Recognition (TrOCR)    → Reads the cropped text
  3. Correction  (LLM/etc.) → Uses language context to fix OCR errors

Usage:
  python -m src.pipeline --image path/to/image.jpg
  python -m src.pipeline --image path/to/image.jpg --corrector ollama --llm_model phi4-mini
  python -m src.pipeline --image path/to/image.jpg --corrector gemini
  python -m src.pipeline --image path/to/image.jpg --no_correction
"""

import os
import argparse
import time

from src.detection.craft_detector import TextDetector
from src.recognition.trocr_recognizer import TrOCRRecognizer
from src.correction.llm_corrector import LLMContextualCorrector
from src.utils.visualizer import visualize_results


class EndToEndPipeline:
    def __init__(
        self,
        model_path="microsoft/trocr-small-handwritten",
        corrector_backend=None,
        llm_model=None,
        device=None,
    ):
        """
        :param model_path: Path to TrOCR checkpoint or HuggingFace model ID
        :param corrector_backend: "ollama", "gemini", "spellcheck", or None (no correction)
        :param llm_model: Model name for the corrector backend (e.g. "phi4-mini", "gemini-2.0-flash")
        :param device: "cuda" or "cpu"
        """
        print("=" * 50)
        print("  Initializing End-to-End OCR Pipeline")
        print("=" * 50)

        # Stage 1: Detection
        self.detector = TextDetector(cuda=(device == "cuda"))

        # Stage 2: Recognition
        self.recognizer = TrOCRRecognizer(model_path=model_path, device=device)

        # Stage 3: Correction
        self.corrector_backend = corrector_backend
        self.corrector = None
        if corrector_backend:
            kwargs = {}
            if llm_model:
                kwargs["model_name"] = llm_model
            self.corrector = LLMContextualCorrector(backend=corrector_backend, **kwargs)

        print("=" * 50)
        print(f"  Pipeline ready. Correction: {corrector_backend or 'DISABLED'}")
        print("=" * 50)

    def process_image(self, image_path):
        """
        Process a full document image through the 3-stage pipeline.

        :param image_path: Path to the input image
        :return: dict with keys:
                 'raw_text', 'corrected_text', 'text_regions',
                 'timings' (dict of stage durations in seconds)
        """
        print(f"\n--- Processing: {image_path} ---")
        timings = {}
        start_time = time.time()

        # 1. Detection
        t0 = time.time()
        text_regions = self.detector.detect_and_crop(image_path)
        timings["detection"] = time.time() - t0

        if not text_regions:
            print("[Pipeline] No text regions detected.")
            return {
                "raw_text": "",
                "corrected_text": "",
                "text_regions": [],
                "timings": timings,
            }

        # 2. Recognition
        t0 = time.time()
        raw_text = self.recognizer.recognize_regions(text_regions)
        timings["recognition"] = time.time() - t0

        # 3. Correction
        corrected_text = raw_text
        if self.corrector:
            t0 = time.time()
            corrected_text = self.corrector.correct_text(raw_text)
            timings["correction"] = time.time() - t0

        timings["total"] = time.time() - start_time

        # Print summary
        print(f"  Detection:   {timings['detection']:.2f}s  ({len(text_regions)} regions)")
        print(f"  Recognition: {timings['recognition']:.2f}s")
        if "correction" in timings:
            print(f"  Correction:  {timings['correction']:.2f}s  ({self.corrector_backend})")
        print(f"  Total:       {timings['total']:.2f}s")

        return {
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "text_regions": text_regions,
            "timings": timings,
        }


# --------------------------------------------------------------------------- #
#  CLI entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Contextual OCR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/trocr-small-handwritten",
        help="Path to fine-tuned TrOCR model or HF model ID",
    )
    parser.add_argument(
        "--corrector",
        type=str,
        choices=["ollama", "gemini", "spellcheck"],
        default=None,
        help="Correction backend to use (default: none)",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="Model name for the corrector (e.g. phi4-mini, gemini-2.0-flash)",
    )
    parser.add_argument("--no_correction", action="store_true", help="Disable correction stage")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument(
        "--save_viz",
        type=str,
        default="output_visualization.png",
        help="Path to save visualization image",
    )

    args = parser.parse_args()

    # Determine corrector backend
    corrector_backend = None if args.no_correction else args.corrector

    pipeline = EndToEndPipeline(
        model_path=args.model_path,
        corrector_backend=corrector_backend,
        llm_model=args.llm_model,
        device=args.device,
    )

    result = pipeline.process_image(args.image)

    # Print results
    print("\n" + "=" * 50)
    print("RAW OCR OUTPUT:")
    print("-" * 50)
    print(result["raw_text"])
    print("=" * 50)

    if corrector_backend:
        print(f"CORRECTED OUTPUT ({corrector_backend}):")
        print("-" * 50)
        print(result["corrected_text"])
        print("=" * 50)

    # Save visualization
    visualize_results(
        args.image,
        result["text_regions"],
        result["raw_text"],
        result["corrected_text"] if corrector_backend else None,
        save_path=args.save_viz,
    )


if __name__ == "__main__":
    main()
