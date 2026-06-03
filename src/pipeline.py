import os
import argparse
from src.detection.craft_detector import TextDetector
from src.recognition.trocr_recognizer import TrOCRRecognizer
from src.correction.llm_corrector import LLMContextualCorrector
from src.utils.visualizer import visualize_results
import time

class EndToEndPipeline:
    def __init__(self, model_path="microsoft/trocr-small-handwritten", use_llm=True, device=None):
        print("Initializing End-to-End OCR Pipeline...")
        
        # Stage 1: Detection
        self.detector = TextDetector(cuda=(device=="cuda"))
        
        # Stage 2: Recognition
        self.recognizer = TrOCRRecognizer(model_path=model_path, device=device)
        
        # Stage 3: Correction
        self.use_llm = use_llm
        if self.use_llm:
            self.corrector = LLMContextualCorrector()
        else:
            self.corrector = None
            
        print("Pipeline initialization complete.")

    def process_image(self, image_path):
        """
        Process a full document image through the pipeline.
        """
        print(f"\nProcessing {image_path}...")
        start_time = time.time()
        
        # 1. Detection
        det_start = time.time()
        text_regions = self.detector.detect_and_crop(image_path)
        det_time = time.time() - det_start
        print(f"Detected {len(text_regions)} text regions in {det_time:.2f}s")
        
        if not text_regions:
            return "", ""
            
        # 2. Recognition
        rec_start = time.time()
        raw_text = self.recognizer.recognize_regions(text_regions)
        rec_time = time.time() - rec_start
        print(f"Recognized text in {rec_time:.2f}s")
        
        # 3. Correction
        corrected_text = raw_text
        if self.use_llm and self.corrector:
            cor_start = time.time()
            corrected_text = self.corrector.correct_text(raw_text)
            cor_time = time.time() - cor_start
            print(f"LLM correction applied in {cor_time:.2f}s")
            
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f}s\n")
        
        return raw_text, corrected_text, text_regions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Contextual OCR Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, default="microsoft/trocr-small-handwritten", help="Path to fine-tuned TrOCR model")
    parser.add_argument("--no_llm", action="store_true", help="Disable LLM correction stage")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    
    args = parser.parse_args()
    
    pipeline = EndToEndPipeline(
        model_path=args.model_path,
        use_llm=not args.no_llm,
        device=args.device
    )
    
    raw, corrected, text_regions = pipeline.process_image(args.image)
    
    print("="*40)
    print("RAW OCR OUTPUT:")
    print("-" * 40)
    print(raw)
    print("="*40)
    
    if not args.no_llm:
        print("LLM CORRECTED OUTPUT:")
        print("-" * 40)
        print(corrected)
        print("="*40)
        
    visualize_results(args.image, text_regions, raw, corrected, save_path="output_visualization.png")
