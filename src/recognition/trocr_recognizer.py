"""
TrOCR-based handwritten text recognition.

Handles loading of fine-tuned checkpoints where the tokenizer files may not be
present in the checkpoint directory (they are loaded from the base model instead).
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# The base model ID whose processor/tokenizer we always fall back to
BASE_MODEL_ID = "microsoft/trocr-small-handwritten"


class TrOCRRecognizer:
    def __init__(self, model_path=BASE_MODEL_ID, device=None):
        """
        Initialize the TrOCR model and processor.

        :param model_path: Path to fine-tuned checkpoint directory OR a HuggingFace
                           model hub id (e.g. "microsoft/trocr-small-handwritten").
        :param device: "cuda" or "cpu". Auto-detected if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[TrOCR] Loading model from: {model_path}")
        print(f"[TrOCR] Device: {self.device}")

        # --- Load processor (image feature extractor + tokenizer) ---
        # Fine-tuned checkpoints saved by HF Trainer typically contain only
        # preprocessor_config.json (image processor) but NOT the tokenizer
        # files. So we always load the processor from the base model ID to
        # guarantee the tokenizer is present.
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_path)
            print(f"[TrOCR] Processor loaded from: {model_path}")
        except (OSError, EnvironmentError):
            print(f"[TrOCR] Processor not found in checkpoint, loading from base: {BASE_MODEL_ID}")
            self.processor = TrOCRProcessor.from_pretrained(BASE_MODEL_ID)

        # --- Load model weights ---
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"[TrOCR] Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def recognize_crop(self, image_crop):
        """
        Run TrOCR on a single image crop.

        :param image_crop: numpy array (BGR from OpenCV) or PIL Image
        :return: Recognized text string
        """
        # Handle both numpy (BGR) and PIL inputs
        if isinstance(image_crop, Image.Image):
            pil_image = image_crop
        else:
            rgb_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

        pixel_values = self.processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
            )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def recognize_regions(self, regions):
        """
        Run TrOCR on a list of detected text regions.

        Uses line grouping information (if present) to reconstruct multi-line
        text with proper newlines.

        :param regions: List of dicts with at least 'crop' key.
                        Optionally 'line_idx' for line grouping.
        :return: Full transcribed text string
        """
        if not regions:
            return ""

        words = []
        current_line_idx = None

        for region in regions:
            text = self.recognize_crop(region["crop"])
            if not text:
                continue

            line_idx = region.get("line_idx")
            if line_idx is not None and current_line_idx is not None and line_idx != current_line_idx:
                # New line detected — join accumulated words with space, then add newline
                words.append("\n")
            current_line_idx = line_idx
            words.append(text)

        # Join with spaces, but respect the newline markers
        result_parts = []
        current_line_words = []
        for w in words:
            if w == "\n":
                if current_line_words:
                    result_parts.append(" ".join(current_line_words))
                    current_line_words = []
            else:
                current_line_words.append(w)
        if current_line_words:
            result_parts.append(" ".join(current_line_words))

        return "\n".join(result_parts) if result_parts else " ".join(
            w for w in words if w != "\n"
        )


if __name__ == "__main__":
    # Quick smoke test — just checks initialization, no actual inference
    print("Attempting to load TrOCR model...")
    try:
        recognizer = TrOCRRecognizer(model_path=BASE_MODEL_ID, device="cpu")
        print("TrOCR Recognizer initialized successfully.")
    except Exception as e:
        print(f"Error: {e}")
