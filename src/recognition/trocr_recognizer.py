import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class TrOCRRecognizer:
    def __init__(self, model_path="microsoft/trocr-small-handwritten", device=None):
        """
        Initialize the TrOCR model and processor.
        :param model_path: Path to fine-tuned model checkpoint or Hugging Face model hub id
        :param device: "cuda" or "cpu"
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading TrOCR model from {model_path} on {self.device}...")
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def recognize_crop(self, image_crop):
        """
        Run TrOCR on a single image crop.
        :param image_crop: numpy array (BGR from OpenCV)
        :return: string, transcribed text
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0
            )
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()

    def recognize_regions(self, regions):
        """
        Run TrOCR on a list of image crops and reconstruct the full text.
        :param regions: List of dictionaries containing 'crop'
        :return: string, full transcribed text separated by spaces
        """
        words = []
        for region in regions:
            text = self.recognize_crop(region["crop"])
            if text:
                words.append(text)
        
        # Combine words into a single string
        # Future improvement: Use spatial coordinates to insert newlines appropriately
        return " ".join(words)

if __name__ == "__main__":
    # Test initialization
    recognizer = TrOCRRecognizer(model_path="../../trocr_handwritten/checkpoint-6770", device="cpu")
    print("TrOCR Recognizer initialized successfully.")
