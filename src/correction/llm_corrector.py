import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class LLMContextualCorrector:
    def __init__(self, model_name="gemini-1.5-pro-latest"):
        """
        Initialize the LLM API for contextual error correction.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found. Please set it in a .env file.")
            
        genai.configure(api_key=api_key)
        
        # Use gemini-1.5-pro for high reasoning capabilities
        self.model = genai.GenerativeModel(model_name=model_name)

    def correct_text(self, raw_ocr_text):
        """
        Send the raw OCR text to the LLM for contextual correction.
        :param raw_ocr_text: String containing raw OCR output with potential errors
        :return: String containing the corrected text
        """
        prompt = f"""
You are an expert Optical Character Recognition (OCR) error correction system.
I am providing you with the raw text output from a Handwritten Text Recognition model.
Because handwriting can be messy and ambiguous, the OCR model often makes spelling mistakes, hallucinates characters, or merges words incorrectly.

Your task is to correct the text using the context of the sentence. 
- Fix spelling errors.
- Fix grammar errors that were clearly caused by bad OCR (e.g., "tbe" -> "the").
- Do NOT change the original intent, meaning, or style of the text.
- Do NOT add external information.
- Only output the corrected text, without any conversational filler or explanations.

Raw OCR Text:
{raw_ocr_text}
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error during LLM correction: {e}")
            return raw_ocr_text # Fallback to raw text if API fails

if __name__ == "__main__":
    # Test initialization
    try:
        corrector = LLMContextualCorrector()
        print("LLM Corrector initialized successfully.")
        # sample_text = "Th1s is a s3ntence with some 0CR errcrs."
        # print("Corrected:", corrector.correct_text(sample_text))
    except ValueError as e:
        print(e)
