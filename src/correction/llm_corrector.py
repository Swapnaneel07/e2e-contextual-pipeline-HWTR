"""
LLM-based contextual error correction for OCR output.

Supports multiple backends:
  - "ollama"  : Local LLM via Ollama HTTP API (default, free, reproducible)
  - "gemini"  : Google Gemini API (high quality, requires API key)
  - "spellcheck" : Simple rule-based spell-check baseline (no LLM)
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
#  Shared prompt template
# --------------------------------------------------------------------------- #

CORRECTION_PROMPT = """You are an expert OCR error correction system.
Below is the raw text output from a Handwritten Text Recognition model.
The OCR model often makes spelling mistakes, hallucinates characters, or merges/splits words incorrectly.

Rules:
- Fix spelling errors using sentence context (e.g., "tbe" -> "the").
- Fix grammar errors that were clearly caused by bad OCR.
- Do NOT change the original intent, meaning, or style.
- Do NOT add information that was not in the original text.
- Output ONLY the corrected text, nothing else.

Raw OCR Text:
{raw_text}"""


# --------------------------------------------------------------------------- #
#  Ollama backend (local LLM)
# --------------------------------------------------------------------------- #

class OllamaCorrector:
    """Connects to a locally-running Ollama server."""

    def __init__(self, model_name="phi4-mini", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._verify_connection()

    def _verify_connection(self):
        """Check that Ollama is reachable and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if any available model starts with the requested model name
            matched = [m for m in models if m.startswith(self.model_name)]
            if not matched:
                print(f"[WARNING] Model '{self.model_name}' not found in Ollama.")
                print(f"  Available models: {models}")
                print(f"  Run: ollama pull {self.model_name}")
            else:
                print(f"[Ollama] Connected. Using model: {matched[0]}")
                self.model_name = matched[0]  # Use the full tag
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)."
            )

    def correct(self, raw_ocr_text):
        prompt = CORRECTION_PROMPT.format(raw_text=raw_ocr_text)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # Low temp for deterministic correction
                "num_predict": 512,
            }
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            return resp.json().get("response", raw_ocr_text).strip()
        except Exception as e:
            print(f"[Ollama] Error: {e}")
            return raw_ocr_text


# --------------------------------------------------------------------------- #
#  Gemini backend (API)
# --------------------------------------------------------------------------- #

class GeminiCorrector:
    """Uses Google Gemini API for correction."""

    def __init__(self, model_name="gemini-2.0-flash"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in a .env file or environment variable."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        print(f"[Gemini] Initialized with model: {model_name}")

    def correct(self, raw_ocr_text):
        prompt = CORRECTION_PROMPT.format(raw_text=raw_ocr_text)
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return raw_ocr_text


# --------------------------------------------------------------------------- #
#  Spell-check baseline (no LLM)
# --------------------------------------------------------------------------- #

class SpellCheckCorrector:
    """Simple dictionary-based spell check baseline using pyspellchecker."""

    def __init__(self):
        try:
            from spellchecker import SpellChecker
        except ImportError:
            raise ImportError("Install pyspellchecker: pip install pyspellchecker")
        self.spell = SpellChecker()
        print("[SpellCheck] Initialized dictionary-based corrector.")

    def correct(self, raw_ocr_text):
        from spellchecker import SpellChecker
        words = raw_ocr_text.split()
        corrected = []
        for word in words:
            # Preserve punctuation
            stripped = word.strip(".,!?;:\"'()-")
            prefix = word[:len(word) - len(word.lstrip(".,!?;:\"'()-"))]
            suffix = word[len(stripped) + len(prefix):]

            correction = self.spell.correction(stripped)
            if correction is None:
                correction = stripped
            corrected.append(prefix + correction + suffix)
        return " ".join(corrected)


# --------------------------------------------------------------------------- #
#  Unified corrector factory
# --------------------------------------------------------------------------- #

class LLMContextualCorrector:
    """
    Unified interface for contextual OCR error correction.

    Usage:
        corrector = LLMContextualCorrector(backend="ollama", model_name="phi4-mini")
        corrected = corrector.correct_text("Th1s is sme 0CR txt.")
    """

    BACKENDS = {
        "ollama": OllamaCorrector,
        "gemini": GeminiCorrector,
        "spellcheck": SpellCheckCorrector,
    }

    def __init__(self, backend="ollama", **kwargs):
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(self.BACKENDS.keys())}")
        self.backend_name = backend
        self._corrector = self.BACKENDS[backend](**kwargs)

    def correct_text(self, raw_ocr_text):
        """Correct the raw OCR text using the configured backend."""
        if not raw_ocr_text or not raw_ocr_text.strip():
            return raw_ocr_text
        return self._corrector.correct(raw_ocr_text)

    def __repr__(self):
        return f"LLMContextualCorrector(backend='{self.backend_name}')"


# --------------------------------------------------------------------------- #
#  Quick test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    sample = "Th1s is a s3ntence with some 0CR errcrs in tbe text."
    print(f"Input:  {sample}\n")

    # Test whichever backend is available
    for backend in ["spellcheck", "ollama"]:
        try:
            corrector = LLMContextualCorrector(backend=backend)
            result = corrector.correct_text(sample)
            print(f"[{backend}] Output: {result}")
        except Exception as e:
            print(f"[{backend}] Skipped: {e}")
