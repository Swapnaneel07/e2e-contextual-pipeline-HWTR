# End-to-End Contextual OCR Pipeline

This repository implements a 3-stage Handwritten Text Recognition (HTR) pipeline designed for research and comparative studies on contextual error correction. 

## Architecture

The pipeline processes raw document images containing handwritten text, avoiding the need for pre-cropped word images.

1. **Stage 1 (Detection)**: Uses `CRAFT` (Character Region Awareness for Text detection) to identify and crop text regions on the page, maintaining reading order.
2. **Stage 2 (Recognition)**: Uses a fine-tuned `TrOCR` (Transformer OCR) model to predict the text content of the crops.
3. **Stage 3 (Contextual Correction)**: Passes the raw OCR string to an LLM (Gemini 1.5 Pro) to fix spelling and grammar errors using sentence context.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Create a `.env` file in the root directory and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Model Checkpoint**
   Ensure your fine-tuned TrOCR model is located in the appropriate directory or provide its Hugging Face hub ID.

## Usage

To run the end-to-end pipeline on an image:

```bash
python -m src.pipeline --image path/to/image.jpg --model_path microsoft/trocr-small-handwritten
```

**Arguments:**
- `--image`: Path to the input image (required).
- `--model_path`: Path to fine-tuned TrOCR model (default: "microsoft/trocr-small-handwritten").
- `--no_llm`: Disable the LLM correction stage.
- `--device`: `cuda` or `cpu`.

The script will output the detected text regions, the raw OCR text, the LLM-corrected text, and generate a side-by-side visualization (`output_visualization.png`).

## Evaluation

The `src/evaluation/metrics.py` script provides tools for calculating Character Error Rate (CER) and Word Error Rate (WER) using the Hugging Face `evaluate` library. This is intended to benchmark the CER improvements provided by Stage 3.