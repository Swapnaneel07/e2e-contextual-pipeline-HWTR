# End-to-End Contextual OCR Pipeline

This document outlines the technical plan to upgrade the current Handwritten Text Recognition project into a complete, end-to-end pipeline suitable for a comparative research study.

## Goal Description
Build a robust, end-to-end pipeline that takes a raw document image, detects text regions, recognizes the text using a fine-tuned TrOCR model, and finally uses a Large Language Model (LLM) to perform contextual error correction. We will benchmark the Character Error Rate (CER) and Word Error Rate (WER) before and after LLM correction to demonstrate the value of contextual understanding in OCR.

## User Review Required
> [!IMPORTANT]
> - **Dataset:** To evaluate the *end-to-end* pipeline (full page -> text), we need a dataset with full document images and their corresponding full-text ground truth. The IAM Handwriting Database is the standard for this. Does this sound good, or do you have another dataset in mind?
> - **Text Detection:** For Stage 1 (detecting words/lines on a page), I propose using **CRAFT** (Character Region Awareness for Text detection) as it is highly robust out of the box. Alternatively, we could train a YOLOv8 text detection model. I recommend starting with CRAFT.

## Open Questions
> [!WARNING]
> - Which LLM provider would you like to use for Stage 3? We can use the Gemini API, OpenAI API, or a local model (like Llama 3 via Ollama) if you have the GPU resources. An API is recommended for ease of benchmarking.
> - Do you want to retrain the TrOCR model immediately, or should we build the pipeline first using your existing GNHK fine-tuned model and then decide if we need a stronger recognition model later?

## Proposed Changes

We will restructure the project into a modular architecture to support reproducible research and clear evaluation.

### Project Cleanup
We will remove old UI apps and redundant scripts that are no longer needed for this research-focused pipeline.
#### [DELETE] `trocr_flask_app.py`
#### [DELETE] `trocr_pretrained_flask_app.py`
#### [DELETE] `trocr_ui_app.py`
#### [DELETE] `.gradio/`
#### [DELETE] `custom_inference.py`, `inference_fine_tuned.py`, `inference_fine_tuned_full.py`, `inference_pretrained.py`

---

### Core Pipeline Architecture (`src/`)

#### [NEW] `src/pipeline.py`
The master orchestrator that runs the end-to-end inference.
- Inputs an image -> Calls Detection -> Calls Recognition -> Calls Correction -> Outputs final text.

#### [NEW] `src/detection/craft_detector.py`
Stage 1: Text Detection
- Implements the CRAFT algorithm to detect text lines or words on the raw document image.
- Outputs cropped bounding boxes and their spatial coordinates so we can order the text properly (top-to-bottom, left-to-right).

#### [NEW] `src/recognition/trocr_recognizer.py`
Stage 2: Text Recognition
- Loads the fine-tuned TrOCR model.
- Processes the ordered text crops from Stage 1.
- Outputs the raw OCR string.

#### [NEW] `src/correction/llm_corrector.py`
Stage 3: Contextual Error Correction
- Integrates with the chosen LLM API.
- Implements a carefully crafted prompt instructing the LLM to fix spelling and recognition errors based on the context of the sentence without altering the original intent.

#### [NEW] `src/evaluation/metrics.py`
Stage 4: Benchmarking and Evaluation
- Computes Character Error Rate (CER) and Word Error Rate (WER).
- Script to run over a validation dataset and output a comparative CSV:
  `Filename | Ground Truth | Raw TrOCR | TrOCR + LLM | CER (Raw) | CER (Corrected)`

#### [NEW] `src/utils/visualizer.py`
- Generates side-by-side visualizations of the document image, bounding boxes, and the resulting transcriptions for use in your research paper.

## Verification Plan

### Automated Benchmarking
- We will run the entire `src/pipeline.py` over a test set (e.g., IAM test split).
- `src/evaluation/metrics.py` will generate the comparative study metrics (CER & WER improvements).

### Manual Verification
- We will manually review the `visualizer.py` outputs on 10-20 complex handwriting samples to qualitatively assess how well the LLM is fixing contextual errors.
