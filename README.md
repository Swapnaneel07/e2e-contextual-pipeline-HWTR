# End-to-End Contextual Pipeline for Handwritten Text Recognition

A comparative study on LLM-based contextual post-processing for improving
Handwritten Text Recognition (HTR) accuracy.

## Architecture

Three-stage pipeline that processes raw document images:

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  Stage 1     │     │  Stage 2     │     │  Stage 3            │
│  CRAFT       │ ──► │  TrOCR       │ ──► │  Contextual         │
│  Detection   │     │  Recognition │     │  Correction (LLM)   │
└──────────────┘     └──────────────┘     └─────────────────────┘
  Raw page image       Cropped word         Corrected text
  → Bounding boxes     images → Raw text    (Ollama / Gemini / SpellCheck)
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **For local LLM correction** (recommended):
   - Install [Ollama](https://ollama.ai/)
   - Pull a model: `ollama pull phi4-mini`
   - Start the server: `ollama serve`

3. **For Gemini API correction** (optional):
   - Create a `.env` file: `GEMINI_API_KEY=your_key_here`

## Usage

```bash
# No correction (raw OCR only)
python -m src.pipeline --image path/to/image.jpg --no_correction

# With local LLM correction (Ollama)
python -m src.pipeline --image path/to/image.jpg --corrector ollama --llm_model phi4-mini

# With Gemini API correction
python -m src.pipeline --image path/to/image.jpg --corrector gemini

# With spell-check baseline
python -m src.pipeline --image path/to/image.jpg --corrector spellcheck

# Use fine-tuned model
python -m src.pipeline --image path/to/image.jpg --model_path trocr_handwritten/checkpoint-6770
```

## Project Structure

```
├── src/
│   ├── pipeline.py                  # Main orchestrator
│   ├── detection/
│   │   └── craft_detector.py        # Stage 1: CRAFT text detection
│   ├── recognition/
│   │   └── trocr_recognizer.py      # Stage 2: TrOCR inference
│   ├── correction/
│   │   └── llm_corrector.py         # Stage 3: LLM/SpellCheck correction
│   ├── evaluation/
│   │   └── metrics.py               # CER/WER computation
│   └── utils/
│       └── visualizer.py            # Result visualization
├── preprocess_gnhk_dataset.py       # Dataset preprocessing
├── Fine_Tune_TrOCR_Handwritten.ipynb # Model training notebook
├── requirements.txt
└── README.md
```

## Evaluation

The comparative study benchmarks CER/WER across correction methods:
1. **No correction** — raw TrOCR output (baseline)
2. **SpellCheck** — dictionary-based baseline
3. **Local LLM** — Phi-4-mini via Ollama
4. **Cloud LLM** — Gemini Flash via API