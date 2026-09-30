# Testing on Bigger Machine — Setup & Run Guide

> This folder contains scripts for running the full comparative benchmark on
> a machine with GPU and local LLM support (Ollama).

## Quick Setup

### 1. Clone the repo
```bash
git clone https://github.com/Swapnaneel07/e2e-contextual-pipeline-HWTR.git
cd e2e-contextual-pipeline-HWTR
```

### 2. Create virtual environment
```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
# CRAFT needs --no-deps due to outdated opencv pin:
pip install craft-text-detector --no-deps
pip install scipy pyspellchecker python-dotenv
```

### 4. Download the GNHK test dataset
Place the GNHK test data in `input/gnhk_dataset/test_data/test/`
(172 .jpg + 172 .json files).

### 5. (Optional) Copy the fine-tuned checkpoint
Copy `trocr_handwritten/checkpoint-6770/` from the original machine.
If not available, the pretrained model is used automatically.

---

## Install Ollama + LLM Model

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the LLM model
ollama pull phi4-mini

# Verify it's running
ollama list
```

---

## Running Benchmarks

### Step 1: Quick test (3 samples, spellcheck only)
```bash
python Testing/run_benchmark.py --mode quick_test
```

### Step 2: SpellCheck-only benchmark (full dataset)
```bash
python Testing/run_benchmark.py --mode spellcheck_only
```

### Step 3: Full comparative benchmark (all backends)
```bash
python Testing/run_benchmark.py --mode full
```

### Step 4: Custom run
```bash
python -m src.evaluation.benchmark \
    --dataset input/gnhk_dataset/test_data/test \
    --model_path trocr_handwritten/checkpoint-6770 \
    --backends spellcheck ollama \
    --llm_model phi4-mini \
    --device cuda
```

---

## Analyzing Results

After a benchmark run, analyze the CSV:

```bash
python Testing/analyze_and_plot.py --csv results/benchmark_XXXX.csv
```

This generates:
- **Summary table** (printed to console)
- **`_cer_comparison.png`** — Bar chart of avg CER per method
- **`_per_sample.png`** — Scatter plot of CER per sample
- **`_cer_delta.png`** — Histogram of CER improvement distribution
- **`_summary.json`** — Machine-readable summary statistics

---

## Expected Folder Structure After Running

```
results/
├── benchmark_spellcheck_20261001_120000.csv
├── benchmark_spellcheck_20261001_120000_cer_comparison.png
├── benchmark_spellcheck_20261001_120000_per_sample.png
├── benchmark_spellcheck_20261001_120000_cer_delta.png
├── benchmark_spellcheck_20261001_120000_summary.json
├── benchmark_spellcheck_ollama_20261001_130000.csv
└── ...
```

---

## GPU Notes

- For CUDA: make sure you have `torch` with CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- The `--device cuda` flag enables GPU for both CRAFT and TrOCR
- Ollama automatically uses GPU if available
