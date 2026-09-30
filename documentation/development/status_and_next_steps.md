# Project Status Update & Next Steps

## Quick Recap: What This Project Is

A 3-stage pipeline for Handwritten Text Recognition:
1. **Detection (CRAFT)** → Finds text regions on a raw page image
2. **Recognition (TrOCR)** → Reads the cropped text
3. **Correction (LLM)** → Uses language context to fix OCR errors

The core research question: *How much does LLM-based contextual post-processing improve OCR accuracy?*

---

## Current Project State

### What's Done ✅
| Component | File | Status |
|-----------|------|--------|
| CRAFT Detector | [craft_detector.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/detection/craft_detector.py) | Implemented, untested |
| TrOCR Recognizer | [trocr_recognizer.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/recognition/trocr_recognizer.py) | Implemented, untested |
| LLM Corrector | [llm_corrector.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/correction/llm_corrector.py) | Implemented (uses Gemini API — **needs replacement**) |
| Pipeline Orchestrator | [pipeline.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/pipeline.py) | Implemented, untested |
| Metrics (CER/WER) | [metrics.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/evaluation/metrics.py) | Implemented, basic |
| Visualizer | [visualizer.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/utils/visualizer.py) | Implemented, basic |
| Fine-tuned TrOCR | `trocr_handwritten/checkpoint-6770` | Exists (246 MB, CER ≈ 0.23 on GNHK) |
| GNHK Dataset | `input/gnhk_dataset/` | Present (train + test, processed crops + CSVs) |

### What's NOT Done ❌
- Pipeline has **never been tested** end-to-end
- No `.env` file exists (LLM corrector will crash)
- No benchmark/evaluation script that runs the full pipeline over a dataset and outputs a comparison CSV
- The evaluation `metrics.py` module is **not wired into the pipeline** at all
- Line structure is lost — `recognize_regions()` just joins all words with spaces
- No IAM dataset downloaded yet (needed for formal benchmarking)

### Issues Found ⚠️
- **Overfitting signals**: Train loss dropped to 0.036 but eval CER only reached 0.23 (eval loss plateaued after epoch 4–5). We may want to retrain with early stopping later.
- **Checkpoint loading risk**: The checkpoint may not have tokenizer files bundled. Loading with `TrOCRProcessor.from_pretrained(checkpoint_path)` could fail — we may need to load the processor from the base model ID separately.
- **Duplicate metrics code**: Old [compute_metrics.py](file:///e:/Handwritten_Text_Recognition_using_OCR/compute_metrics.py) (root) uses `jiwer`, while `src/evaluation/metrics.py` uses HF `evaluate`. We should consolidate.

---

## LLM Comparison: API vs Local Open-Weight Models

Your hardware: **RTX 3060 with 4GB VRAM** (this is the laptop variant, not the 12GB desktop version).

> [!IMPORTANT]
> With only **4GB VRAM**, the GPU is already going to be occupied by CRAFT + TrOCR during inference. Running a local LLM simultaneously would require CPU-only inference for the LLM, or a sequential approach (unload OCR models → load LLM → run correction → unload LLM).

### Option A: Local Open-Weight LLM (via Ollama)

| Model | Size (Q4) | RAM Needed | Quality for Text Correction | Notes |
|-------|-----------|------------|------------------------------|-------|
| **Phi-4-mini (3.8B)** | ~2.5 GB | ~4 GB RAM | ★★★★☆ Good | Best fit for 4GB VRAM. Fast, strong instruction following. |
| **Qwen 2.5 3B** | ~2 GB | ~3.5 GB RAM | ★★★☆☆ Decent | Very small, might miss subtle corrections. |
| **Gemma 3 4B** | ~2.8 GB | ~4.5 GB RAM | ★★★★☆ Good | Strong reasoning, efficient. |
| **Qwen 2.5 7B** | ~4.5 GB | ~6 GB RAM | ★★★★★ Excellent | CPU-only on your setup; slower but high quality. |

**Pros:**
- ✅ Free, no API costs — critical for running benchmarks over hundreds/thousands of samples
- ✅ Fully reproducible — anyone can replicate your paper results
- ✅ No internet dependency
- ✅ You control the exact model version (important for reproducibility in a paper)

**Cons:**
- ❌ Slower inference (especially CPU-only for larger models)
- ❌ Quality ceiling is lower than top-tier API models

### Option B: Gemini API

**Pros:**
- ✅ Highest quality corrections (Gemini 1.5 Pro / 2.0 Flash)
- ✅ Fast inference, no local resource constraints

**Cons:**
- ❌ API costs add up over a full benchmark dataset
- ❌ Rate limits can slow down batch evaluation
- ❌ **Not reproducible** — model versions change, making paper results hard to replicate
- ❌ Requires internet connection

### Option C: Both (Recommended for a Comparative Study Paper) ✨

> [!TIP]
> Since your goal is a **comparative study paper**, testing multiple correction methods is actually the core contribution. We should benchmark:
> 1. **No correction** (raw TrOCR output) — the baseline
> 2. **Local small LLM correction** (e.g., Phi-4-mini via Ollama) — the practical, reproducible option
> 3. **API LLM correction** (Gemini Flash) — the high-end option
> 4. (Optional) **Simple rule-based spell-check** (e.g., using `pyspellchecker`) — a classical baseline
>
> This gives you a 4-way comparison table, which is much stronger for a paper than just "before vs after".

### My Recommendation

**Go with Option C (both), but make the local LLM the primary focus:**
- Use **Phi-4-mini** (via Ollama, runs on your GPU in Q4 at ~2.5 GB) as the primary local corrector.
- Use **Gemini 2.0 Flash** (via API, free tier should suffice for a test set) as the "upper bound" comparison.
- This way your paper shows: *"Even a small, locally-deployable LLM achieves significant CER improvement, while a large cloud LLM provides the theoretical ceiling."*

---

## Revised Next Steps

### Phase 1: Fix & Test the Pipeline (Immediate)
1. Fix the `llm_corrector.py` to support **Ollama** (local LLM via HTTP API) instead of only Gemini.
2. Fix the checkpoint loading issue (load processor from base model ID).
3. **Test the full pipeline end-to-end** on a single GNHK sample image.
4. Verify detection → recognition → correction flow works.

### Phase 2: Build the Benchmarking Engine
5. Write a `benchmark.py` script that:
   - Iterates over the GNHK test set (or IAM test set)
   - Runs each sample through the pipeline with/without correction
   - Outputs a CSV: `Image | Ground Truth | Raw OCR | LLM Corrected | CER (Raw) | CER (Corrected) | WER (Raw) | WER (Corrected)`
   - Computes aggregate statistics
6. Wire up `metrics.py` properly into the pipeline.

### Phase 3: Run the Comparative Study
7. Run benchmark with **no correction** → save results
8. Run benchmark with **Phi-4-mini** → save results
9. Run benchmark with **Gemini Flash** → save results
10. (Optional) Run with **spell-checker baseline** → save results
11. Generate comparison charts and tables for the paper.

### Phase 4: Paper Preparation
12. Generate visualizations (side-by-side examples, bar charts, etc.)
13. Write up findings

> [!IMPORTANT]
> **Which step should we start with?** I recommend starting with **Phase 1** — fixing the corrector to use Ollama and testing the pipeline end-to-end. Once we see it working on one image, we can confidently build the benchmarking engine.
