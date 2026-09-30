# End-to-End Contextual OCR Pipeline — Task List

- [x] **Phase 1: Fix & Rebuild Pipeline**
  - [x] Clean up old files
  - [x] Rewrite `llm_corrector.py` — multi-backend support (Ollama, Gemini, SpellCheck)
  - [x] Fix `trocr_recognizer.py` — always load processor from base model
  - [x] Update `craft_detector.py` — line_idx tagging, adaptive line tolerance
  - [x] Patch CRAFT compatibility (torchvision model_urls + numpy ragged arrays)
  - [x] Rewrite `pipeline.py` — structured output with timings
  - [x] Update `metrics.py`, `visualizer.py`, `requirements.txt`, `.gitignore`, `README.md`

- [x] **Phase 2: End-to-End Testing**
  - [x] Install dependencies (craft-text-detector, scipy, pyspellchecker, etc.)
  - [x] Test pipeline with pretrained model (no correction) — ✅ works, poor accuracy
  - [x] Test pipeline with fine-tuned model (no correction) — ✅ works, nearly perfect
  - [x] Test pipeline with fine-tuned model + spellcheck — ✅ works, degrades proper nouns
  - [x] Fix fine-tuned model empty output (processor loading from checkpoint bug)
  - [x] Verify visualization output — ✅ correct bounding boxes, color-coded lines
  - [x] Document concepts and test results for paper

- [ ] **Phase 3: Build Benchmarking Engine**
  - [ ] Write `src/evaluation/benchmark.py` — batch evaluation over GNHK test set
  - [ ] Output comparative CSV: Image | Ground Truth | Raw OCR | SpellCheck | Local LLM | Cloud LLM | CER/WER for each
  - [ ] Compute aggregate statistics and improvement percentages
  - [ ] Generate comparison charts (bar charts, per-sample scatter plots)

- [ ] **Phase 4: LLM Correction Testing (on powerful hardware)**
  - [ ] Install Ollama + pull Phi-4-mini on college system
  - [ ] Run full benchmark with Ollama/Phi-4-mini correction
  - [ ] Run full benchmark with Gemini API correction
  - [ ] Compare all 4 methods and analyze results

- [ ] **Phase 5: Model Retraining (if needed)**
  - [ ] Evaluate if CER is sufficient or if retraining improves results
  - [ ] Retrain with early stopping, better augmentation
  - [ ] Re-run benchmarks with improved model

- [ ] **Phase 6: Paper Preparation**
  - [ ] Generate all figures and tables
  - [ ] Write up findings
