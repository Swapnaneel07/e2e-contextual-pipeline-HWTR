# Walkthrough: Pipeline Testing & Documentation

## Summary

Tested the end-to-end OCR pipeline on a GNHK handwritten sample and documented all concepts for the paper. Found and fixed 3 runtime bugs during testing.

---

## Bugs Found & Fixed

### Bug 1: CRAFT + torchvision incompatibility
- **Error:** `ImportError: cannot import name 'model_urls' from 'torchvision.models.vgg'`
- **Cause:** `craft-text-detector` v0.4.3 imports a deprecated `model_urls` dict that was removed in torchvision ≥ 0.15
- **Fix:** Monkey-patch `torchvision.models.vgg.model_urls` before CRAFT loads
- **File:** [craft_detector.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/detection/craft_detector.py)

### Bug 2: CRAFT + numpy incompatibility
- **Error:** `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape`
- **Cause:** numpy ≥ 1.24 rejects `np.array()` on lists of polygons with different vertex counts
- **Fix:** Patched `adjustResultCoordinates` to iterate and patched `predict.py` directly
- **Files:** [craft_detector.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/detection/craft_detector.py), `.venv/.../predict.py`

### Bug 3: Fine-tuned model producing empty output
- **Symptom:** Fine-tuned checkpoint generated empty strings for every crop
- **Cause:** `TrOCRProcessor.from_pretrained(checkpoint_path)` loaded a broken tokenizer from the incomplete checkpoint files, which decoded all tokens to `""`
- **Fix:** Always load processor from base model ID (`microsoft/trocr-small-handwritten`)
- **File:** [trocr_recognizer.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/recognition/trocr_recognizer.py)

### Bug 4: SpellCheck losing newlines
- **Symptom:** Multi-line text collapsed into single line after spell correction
- **Cause:** `.split()` without argument removes all whitespace including `\n`
- **Fix:** Process text line-by-line with `raw_ocr_text.split("\n")`
- **File:** [llm_corrector.py](file:///e:/Handwritten_Text_Recognition_using_OCR/src/correction/llm_corrector.py)

---

## Test Results

**Test image:** `eng_AF_011.jpg` — Handwritten quote: *"Life without Liberty is like a body without Spirit — Kahil Gibran"*

| Configuration | Output | Verdict |
|--------------|--------|---------|
| Pretrained + No Correction | "is a messes successful 333333..." | ❌ Completely wrong |
| Fine-tuned + No Correction | "Life without Liberty is / like a body without spirit / Kahl Qibran" | ✅ Nearly correct |
| Fine-tuned + SpellCheck | "Life without Liberty is / like a body without spirit / karl gibran" | ⚠️ Degraded proper nouns |

**Key finding:** SpellCheck **actively degrades** accuracy on proper nouns and names — strong motivation for LLM-based correction.

---

## Artifacts Created

| Document | Purpose |
|----------|---------|
| [pipeline_concepts.md](file:///C:/Users/user/.gemini/antigravity/brain/05d90620-5081-45de-939d-b276d469262f/pipeline_concepts.md) | Full technical reference for paper writing — covers architecture, algorithms, code, design decisions, and test results |

---

## Git Commits

```
d8e5605  Fix pipeline: patch CRAFT for modern numpy/torchvision, fix processor loading, fix spellcheck newlines
f8e2a7f  Update README for multi-backend comparative study architecture
2d3ecc2  Phase 1: Rewrite pipeline with multi-backend LLM support
```

---

## Next Steps

1. **Phase 3: Benchmarking Engine** — Write `benchmark.py` to run all correction methods over the full GNHK test set
2. **Phase 4: LLM Testing** — Install Ollama on powerful hardware, run Phi-4-mini and Gemini corrections
3. **Phase 5: Paper** — Compile results into figures and tables
