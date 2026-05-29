# TrueFrame Accuracy Improvement Plan

## Root Cause Analysis

**No trained ML model exists.** All 4 expected ONNX model files are missing:
- `models/lightfakedetect.onnx` (not found)
- `models/efficientnet_b0_v1.onnx` (not found)
- `models/swin_v2_l_deepfake.onnx` (not found)
- `models/trueframe_reels_detector.onnx` (not found)

The system runs entirely on **signal-analysis fallback** (`main.py:1128-1131`). Signal detectors (OpenCV frequency analysis, texture checks, edge detection) are unreliable — real images with JPEG compression trigger false positives, and sophisticated deepfakes evade detection.

Additionally, a `generate_dummy_model.py` exists that would create **random-weight ONNX models** producing garbage predictions.

---

## Implementation Phases

### Phase 1: HuggingFace Model Integration (Quick Win)
**Goal**: Get a working ML model powering predictions immediately.

**Files to modify:**
- `verified-stream/ai_service/main.py` (lines 262-299, 1101-1132)

**Changes:**
1. Import `HuggingFaceDeepfakeDetector` from `ai_core.models` in `main.py`
2. Create a module-level singleton for `HuggingFaceDeepfakeDetector`
3. In `analyze()` (line 1102): When ONNX session is `None`, fall back to HuggingFace model instead of pure signal analysis
4. Fusion: `final_score = 0.70 * hf_score + 0.30 * signal_score`
5. Add `requirements.txt` dependencies: `torch`, `transformers`, `pillow`

**Benefit:** The `dima806/deepfake_vs_real_image_detection` model (already used in `selfie_verify.py` and `models.py`) is pre-trained on deepfake datasets and will dramatically improve accuracy vs pure signal heuristics.

**Risk:** First inference downloads ~500MB model. Cold start latency.

---

### Phase 2: Train LightFakeDetect (Full Solution)
**Goal**: Train the custom MobileNetV2+CBAM+GRU model and export to ONNX.

**Files to use:**
- `verified-stream/ai_service/training/trainer.py` (training loop)
- `verified-stream/ai_service/training/model.py` (model arch)
- `verified-stream/ai_service/training/dataset.py` (data loading)
- `verified-stream/ai_service/training/export.py` (ONNX export)
- `verified-stream/ai_service/training/config.py` (hyperparams)

**Steps:**
1. Install training deps: `pip install -r requirements-training.txt`
2. Download datasets:
   - Celeb-DF v2 (official download)
   - FaceForensics++ (official download)
   - Place under `TRUEFRAME_DATA_DIR` (default: `project_root/data/`)
3. Run training: `python -m training.trainer` (40 epochs, ~4-8h on GPU)
4. Export to ONNX: `python -m training.export`
5. Output placed at `ai_service/models/lightfakedetect.onnx`

**Benefit:** Custom lightweight model optimized for the TrueFrame use case. No external API dependency.

**Risk:** Requires GPU, dataset download (tens of GB), and significant time.

---

### Phase 3: Signal Analysis Tuning
**Goal**: Reduce false positives from signal analysis by tuning thresholds.

**Files to modify:**
- `verified-stream/ai_service/main.py` (signal functions, boost logic lines 1138-1159)

**Changes:**
1. Run test suite against `test_media/` (contains real + deepfake samples)
2. Measure per-signal false positive rate on real images
3. Adjust per-signal thresholds (most signals are too sensitive)
4. Lower boost values further (current cap at 0.45 may still be too high)
5. Tweak fusion weights for signal-only path

**Benefit:** Reduces noise even before ML model is available.

---

### Phase 4: Add Automated Testing
**Goal**: Ensure accuracy is measurable and regressions are caught.

**Files to create/modify:**
- `verified-stream/ai_service/test_accuracy.py` (improve existing)
- Add curated test images with known ground truth labels

**Add to test suite:**
1. Test real images → expect `APPROVED` (score < 0.40)
2. Test deepfake images → expect `REJECTED` (score >= 0.75)
3. Test borderline cases → expect `UNDER_REVIEW`
4. Track accuracy metrics over time

---

## Verification

After each phase, verify accuracy with:
```bash
# Test individual files
python verified-stream/ai_service/main.py test_media/real/natural_face.png
python verified-stream/ai_service/main.py test_media/deepfake/gan_clean_face.png

# Run batch accuracy test
python test_accuracy.py
```

Expected improvements:
| Metric | Before (signal-only) | After Phase 1 | After Phase 2 |
|--------|---------------------|---------------|---------------|
| Real → APPROVED | ~40-60% | ~80-90% | ~90-95% |
| Deepfake → REJECTED | ~50-70% | ~85-95% | ~90-97% |
| False Positive Rate | ~30-50% | ~5-15% | ~3-10% |

---

## Critical Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `verified-stream/ai_service/main.py` | Main inference pipeline | 1-1261 |
| `verified-stream/ai_service/main.py` (ONNX loading) | Missing model detection | 89-112 |
| `verified-stream/ai_service/main.py` (fusion logic) | Score combination | 1118-1131 |
| `verified-stream/ai_service/main.py` (signal boosting) | Boost values causing FPs | 1138-1159 |
| `verified-stream/ai_service/ai_core/models.py` | HuggingFace model ready to use | 224-306 |
| `verified-stream/ai_service/config.py` | Thresholds (0.40/0.75) | 17-18 |
| `verified-stream/ai_service/selfie_verify.py` | Already uses HF model | 240,276 |
| `verified-stream/ai_service/training/trainer.py` | Training loop | 1-581 |
| `verified-stream/ai_service/training/config.py` | Training hyperparams | 1-327 |
| `verified-stream/ai_service/training/export.py` | ONNX export | 1-220 |
| `verified-stream/ai_service/training/model.py` | LightFakeDetect arch | 1-353 |
| `test_accuracy.py` | Accuracy test script | 1-92 |
