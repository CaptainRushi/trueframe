# Graph Report - .  (2026-05-25)

## Corpus Check
- 150 files · ~160,195 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 683 nodes · 932 edges · 56 communities detected
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 68 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `FaceAnalyzer` - 29 edges
2. `SwinLONNXDetector` - 21 edges
3. `MetricTracker` - 20 edges
4. `LightFakeDetect` - 17 edges
5. `_run_signal_analysis()` - 16 edges
6. `HuggingFaceDeepfakeDetector` - 14 edges
7. `_run_signal_analysis()` - 13 edges
8. `TrueFrameTrainer` - 11 edges
9. `analyze()` - 10 edges
10. `_log()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `TrueFrame — Model Export & Deployment ====================================== Exp` --uses--> `LightFakeDetect`  [INFERRED]
  verified-stream\ai_service\training\export.py → verified-stream\ai_service\training\model.py
- `Wraps LightFakeDetect to output only P(fake) for ONNX export.` --uses--> `LightFakeDetect`  [INFERRED]
  verified-stream\ai_service\training\export.py → verified-stream\ai_service\training\model.py
- `Export a trained LightFakeDetect model to ONNX + metadata.      The exported ONN` --uses--> `LightFakeDetect`  [INFERRED]
  verified-stream\ai_service\training\export.py → verified-stream\ai_service\training\model.py
- `Export model to ONNX with dynamic batch and sequence length axes.          Args:` --uses--> `LightFakeDetect`  [INFERRED]
  verified-stream\ai_service\training\export.py → verified-stream\ai_service\training\model.py
- `Run a quick inference check on the exported ONNX model.` --uses--> `LightFakeDetect`  [INFERRED]
  verified-stream\ai_service\training\export.py → verified-stream\ai_service\training\model.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (6): fetchComments(), handleSubmit(), fetchCreatorStatus(), handleApply(), fetchHistory(), initProfile()

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (31): FaceAnalyzer, Face & Region Extraction using MediaPipe (or Haar Cascade fallback).         Ret, EfficientNetONNXDetector, HuggingFaceDeepfakeDetector, _log(), Load the Swin-L ONNX model for tertiary review., Load the EfficientNet ONNX model using ONNX Runtime., SwinLONNXDetector (+23 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (39): LightFakeDetectExporter, _ONNXWrapper, TrueFrame — Model Export & Deployment ====================================== Exp, Run a quick inference check on the exported ONNX model., Save model metadata JSON alongside the ONNX file., Export ONNX + metadata. Returns paths dict., Wraps LightFakeDetect to output only P(fake) for ONNX export., Export a trained LightFakeDetect model to ONNX + metadata.      The exported ONN (+31 more)

### Community 3 - "Community 3"
Cohesion: 0.04
Nodes (0): 

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (44): analyze_video(), _build_detector(), _build_result(), _extract_face_crops(), _extract_raw_frames(), _filter_similar_frames(), _get_detector(), _get_onnx_session() (+36 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (40): analyze(), _build_detector(), _build_result(), _detect_face(), _expression_rois(), _flow_magnitude(), _get_detector(), _get_face_crops() (+32 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (29): _balance_classes(), CelebDFLoader, collate_fn(), create_dataloaders(), CustomReelsLoader, DFDCLoader, FaceForensicsLoader, load_all_entries() (+21 more)

### Community 7 - "Community 7"
Cohesion: 0.09
Nodes (20): MetricTracker, TrueFrame Reels — Evaluation Metrics ====================================== Metr, build_optimizer(), build_scheduler(), _cosine_warmup_scheduler(), EarlyStopping, TrueFrame Reels — Training Engine ==================================== Full trai, Stops training when the monitored metric stops improving. (+12 more)

### Community 8 - "Community 8"
Cohesion: 0.1
Nodes (15): AntiDeepfakeSignalAnalyzer, BlinkDetector, GANFingerprintDetector, LipSyncDetector, MetadataValidator, TrueFrame Reels — Additional Anti-Deepfake Signals =============================, Compute magnitude of optical flow between mouth frames., Analyze if mouth movements show unnatural patterns. (+7 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (17): AugmentationConfig, DatasetConfig, DeploymentConfig, EvaluationConfig, FrameExtractionConfig, ModelConfig, TrueFrame Reels — Training Configuration =======================================, Training schedule and optimization settings. (+9 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (18): _build_mtcnn_detector(), crops_to_tensor(), extract_frames(), filter_similar_frames(), _get_detector(), normalize_frame(), preprocess_image(), preprocess_video() (+10 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (9): TrueFrame Reels — Video Deepfake Detector Test Suite ===========================, Verify that block artifacts detection scoring responds to grid blocks., Verify CLI execution on the test video file returns expected schema and verdict., Verify CLI execution handles non-existent video path by returning insufficient_f, Verify CLI execution fails with code 1 and REJECTED when no arguments are provid, Verify triage classification with an under-review band., Verify temporal flicker signal scores flickering vs static inputs., Verify color consistency signal detects hue fluctuations. (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (2): resetUpload(), stopCamera()

### Community 13 - "Community 13"
Cohesion: 0.21
Nodes (9): _BasicTransform, _get_basic_transform(), get_train_transforms(), get_val_transforms(), TrueFrame Reels — Data Augmentation Pipeline ===================================, Validation / test transforms — minimal processing., Fallback transform when albumentations is not installed., Fallback when albumentations is not available. (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.29
Nodes (9): check_patterns(), check_temporal_consistency(), classify_claim_type(), extract_claims(), Step 1: Claim Extraction (NLP/Rule-based)     Extracts factual assertions., Step 2: Claim Type Classification, Step 3: Temporal & Context Check (CRITICAL)     Compares media creation age wit, Step 5: Known Misinfo Patterns (+1 more)

### Community 15 - "Community 15"
Cohesion: 0.33
Nodes (5): addToRemoveQueue(), dispatch(), genId(), reducer(), toast()

### Community 16 - "Community 16"
Cohesion: 0.31
Nodes (6): deleteVerificationData(), fetchStatus(), resetFlow(), retakePhoto(), startCamera(), stopCamera()

### Community 17 - "Community 17"
Cohesion: 0.39
Nodes (5): applyTrustPenalty(), notifyCommunityVerifiers(), removeConfirmedDeepfake(), restorePost(), triggerSecondaryReview()

### Community 18 - "Community 18"
Cohesion: 0.33
Nodes (2): DummyEfficientNet, DummyReelsDetector

### Community 19 - "Community 19"
Cohesion: 0.33
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 0.5
Nodes (2): getPythonCommand(), runSelfieVerification()

### Community 21 - "Community 21"
Cohesion: 0.4
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 0.83
Nodes (3): callRemoteAIService(), getPythonCommand(), runAIScript()

### Community 23 - "Community 23"
Cohesion: 0.67
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 0.67
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 0.67
Nodes (0): 

### Community 26 - "Community 26"
Cohesion: 0.67
Nodes (0): 

### Community 27 - "Community 27"
Cohesion: 0.67
Nodes (0): 

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (0): 

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (0): 

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "Community 34"
Cohesion: 2.0
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (0): 

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (0): 

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (0): 

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (0): 

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (0): 

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (0): 

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (0): 

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (0): 

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (0): 

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (0): 

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (0): 

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (0): 

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (0): 

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (0): 

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (0): 

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (0): 

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **117 isolated node(s):** `Step 1: Claim Extraction (NLP/Rule-based)     Extracts factual assertions.`, `Step 2: Claim Type Classification`, `Step 3: Temporal & Context Check (CRITICAL)     Compares media creation age wit`, `Step 5: Known Misinfo Patterns`, `TrueFrame AI Service — Deepfake Detector (Images + Videos) =====================` (+112 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 28`** (2 nodes): `plot_results.py`, `interpolate()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (2 nodes): `test_supabase.ts`, `test()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (2 nodes): `check_logs.ts`, `checkLogs()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (2 nodes): `sync_profiles.ts`, `syncAllProfilesDetailed()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (2 nodes): `test_feed_query.ts`, `testQuery()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (2 nodes): `paths.ts`, `getAiServicePath()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (2 nodes): `redis.ts`, `connectRedis()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (2 nodes): `account.ts`, `accountRoutes()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (2 nodes): `community.ts`, `communityRoutes()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (2 nodes): `social.ts`, `socialRoutes()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (2 nodes): `transparency.ts`, `transparencyRoutes()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (2 nodes): `gen_pdf_report.py`, `create_pdf_report()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (2 nodes): `StoryCircle.tsx`, `StoryCircle()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (2 nodes): `UploadStep.tsx`, `UploadStep()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (2 nodes): `NotFound.tsx`, `NotFound()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `eslint.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `postcss.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `tailwind.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `plot_table.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `gen_metrics_plot.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `html-to-pdf.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `generate-version.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `redeploy.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `vite-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `aspect-ratio.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `collapsible.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Are the 23 inferred relationships involving `FaceAnalyzer` (e.g. with `FrequencyAnalyzer` and `PatchConsistencyAnalyzer`) actually correct?**
  _`FaceAnalyzer` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `SwinLONNXDetector` (e.g. with `FrequencyAnalyzer` and `PatchConsistencyAnalyzer`) actually correct?**
  _`SwinLONNXDetector` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `MetricTracker` (e.g. with `EarlyStopping` and `TrueFrameTrainer`) actually correct?**
  _`MetricTracker` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `LightFakeDetect` (e.g. with `_ONNXWrapper` and `LightFakeDetectExporter`) actually correct?**
  _`LightFakeDetect` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Step 1: Claim Extraction (NLP/Rule-based)     Extracts factual assertions.`, `Step 2: Claim Type Classification`, `Step 3: Temporal & Context Check (CRITICAL)     Compares media creation age wit` to the rest of the system?**
  _117 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.03 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._