import os

# --- ENSEMBLE MODEL FUSION WEIGHTS ---
WEIGHT_MODEL = 0.40
WEIGHT_ARTIFACT = 0.20
WEIGHT_TEMPORAL = 0.15
WEIGHT_METADATA = 0.10
WEIGHT_COMPRESSION = 0.15

# --- DECISION THRESHOLDS (3-tier) ---
# < THRESHOLD_APPROVE  → APPROVED (real)
# THRESHOLD_APPROVE to THRESHOLD_REJECT → UNDER_REVIEW
# >= THRESHOLD_REJECT  → REJECTED (deepfake)
THRESHOLD_APPROVE = 0.60
THRESHOLD_REJECT = 0.80

# --- MULTI-PATCH CONFIG ---
PATCH_COUNT = 10

# --- PERFORMANCE CONFIG ---
MAX_FRAMES_TO_PROCESS = 32
FRAME_SAMPLE_RATE = 1

# --- PATHS ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "efficientnet_b0_v1.onnx")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# --- TERTIARY REVIEW (Swin-L) ---
SWIN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "swin_v2_l_deepfake.onnx")
SWIN_FAKE_INDEX = int(os.getenv("SWIN_FAKE_INDEX", "1"))
SWIN_HIGH_RISK_THRESHOLD = float(os.getenv("SWIN_HIGH_RISK_THRESHOLD", "0.85"))
SWIN_ELEVATED_RISK_THRESHOLD = float(os.getenv("SWIN_ELEVATED_RISK_THRESHOLD", "0.70"))
