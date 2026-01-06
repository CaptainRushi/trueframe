import os

# --- EFFICIENTNET FUSION WEIGHTS ---
WEIGHT_MODEL = 0.45
WEIGHT_ARTIFACT = 0.25
WEIGHT_TEMPORAL = 0.20
WEIGHT_METADATA = 0.10

# --- DECISION THRESHOLDS ---
# final_score >= 0.60 -> REJECT (Deepfake)
# 0.40 – 0.59       → REJECT (Unsafe)
# < 0.40            → APPROVE
THRESHOLD_REJECT = 0.60
THRESHOLD_UNSAFE = 0.40

# --- PERFORMANCE CONFIG ---
MAX_FRAMES = 32
FRAME_SAMPLE_RATE = 1  # 1 FPS

# --- PATHS ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "efficientnet_b0_v1.onnx")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
