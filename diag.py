import os
model_dir = os.path.join("verified-stream", "ai_service", "models")
onnx_path = os.path.join(model_dir, "lightfakedetect.onnx")
print(f"Checking: {onnx_path}")
print(f"Exists: {os.path.exists(onnx_path)}")
print(f"Contents of {model_dir}:")
if os.path.exists(model_dir):
    print(os.listdir(model_dir))
else:
    print("Model dir missing")
