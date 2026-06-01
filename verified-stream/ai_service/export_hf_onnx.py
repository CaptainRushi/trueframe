"""
Export HuggingFace deepfake detector to ONNX for production inference.

Exports dima806/deepfake_vs_real_image_detection (trained on FaceForensics++)
to ONNX format so main.py can use fast ONNX inference instead of
pure signal analysis.

Usage:
    python export_hf_onnx.py
"""

import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("export_hf_onnx")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def export_hf_to_onnx(model_name="dima806/deepfake_vs_real_image_detection", output_name=None):
    """
    Export a HuggingFace image classification model to ONNX.
    Output: models/{output_name}.onnx
    """
    if output_name is None:
        base = model_name.split("/")[-1]
        output_name = base.replace("-", "_")
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        logger.error("torch/transformers not installed. Install with: pip install torch transformers")
        return False

    logger.info(f"Loading HuggingFace model: {model_name}")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

    # Determine fake label index
    KNOWN_FAKE_INDICES = {
        "dima806/deepfake_vs_real_image_detection": 1,
        "prithivMLmods/Deep-Fake-Detector-v2-Model": 1,
    }
    if model_name in KNOWN_FAKE_INDICES:
        fake_idx = KNOWN_FAKE_INDICES[model_name]
    else:
        detected_idx = None
        for idx, label in model.config.id2label.items():
            if "fake" in label.lower() or "forged" in label.lower() or "manipulat" in label.lower():
                detected_idx = int(idx)
        fake_idx = detected_idx if detected_idx is not None else 0

    logger.info(f"Fake label index: {fake_idx}, id2label: {model.config.id2label}")
    logger.info(f"Model type: {model.config.model_type}")

    device = torch.device("cpu")
    model.to(device)

    # Create a wrapper that outputs only P(fake)
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, base_model, fake_idx):
            super().__init__()
            self.base_model = base_model
            self.fake_idx = fake_idx

        def forward(self, pixel_values):
            outputs = self.base_model(pixel_values)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            return probs[:, self.fake_idx]

    wrapper = ONNXWrapper(model, fake_idx).to(device)

    # Get expected input shape from the processor
    if hasattr(processor, "size"):
        size = processor.size
        if isinstance(size, dict):
            h = w = size.get("shortest_edge", 224)
        elif isinstance(size, (list, tuple)):
            h, w = size[:2]
        else:
            h = w = int(size)
    else:
        h = w = 224

    dummy_input = torch.randn(1, 3, h, w)

    # Export to ONNX with dynamic batch size
    onnx_path = os.path.join(MODELS_DIR, f"{output_name}.onnx")
    logger.info(f"Exporting ONNX to: {onnx_path}")
    logger.info(f"  Input shape: (1, 3, {h}, {w})")

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_input,
                onnx_path,
                opset_version=17,
                input_names=["pixel_values"],
                output_names=["deepfake_prob"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "deepfake_prob": {0: "batch_size"},
                },
                do_constant_folding=True,
            )
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"  Exported ({size_mb:.1f} MB)")

    # Validate
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_input = {sess.get_inputs()[0].name: dummy_input.numpy()}
        ort_out = sess.run(None, ort_input)
        prob = float(ort_out[0][0])
        logger.info(f"  Validation: dummy P(fake) = {prob:.4f}")
    except Exception as e:
        logger.warning(f"  Validation skipped: {e}")

    # Save metadata
    meta = {
        "model_name": model_name,
        "architecture": model.config.model_type,
        "input_shape": [1, 3, h, w],
        "output": "deepfake_probability in [0, 1]",
        "fake_label_index": fake_idx,
        "mean": processor.image_mean if hasattr(processor, "image_mean") else [0.5, 0.5, 0.5],
        "std": processor.image_std if hasattr(processor, "image_std") else [0.5, 0.5, 0.5],
    }
    meta_path = os.path.join(MODELS_DIR, f"{output_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata saved to {meta_path}")

    logger.info("Done! ONNX model ready for inference.")
    return True


def export_gan_to_onnx(model_name="prithivMLmods/Deep-Fake-Detector-v2-Model"):
    """Export GAN-specific detector to ONNX."""
    return export_hf_to_onnx(model_name, output_name="deep_fake_detector_v2_model")

if __name__ == "__main__":
    success1 = export_hf_to_onnx(output_name="hf_deepfake_detector")
    success2 = export_gan_to_onnx()
    if success1:
        logger.info("Primary deepfake detector exported successfully.")
    if success2:
        logger.info("GAN detector exported successfully.")
    sys.exit(0 if (success1 or success2) else 1)
