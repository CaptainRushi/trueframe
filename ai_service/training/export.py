"""
TrueFrame — Model Export & Deployment
======================================
Export trained LightFakeDetect (MobileNetV2 + CBAM + GRU) to ONNX
for production inference without requiring PyTorch at runtime.

Output: ai_service/models/lightfakedetect.onnx
  - Input:  (batch, seq_len, 3, 224, 224)  — sequence of normalized face crops
  - Output: (batch,)                        — P(fake) for each sequence

Usage:
    python -m training.export                          # export best checkpoint
    python -m training.export --checkpoint path/to.pth
    python -m training.export --seq-len 10             # override sequence length

After export, place the .onnx file in ai_service/models/ and the inference
pipeline in main.py / reel_inference.py will auto-detect and use it.
"""

import os
import sys
import json
import logging

import numpy as np
import torch
import torch.nn as nn

# Ensure the parent directory is on sys.path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import CONFIG, EXPORT_DIR, CHECKPOINT_DIR
from training.model import LightFakeDetect

logger = logging.getLogger("trueframe.export")


# ─────────────────── ONNX WRAPPER ────────────────────

class _ONNXWrapper(nn.Module):
    """Wraps LightFakeDetect to output only P(fake) for ONNX export."""
    def __init__(self, model: LightFakeDetect):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["deepfake_prob"]   # (batch,)


# ─────────────────── EXPORTER ────────────────────────

class LightFakeDetectExporter:
    """
    Export a trained LightFakeDetect model to ONNX + metadata.

    The exported ONNX file supports:
    - Dynamic batch size (batch dimension is dynamic)
    - Dynamic sequence length (seq_len dimension is dynamic)
      → Handles different video lengths at inference time

    GRU note: PyTorch's GRU is exportable to ONNX opset >= 9.
    Dynamic axes cover variable-length sequences.
    """

    def __init__(self, checkpoint_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)

        ckpt_path = checkpoint_path or os.path.join(CHECKPOINT_DIR, "best_model.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Train the model first: python -m training.trainer"
            )

        logger.info(f"Loading checkpoint: {ckpt_path}")
        self.model = LightFakeDetect()
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)

        logger.info("Model loaded successfully")
        logger.info(f"Training epoch: {ckpt.get('epoch', 'unknown')}")
        logger.info(f"Best metric:    {ckpt.get('best_metric', 'unknown')}")

    def export_onnx(
        self,
        output_path: str = None,
        seq_len: int = None,
        opset: int = 17,
    ) -> str:
        """
        Export model to ONNX with dynamic batch and sequence length axes.

        Args:
            output_path: Where to save the .onnx file
            seq_len:     Sequence length for dummy input (default: config value)
            opset:       ONNX opset version (>= 9 required for GRU)

        Returns:
            Path to the exported ONNX file
        """
        output_path = output_path or os.path.join(EXPORT_DIR, "lightfakedetect.onnx")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        seq_len = seq_len or CONFIG.frames.SEQUENCE_LENGTH
        h, w = CONFIG.frames.FRAME_SIZE

        # Dummy input: batch=1, seq_len frames, 3×224×224
        dummy = torch.randn(1, seq_len, 3, h, w, device=self.device)

        wrapper = _ONNXWrapper(self.model)
        wrapper.eval()

        # Dynamic axes: batch size AND sequence length are variable at inference
        dynamic_axes = {
            "frames":       {0: "batch_size", 1: "seq_len"},
            "deepfake_prob":{0: "batch_size"},
        }

        logger.info(f"Exporting ONNX to: {output_path}")
        logger.info(f"  Dummy input shape: {list(dummy.shape)}")
        logger.info(f"  Opset version: {opset}")

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                output_path,
                opset_version=opset,
                input_names=["frames"],
                output_names=["deepfake_prob"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"  ✓ Exported ({size_mb:.1f} MB)")

        # Validate the exported ONNX
        self._validate_onnx(output_path, dummy)

        return output_path

    def _validate_onnx(self, onnx_path: str, dummy_input: torch.Tensor):
        """Run a quick inference check on the exported ONNX model."""
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            ort_input = {sess.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_out = sess.run(None, ort_input)
            prob = float(ort_out[0][0])
            logger.info(f"  ✓ ONNX validation passed (dummy P(fake)={prob:.4f})")
        except ImportError:
            logger.warning("  onnxruntime not installed — skipping ONNX validation")
        except Exception as e:
            logger.error(f"  ONNX validation failed: {e}")

    def export_metadata(self, output_path: str = None) -> str:
        """Save model metadata JSON alongside the ONNX file."""
        output_path = output_path or os.path.join(EXPORT_DIR, "lightfakedetect_metadata.json")
        metadata = {
            "model_name": "LightFakeDetect",
            "version": "2.0.0",
            "architecture": "MobileNetV2 + CBAM + GRU",
            "paper": "LightFakeDetect (MDPI Applied Sciences, 2024)",
            "backbone": CONFIG.model.BACKBONE,
            "gru_hidden_dim": CONFIG.model.GRU_HIDDEN_DIM,
            "gru_num_layers": CONFIG.model.GRU_NUM_LAYERS,
            "use_cbam": CONFIG.model.USE_CBAM,
            "input_shape": ["batch", "seq_len", 3, *CONFIG.frames.FRAME_SIZE],
            "output": "deepfake_probability in [0, 1]",
            "threshold_reject": 0.80,
            "normalization": {
                "mean": list(CONFIG.augmentation.NORMALIZE_MEAN),
                "std":  list(CONFIG.augmentation.NORMALIZE_STD),
            },
        }
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✓ Metadata saved: {output_path}")
        return output_path

    def export_all(self, seq_len: int = None) -> dict:
        """Export ONNX + metadata. Returns paths dict."""
        paths = {}
        paths["onnx"]     = self.export_onnx(seq_len=seq_len)
        paths["metadata"] = self.export_metadata()
        logger.info("\nExport complete!")
        logger.info(f"  ONNX:     {paths['onnx']}")
        logger.info(f"  Metadata: {paths['metadata']}")
        logger.info("\nNext: copy the .onnx file to ai_service/models/lightfakedetect.onnx")
        return paths


# ─────────────────── CLI ─────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Export LightFakeDetect model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (default: training/checkpoints/best_model.pth)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path (default: ai_service/models/lightfakedetect.onnx)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length for dummy input (default: config SEQUENCE_LENGTH)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    exporter = LightFakeDetectExporter(checkpoint_path=args.checkpoint)
    exporter.export_onnx(output_path=args.output, seq_len=args.seq_len, opset=args.opset)
    exporter.export_metadata()
