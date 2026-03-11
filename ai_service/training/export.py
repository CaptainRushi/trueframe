"""
TrueFrame Reels — Model Export & Deployment
=============================================
Export trained model to ONNX and TorchScript for production inference.
Integrates with the existing TrueFrame ai_service pipeline.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn

from training.config import CONFIG, EXPORT_DIR, CHECKPOINT_DIR
from training.model import TrueFrameReelsDetector

logger = logging.getLogger("trueframe.export")


class ReelsModelExporter:
    """Export trained EfficientNet-B4+LSTM to ONNX and TorchScript."""

    def __init__(self, checkpoint_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt_path = checkpoint_path or os.path.join(
            CHECKPOINT_DIR, "best_model.pth"
        )

        logger.info(f"Loading checkpoint: {ckpt_path}")
        self.model = TrueFrameReelsDetector()
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)

    def export_onnx(self, output_path: str = None) -> str:
        """Export to ONNX format for cross-platform inference."""
        output_path = output_path or os.path.join(
            EXPORT_DIR, "trueframe_reels_detector.onnx"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        seq_len = CONFIG.frames.SEQUENCE_LENGTH
        h, w = CONFIG.frames.FRAME_SIZE
        dummy = torch.randn(1, seq_len, 3, h, w, device=self.device)

        # Wrapper to output only deepfake probability
        class ONNXWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                return out["deepfake_prob"]

        wrapper = ONNXWrapper(self.model)
        wrapper.eval()

        dynamic_axes = None
        if CONFIG.deployment.ONNX_DYNAMIC_AXES:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

        torch.onnx.export(
            wrapper, dummy, output_path,
            opset_version=CONFIG.deployment.ONNX_OPSET_VERSION,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"ONNX exported: {output_path} ({size_mb:.1f} MB)")
        return output_path

    def export_torchscript(self, output_path: str = None) -> str:
        """Export to TorchScript for PyTorch inference."""
        output_path = output_path or os.path.join(
            EXPORT_DIR, "trueframe_reels_detector.pt"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        seq_len = CONFIG.frames.SEQUENCE_LENGTH
        h, w = CONFIG.frames.FRAME_SIZE
        dummy = torch.randn(1, seq_len, 3, h, w, device=self.device)

        traced = torch.jit.trace(self.model, dummy)
        traced.save(output_path)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"TorchScript exported: {output_path} ({size_mb:.1f} MB)")
        return output_path

    def export_metadata(self, output_path: str = None):
        """Save model metadata for the inference pipeline."""
        output_path = output_path or os.path.join(
            EXPORT_DIR, "model_metadata.json"
        )
        metadata = {
            "model_name": "TrueFrame Reels Deepfake Detector",
            "version": "1.0.0",
            "architecture": f"{CONFIG.model.BACKBONE}+LSTM",
            "input_shape": [
                CONFIG.frames.SEQUENCE_LENGTH, 3,
                *CONFIG.frames.FRAME_SIZE,
            ],
            "output": "deepfake_probability (0-1)",
            "thresholds": {
                "real": f"< {CONFIG.deployment.THRESHOLD_REAL}",
                "under_review": (
                    f"{CONFIG.deployment.THRESHOLD_REAL} - "
                    f"{CONFIG.deployment.THRESHOLD_REVIEW}"
                ),
                "deepfake": f">= {CONFIG.deployment.THRESHOLD_DEEPFAKE}",
            },
            "normalization": {
                "mean": list(CONFIG.augmentation.NORMALIZE_MEAN),
                "std": list(CONFIG.augmentation.NORMALIZE_STD),
            },
        }
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {output_path}")

    def export_all(self):
        """Export ONNX + TorchScript + metadata."""
        paths = {}
        if CONFIG.deployment.EXPORT_ONNX:
            paths["onnx"] = self.export_onnx()
        if CONFIG.deployment.EXPORT_TORCHSCRIPT:
            paths["torchscript"] = self.export_torchscript()
        self.export_metadata()
        return paths


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export TrueFrame model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--format", choices=["onnx", "torchscript", "all"],
                        default="all")
    args = parser.parse_args()

    exporter = ReelsModelExporter(checkpoint_path=args.checkpoint)
    if args.format == "onnx":
        exporter.export_onnx()
    elif args.format == "torchscript":
        exporter.export_torchscript()
    else:
        exporter.export_all()
