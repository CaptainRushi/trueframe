"""
TrueFrame Reels — Video Deepfake Detector Test Suite
===================================================
Automated unit and integration testing for `reel_inference.py`.

Tests:
1. Signal calculation functions (mocked/synthetic frames)
2. Verdict boundary conditions (binary APPROVED/REJECTED mapping)
3. Integration via CLI execution on real and invalid video files
"""

import os
import sys
import json
import unittest
import subprocess
import numpy as np
import cv2

# Add parent directory to path to import reel_inference
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training import reel_inference


class TestVideoDetector(unittest.TestCase):
    def setUp(self):
        # Create dummy frame matrices for testing
        self.height, self.width = 224, 224
        # Clean frame
        self.clean_frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        
        # Blocky frame (with high DCT compression/grid periodic structure)
        self.blocky_frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        # Create a grid-like artifact with 8x8 blocks
        for y in range(0, self.height, 8):
            for x in range(0, self.width, 8):
                if (x // 8 + y // 8) % 2 == 0:
                    self.blocky_frame[y:y+8, x:x+8] = 200

    def test_verdict_boundaries(self):
        """Verify binary classification: score >= 0.60 rejected, score < 0.60 approved."""
        # 1. Under-approve score (e.g. 0.35) -> APPROVED
        res_low = reel_inference._build_result(0.35, [], 0)
        self.assertEqual(res_low["verdict"], "APPROVED")
        self.assertEqual(res_low["confidence"], "HIGH")
        
        # 2. Borderline score (e.g. 0.67) -> REJECTED (Previously APPROVED under 0.80 threshold)
        res_borderline = reel_inference._build_result(0.67, [], 0)
        self.assertEqual(res_borderline["verdict"], "REJECTED")
        self.assertEqual(res_borderline["confidence"], "HIGH")
        self.assertIn("deepfake_detected", res_borderline["signals"])
        
        # 3. High score (e.g. 0.85) -> REJECTED
        res_high = reel_inference._build_result(0.85, [], 0)
        self.assertEqual(res_high["verdict"], "REJECTED")
        self.assertEqual(res_high["confidence"], "HIGH")
        self.assertIn("deepfake_detected", res_high["signals"])

    def test_temporal_flicker_signal(self):
        """Verify temporal flicker signal scores flickering vs static inputs."""
        # Clean/identical frames -> low flicker score
        static_frames = [self.clean_frame.copy() for _ in range(5)]
        score, triggered = reel_inference.signal_temporal_flicker(static_frames)
        self.assertLess(score, 0.2)
        self.assertFalse(triggered)

        # Flickering frames -> high flicker score
        flicker_frames = []
        for i in range(5):
            val = 50 if i % 2 == 0 else 220
            flicker_frames.append(np.ones((self.height, self.width, 3), dtype=np.uint8) * val)
        score, triggered = reel_inference.signal_temporal_flicker(flicker_frames)
        self.assertGreater(score, 0.7)
        self.assertTrue(triggered)

    def test_color_consistency_signal(self):
        """Verify color consistency signal detects hue fluctuations."""
        # Constant color -> low score
        constant_frames = []
        for _ in range(5):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :, 0] = 50  # Hue angle BGR -> HSV
            constant_frames.append(frame)
        score, triggered = reel_inference.signal_color_consistency(constant_frames)
        self.assertLess(score, 0.2)
        self.assertFalse(triggered)

        # Fluctuating colors -> high score
        color_frames = []
        for i in range(5):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if i % 2 == 0:
                frame[:, :, 2] = 255  # Red
            else:
                frame[:, :, 0] = 255  # Blue
            color_frames.append(frame)
        score, triggered = reel_inference.signal_color_consistency(color_frames)
        self.assertGreater(score, 0.5)
        self.assertTrue(triggered)

    def test_block_artifacts_signal(self):
        """Verify that block artifacts detection scoring responds to grid blocks."""
        # Smooth/clean frames
        clean_frames = [self.clean_frame for _ in range(3)]
        score_clean, triggered_clean = reel_inference.signal_block_artifacts(clean_frames)
        self.assertLess(score_clean, 0.3)
        
        # Grid block frames
        blocky_frames = [self.blocky_frame for _ in range(3)]
        score_block, triggered_block = reel_inference.signal_block_artifacts(blocky_frames)
        self.assertGreater(score_block, score_clean)

    def test_cli_execution_real_video(self):
        """Verify CLI execution on the test video file returns expected schema and verdict."""
        # Locate the test video file in parent directory
        test_video = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "176527-855920754_medium.mp4"))
        
        if not os.path.exists(test_video):
            self.skipTest(f"Test video not found at: {test_video}")
            
        inference_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "training", "reel_inference.py"))
        
        # Run inference script as a subprocess
        cmd = [sys.executable, inference_script, test_video]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        self.assertEqual(proc.returncode, 0)
        
        # Parse output JSON
        try:
            output = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            self.fail(f"Output is not a valid JSON object. Got:\n{proc.stdout}")
            
        # Verify required keys are present
        required_keys = [
            "model", "model_score", "artifact_score", "temporal_score",
            "metadata_score", "compression_score", "final_score",
            "deepfake_probability", "authenticity_score", "verdict",
            "confidence", "inference_time_ms", "signals"
        ]
        for key in required_keys:
            self.assertIn(key, output)
            
        # The test video is authentic and should be APPROVED under binary mapping with 0.60 threshold
        self.assertEqual(output["verdict"], "APPROVED")
        self.assertEqual(output["confidence"], "HIGH")

    def test_cli_execution_invalid_file(self):
        """Verify CLI execution handles non-existent video path by returning insufficient_frames."""
        inference_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "training", "reel_inference.py"))
        cmd = [sys.executable, inference_script, "non_existent_file.mp4"]
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        # Non-existent file results in empty frames array which completes with code 0 and insufficient_frames signal
        self.assertEqual(proc.returncode, 0)
        
        try:
            output = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            self.fail(f"Output is not a valid JSON. Got:\n{proc.stdout}")
            
        self.assertEqual(output["verdict"], "REJECTED")
        self.assertIn("insufficient_frames", output["signals"])

    def test_cli_execution_no_args(self):
        """Verify CLI execution fails with code 1 and REJECTED when no arguments are provided."""
        inference_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "training", "reel_inference.py"))
        cmd = [sys.executable, inference_script]
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        self.assertEqual(proc.returncode, 1)
        
        try:
            output = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            self.fail(f"Error output is not a valid JSON. Got:\n{proc.stdout}")
            
        self.assertEqual(output["verdict"], "REJECTED")
        self.assertIn("no_file_provided", output["signals"])


if __name__ == "__main__":
    unittest.main()
