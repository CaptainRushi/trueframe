import os
import subprocess
import json

class MetadataScanner:
    def scan(self, file_path):
        """
        Step 1: Fast Pre-check (CPU, <50ms)
        Returns a probability score (0.0 - 1.0) and a list of signals.
        """
        score = 0.0
        signals = []

        # 1. Check file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        # 2. Simulate EXIF/Metadata extraction (using ffprobe usually, mocking locally)
        # Real implementation would use: ffprobe -v quiet -print_format json -show_format -show_streams file_path
        
        # Mock logic for "Suspicious encoding chains"
        # In production, check for 'Lavf58.29.100' or other common ffmpeg default tags often used by GANs
        
        # Mocking a check:
        file_size = os.path.getsize(file_path)
        if file_size < 1000: # Suspiciously small
             score += 0.2
             signals.append("abnormal_file_size")

        # 3. Check for specific known bad signatures (Mock)
        # e.g., missing specific sensor noise metadata
        has_sensor_data = False # Mock
        if not has_sensor_data:
            score += 0.1
            signals.append("missing_sensor_metadata")

        return max(0.0, min(1.0, score)), signals
