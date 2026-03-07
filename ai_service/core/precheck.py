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

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        # 1. File size check
        file_size = os.path.getsize(file_path)
        if file_size < 1000:
            score += 0.2
            signals.append("abnormal_file_size")

        # 2. Real metadata extraction via ffprobe
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                 '-show_format', '-show_streams', file_path],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                metadata = json.loads(result.stdout)

                # Check for known synthetic encoder signatures
                encoder = metadata.get('format', {}).get('tags', {}).get('encoder', '')
                if encoder and ('Lavf' in encoder or 'lavf' in encoder):
                    score += 0.3
                    signals.append("suspicious_encoder_signature")

                # Check for missing camera metadata
                streams = metadata.get('streams', [])
                has_camera_data = any(
                    s.get('tags', {}).get('creation_time') for s in streams
                )
                if not has_camera_data:
                    score += 0.1
                    signals.append("missing_camera_metadata")
            else:
                score += 0.1
                signals.append("metadata_extraction_failed")

        except FileNotFoundError:
            # ffprobe not installed — fall back to basic checks
            score += 0.1
            signals.append("missing_sensor_metadata")
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            score += 0.1
            signals.append("metadata_extraction_failed")

        return max(0.0, min(1.0, score)), signals
