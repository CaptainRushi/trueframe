import cv2
import numpy as np
import random
from config import MAX_FRAMES_TO_PROCESS


class FrameExtractor:
    def extract_frames(self, file_path):
        """
        Smart Frame Sampling.
        Extract 1 FPS + random jitter, max 32 frames.
        For long videos, sample uniformly across entire duration.
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return [], []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        frames = []
        timestamps = []

        if duration <= 0:
            cap.release()
            return [], []

        # For long videos (> MAX_FRAMES seconds), sample uniformly across full duration
        if duration > MAX_FRAMES_TO_PROCESS:
            sample_times = np.linspace(0, duration - 0.5, MAX_FRAMES_TO_PROCESS)
        else:
            sample_times = range(int(duration) + 1)

        for sec in sample_times:
            if len(frames) >= MAX_FRAMES_TO_PROCESS:
                break

            # Add random jitter to avoid I-frame bias
            jitter = random.uniform(-0.3, 0.3)
            target_time = max(0, float(sec) + jitter)

            cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
            ret, frame = cap.read()

            if ret:
                frames.append(frame)
                timestamps.append(target_time)

        cap.release()
        return frames, timestamps
