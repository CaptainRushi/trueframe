import cv2
import numpy as np
import random
from config import MAX_FRAMES_TO_PROCESS

class FrameExtractor:
    def extract_frames(self, file_path):
        """
        Step 2: Smart Frame Sampling
        Rule: Never scan full video. 
        Extract 1 FPS + random jitter, max 32 frames.
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error opening video file {file_path}")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        frames = []
        timestamps = []
        
        # Strategy: Sample 1 frame per second
        for sec in range(int(duration) + 1):
            if len(frames) >= MAX_FRAMES_TO_PROCESS:
                break
                
            # Add small random jitter to avoid I-frame bias
            jitter = random.uniform(-0.1, 0.1)
            target_time = max(0, sec + jitter)
            
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
                timestamps.append(target_time)
        
        cap.release()
        return frames, timestamps
