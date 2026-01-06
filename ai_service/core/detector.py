import cv2
import numpy as np

class FaceAnalyzer:
    def __init__(self):
        # Use simple Haar Cascades for CPU-friendly face detection
        # In a real setup, MediaPipe CPU is better, but Haar is built-in to OpenCV (easiest for now)
        self.face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)

    def get_face_roi(self, frame):
        """
        Step 3: Face & Region Extraction
        Returns crop of the face and key regions (eyes, mouth)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, {}

        # Take the largest face
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        
        face_crop = frame[y:y+h, x:x+w]
        
        # Heuristic ROIs (relative to face box)
        rois = {
            "face": face_crop,
            "eyes": face_crop[int(h*0.2):int(h*0.45), :],
            "mouth": face_crop[int(h*0.65):int(h*0.9), int(w*0.2):int(w*0.8)]
        }
        
        return face_crop, rois
