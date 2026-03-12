import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False


class FaceAnalyzer:
    def __init__(self):
        if _HAS_MEDIAPIPE:
            try:
                import mediapipe.python.solutions as mp_solutions
                self.mp_face = mp_solutions.face_detection
            except ImportError:
                self.mp_face = mp.solutions.face_detection
            self.detector = self.mp_face.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
        else:
            # Fallback to Haar Cascade if mediapipe not installed
            self.detector = None
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

    def get_face_roi(self, frame):
        """
        Face & Region Extraction using MediaPipe (or Haar Cascade fallback).
        Returns crop of the face and key regions (eyes, mouth).
        """
        if _HAS_MEDIAPIPE and self.detector is not None:
            return self._detect_mediapipe(frame)
        return self._detect_haar(frame)

    def _detect_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None, {}

        best = max(results.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box
        h, w = frame.shape[:2]

        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = min(w - x, int(bbox.width * w))
        bh = min(h - y, int(bbox.height * h))

        if bw <= 0 or bh <= 0:
            return None, {}

        face_crop = frame[y:y + bh, x:x + bw]
        return self._extract_rois(face_crop, bw, bh)

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None, {}

        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_crop = frame[y:y + h, x:x + w]
        return self._extract_rois(face_crop, w, h)

    def _extract_rois(self, face_crop, w, h):
        rois = {
            "face": face_crop,
            "eyes": face_crop[int(h * 0.2):int(h * 0.45), :],
            "mouth": face_crop[int(h * 0.65):int(h * 0.9), int(w * 0.2):int(w * 0.8)]
        }
        return face_crop, rois
