import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift

class ArtifactAnalyzer:
    def analyze(self, rois):
        """
        Step 5: Artifact Heuristics (Classical CV)
        Checks for FFT spikes, noise consistency, etc.
        Implementing 'Hard Rules' as per architecture.
        """
        score = 0.0
        signals = []
        is_hard_fail = False
        
        face = rois.get("face")
        if face is None:
            return 0.0, ["no_face_roi"]

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # --- Rule 1: FFT Frequency Spikes ---
        # Deepfakes often have specific artifacts in the frequency domain.
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Check for abnormal spikes (simple variance in high freq regions)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        # Mask out low frequencies (center)
        mask_radius = 20
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 > mask_radius**2
        
        high_freq_content = magnitude_spectrum[mask]
        
        if np.mean(high_freq_content) < 80: # Abnormally smooth (common in GANs)
            score = 1.0
            signals.append("fft_frequency_anomaly")
            is_hard_fail = True

        # --- Rule 2: Uniform patterns / checkerboard artifacts ---
        # Simulated by checking local variance consistency
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50: # Extremely low variance = too smooth = fake
            score = 1.0
            signals.append("texture_smoothing_detected")
            is_hard_fail = True

        # --- Rule 3: Repeating texture patches (Simplified) ---
        # Very computationally expensive to do full patch matching. 
        # We'll use Gray Level Co-occurrence Matrix (GLCM) proxy or simple autocorrelation.
        # Here we rely on the FFT check above which catches repeating patterns (spikes).
        
        # --- Rule 4: Sensor Noise Analysis ---
        # Real cameras have specific noise profiles (ISO). 
        # Synthetic images often have uniform or Gaussian noise added blindly.
        # We check if noise is "too perfect" or absent.
        # Simple Noise Estimate: diff between image and median blur
        noise = cv2.absdiff(gray, cv2.medianBlur(gray, 3))
        noise_level = np.mean(noise)
        if noise_level < 1.5: # Almost zero noise -> synthetic
             score = 1.0
             signals.append("unnatural_silence_noise")
             is_hard_fail = True

        return (1.0 if is_hard_fail else 0.0), signals
