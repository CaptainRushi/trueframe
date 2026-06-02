"""
TrueFrame — Enhanced Signal Detectors v2
=========================================
Supplementary signal detectors that plug into main.py's _run_signal_analysis.
Designed to catch deepfake artifacts the v1 signals miss.

New detectors:
  1. DCT coefficient analysis      — GAN/diffusion models produce characteristic DCT distributions
  2. Multi-scale Laplacian texture — GAN oversmoothing detected at 3 spatial scales
  3. Enhanced blending seam        — Gradient direction discontinuity at face borders
  4. Wavelet coefficient analysis  — Haar-like wavelet decomposition for texture pathology
  5. Color histogram consistency   — GAN faces have unnaturally uniform color distributions
"""

import cv2
import numpy as np


# ────────────────────────────────────────────────────────────
# 1. DCT COEFFICIENT ANALYSIS
# ────────────────────────────────────────────────────────────

def signal_dct_artifacts(crops):
    """
    Analyze DCT coefficient distribution to detect GAN/diffusion fingerprints.
    
    GAN-generated faces have:
    - Abnormal energy distribution in mid-frequency DCT coefficients
    - Lower high-frequency energy (oversmoothing)
    - Different coefficient sparsity patterns
    
    Returns (score, triggered) where score in [0, 1].
    """
    if not crops:
        return 0.0, False
    
    scores = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Divide into 8x8 blocks and compute DCT for each
        h, w = gray.shape
        block_size = 8
        dct_coeffs = []
        
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                block_dct = cv2.dct(block)
                dct_coeffs.append(block_dct)
        
        if not dct_coeffs:
            continue
        
        dct_stack = np.stack(dct_coeffs)
        
        # DC coefficient (0,0) normalized by AC coefficients
        dc_coeffs = dct_stack[:, 0, 0]
        ac_sum = np.sum(np.abs(dct_stack[:, 1:, 1:]), axis=(1, 2))
        ac_mean = np.mean(ac_sum)
        
        # Low-frequency ratio: energy in first 3 DCT coefficients vs total
        total_energy = np.sum(np.abs(dct_stack), axis=(1, 2))
        lf_energy = np.sum(np.abs(dct_stack[:, :3, :3]), axis=(1, 2))
        lf_ratio = np.mean(lf_energy / (total_energy + 1e-6))
        
        # High-frequency energy: GAN faces have less HF energy
        hf_energy = np.sum(np.abs(dct_stack[:, 4:, 4:]), axis=(1, 2))
        hf_ratio = np.mean(hf_energy / (total_energy + 1e-6))
        
        # DCT coefficient sparsity: GAN faces have more zero/small coefficients
        threshold = np.mean(np.abs(dct_stack)) * 0.1
        sparsity = np.mean(np.sum(np.abs(dct_stack) < threshold, axis=(1, 2)) / 64.0)
        
        # DC coefficient cross-block variance: GAN faces have unnaturally uniform DC
        dc_cv = np.std(dc_coeffs) / (np.mean(np.abs(dc_coeffs)) + 1e-6)
        
        # Combine metrics into score
        # GAN signature: high LF ratio (>0.42), low HF ratio (<0.08), high sparsity (>0.45), low DC variance
        lf_score = min(1.0, max(0.0, (lf_ratio - 0.35) / 0.20))
        hf_score = min(1.0, max(0.0, (0.12 - hf_ratio) / 0.08))
        sparsity_score = min(1.0, max(0.0, (sparsity - 0.40) / 0.20))
        dc_score = min(1.0, max(0.0, (0.30 - dc_cv) / 0.20))
        
        combined = 0.30 * lf_score + 0.30 * hf_score + 0.20 * sparsity_score + 0.20 * dc_score
        scores.append(combined)
    
    if not scores:
        return 0.0, False
    
    mean_score = float(np.mean(scores))
    triggered = mean_score > 0.35
    return mean_score, triggered


# ────────────────────────────────────────────────────────────
# 2. MULTI-SCALE LAPLACIAN TEXTURE ANALYSIS
# ────────────────────────────────────────────────────────────

def signal_laplacian_pyramid(crops):
    """
    Multi-scale texture analysis using Laplacian pyramid decomposition.
    
    GAN-oversmoothed faces have uniformly low texture energy at ALL scales,
    while real faces have high-frequency detail at fine scales that
    progressively decreases at coarser scales.
    
    Returns (score, triggered).
    """
    if not crops:
        return 0.0, False
    
    scores = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Build Laplacian pyramid (3 levels)
        pyramid = []
        current = gray.copy()
        for _ in range(3):
            blurred = cv2.GaussianBlur(current, (5, 5), 1.5)
            laplacian = current - blurred
            pyramid.append(laplacian)
            current = cv2.resize(blurred, (current.shape[1] // 2, current.shape[0] // 2))
        
        # Compute texture energy at each scale
        energies = []
        for level, lap in enumerate(pyramid):
            energy = float(np.mean(np.abs(lap)))
            energies.append(energy)
        
        # Real faces: energy decreases from fine to coarse, but fine scale has high energy
        # GAN faces: uniformly low energy at all scales
        fine_energy = energies[0] if energies else 0
        coarse_energy = energies[-1] if energies else 0
        
        # Energy ratio (fine/coarse): real faces have high ratio (>3), GAN faces have low ratio (<2)
        energy_ratio = fine_energy / (coarse_energy + 1e-6)
        
        # Fine-scale absolute energy: real faces > 2.5, GAN faces < 2.0
        fine_score = min(1.0, max(0.0, (4.0 - fine_energy) / 3.0)) if fine_energy < 4.0 else 0.0
        
        # Energy uniformity across scales (CV of energies)
        energy_cv = float(np.std(energies)) / (float(np.mean(energies)) + 1e-6)
        uniformity_score = min(1.0, max(0.0, (0.40 - energy_cv) / 0.30))
        
        # Combined: high fine_score OR high uniformity_score indicates GAN
        combined = 0.60 * fine_score + 0.40 * uniformity_score
        scores.append(combined)
    
    if not scores:
        return 0.0, False
    
    mean_score = float(np.mean(scores))
    triggered = mean_score > 0.38
    return mean_score, triggered


# ────────────────────────────────────────────────────────────
# 3. ENHANCED BLENDING EDGE DETECTOR
# ────────────────────────────────────────────────────────────

def signal_enhanced_seam(crops):
    """
    Enhanced face-swap seam detection using gradient direction analysis.
    
    Face-swap creates a boundary where the inserted face is blended into
    the original image. At this boundary, gradient DIRECTIONS are
    discontinuous — the gradient field has a curl-like pattern.
    
    Additionally measures gradient magnitude asymmetry between
    opposite sides of the face border.
    
    Returns (score, triggered).
    """
    if not crops:
        return 0.0, False
    
    scores = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        
        # Compute gradient magnitude and direction
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)  # in radians
        
        # Border region analysis
        bw = max(8, int(min(h, w) * 0.12))
        
        # Extract border pixels as 4 sides
        top = mag[:bw, :]          # (bw, w)
        bottom = mag[-bw:, :]      # (bw, w)
        left = mag[:, :bw]         # (h, bw)
        right = mag[:, -bw:]       # (h, bw)
        
        # Interior (central region)
        interior = mag[bw:-bw, bw:-bw]
        
        if interior.size < 100:
            continue
        
        border_mean = (np.mean(top) + np.mean(bottom) + np.mean(left) + np.mean(right)) / 4.0
        interior_mean = float(np.mean(interior))
        
        # Edge ratio: high if border has more edges than interior (seam)
        edge_ratio = border_mean / (interior_mean + 1e-6)
        
        # Gradient direction consistency at border
        # At a seam, gradient directions flip abruptly
        # Measure: how consistent is the direction within each border strip?
        top_dir = direction[:bw, :]
        bottom_dir = direction[-bw:, :]
        left_dir = direction[:, :bw]
        right_dir = direction[:, -bw:]
        interior_dir = direction[bw:-bw, bw:-bw]
        
        # Direction circular variance (1 - mean resultant length)
        def _dir_variance(dir_region):
            if dir_region.size == 0:
                return 0.0
            angles = dir_region.ravel()
            r = np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2)
            return float(1.0 - r)  # 0 = perfectly aligned, 1 = random
        
        border_dir_var = np.mean([
            _dir_variance(top_dir), _dir_variance(bottom_dir),
            _dir_variance(left_dir), _dir_variance(right_dir)
        ])
        interior_dir_var = _dir_variance(interior_dir)
        
        # At a seam: border has HIGH directional variance (seam interrupts gradients)
        # while interior has LOWER variance (natural texture)
        # Real faces: border and interior have similar directional variance
        dir_asymmetry = max(0.0, border_dir_var - interior_dir_var)
        
        # Also measure: RMS contrast between opposite borders
        # Abrupt transition from face to background creates opposite-edge asymmetry
        top_bottom_diff = abs(float(np.mean(top)) - float(np.mean(bottom))) / (interior_mean + 1e-6)
        left_right_diff = abs(float(np.mean(left)) - float(np.mean(right))) / (interior_mean + 1e-6)
        border_asymmetry = max(top_bottom_diff, left_right_diff)
        
        # Combined score
        edge_deviation = max(0.0, edge_ratio - 1.0)
        seam_score = (
            0.35 * min(1.0, edge_deviation * 3.0) +
            0.35 * min(1.0, dir_asymmetry * 4.0) +
            0.30 * min(1.0, border_asymmetry * 2.0)
        )
        scores.append(seam_score)
    
    if not scores:
        return 0.0, False
    
    mean_score = float(np.mean(scores))
    triggered = mean_score > 0.33
    return mean_score, triggered


# ────────────────────────────────────────────────────────────
# 4. WAVELET COEFFICIENT ANALYSIS  
# ────────────────────────────────────────────────────────────

def signal_wavelet_analysis(crops):
    """
    Haar-like wavelet analysis for GAN texture pathology detection.
    
    Uses simple Haar-like filters (difference of box means) at multiple
    scales to measure:
    - Horizontal texture energy
    - Vertical texture energy
    - Diagonal texture energy
    
    GAN faces have: low diagonal energy (no diagonal edges), 
    high horizontal/vertical energy ratio (grid artifacts),
    and overall lower wavelet energy.
    
    Returns (score, triggered).
    """
    if not crops:
        return 0.0, False
    
    scores = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Haar-like horizontal edges: difference of adjacent column means
        h_kernel = np.array([[-1, 1]], dtype=np.float32)
        v_kernel = np.array([[-1], [1]], dtype=np.float32)
        
        h_edges = cv2.filter2D(gray, cv2.CV_32F, h_kernel)
        v_edges = cv2.filter2D(gray, cv2.CV_32F, v_kernel)
        
        # Diagonal edges: Laplacian-like
        d_kernel = np.array([[1, -1], [-1, 1]], dtype=np.float32)
        d_edges = cv2.filter2D(gray, cv2.CV_32F, d_kernel)
        
        h_energy = float(np.mean(np.abs(h_edges)))
        v_energy = float(np.mean(np.abs(v_edges)))
        d_energy = float(np.mean(np.abs(d_edges)))
        
        total_energy = h_energy + v_energy + d_energy + 1e-6
        
        # Diagonal energy ratio: real faces have d_ratio ~0.25-0.35
        # GAN faces have d_ratio < 0.20 (oversmoothed diagonals)
        d_ratio = d_energy / total_energy
        
        # H/V ratio: real faces are ~1.0-1.5, GAN faces can be > 2.0 (grid artifacts)
        hv_ratio = h_energy / (v_energy + 1e-6)
        
        # Low total energy suggests overall oversmoothing
        low_energy_score = min(1.0, max(0.0, (15.0 - total_energy) / 10.0))
        
        # Low diagonal ratio suggests GAN oversmoothing
        d_ratio_score = min(1.0, max(0.0, (0.22 - d_ratio) / 0.12))
        
        # Abnormally high HV ratio suggests grid artifacts
        hv_score = min(1.0, max(0.0, (hv_ratio - 2.0) / 1.5)) if hv_ratio > 2.0 else 0.0
        
        combined = 0.40 * low_energy_score + 0.40 * d_ratio_score + 0.20 * hv_score
        scores.append(combined)
    
    if not scores:
        return 0.0, False
    
    mean_score = float(np.mean(scores))
    triggered = mean_score > 0.32
    return mean_score, triggered


# ────────────────────────────────────────────────────────────
# 5. COLOR HISTOGRAM ANALYSIS
# ────────────────────────────────────────────────────────────

def signal_color_histogram(crops):
    """
    Analyze color histogram properties for GAN fingerprints.
    
    GAN-generated faces have:
    - More uniform color distributions (lower histogram entropy)
    - Histogram peaks that are too sharp
    - Color channels that are too well-aligned (perfectly correlated)
    
    Returns (score, triggered).
    """
    if not crops:
        return 0.0, False
    
    scores = []
    for crop in crops:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for each channel (64 bins)
        hist_features = []
        for c in range(3):
            hist = cv2.calcHist([hsv], [c], None, [64], [0, 256])
            hist = hist.flatten() + 1e-6
            hist = hist / hist.sum()  # normalize
            hist_features.append(hist)
        
        # Histogram entropy: lower for GAN faces
        entropies = []
        for hist in hist_features:
            entropy = float(-np.sum(hist * np.log2(hist)))
            entropies.append(entropy)
        
        mean_entropy = float(np.mean(entropies))
        
        # Cross-channel histogram correlation
        corr_hs = float(np.corrcoef(hist_features[0], hist_features[1])[0, 1])
        corr_hv = float(np.corrcoef(hist_features[0], hist_features[2])[0, 1])
        corr_sv = float(np.corrcoef(hist_features[1], hist_features[2])[0, 1])
        mean_corr = float(np.mean([corr_hs, corr_hv, corr_sv]))
        
        # GAN faces: high entropy (>4.5 means natural), low means unnatural uniformity
        entropy_score = min(1.0, max(0.0, (4.5 - mean_entropy) / 1.0))
        
        # GAN faces: high cross-channel correlation (>0.95 = unnatural agreement)
        corr_score = min(1.0, max(0.0, (mean_corr - 0.85) / 0.12))
        
        # Histogram peak sharpness: GAN faces have very sharp peaks
        peak_sharpness = 0.0
        for hist in hist_features:
            peak_val = hist.max()
            peak_sharpness += peak_val / hist.mean()
        peak_sharpness /= 3.0
        peak_score = min(1.0, max(0.0, (peak_sharpness - 20.0) / 30.0))
        
        combined = 0.40 * entropy_score + 0.35 * corr_score + 0.25 * peak_score
        scores.append(combined)
    
    if not scores:
        return 0.0, False
    
    mean_score = float(np.mean(scores))
    triggered = mean_score > 0.38
    return mean_score, triggered


# ────────────────────────────────────────────────────────────
# BATCH RUNNER
# ────────────────────────────────────────────────────────────

def run_all_v2_signals(crops, frames):
    """Run all v2 signal detectors and return raw scores dict."""
    raw = {}
    
    # Face-level detectors (require face crops)
    if crops:
        dct_score, dct_trig = signal_dct_artifacts(crops)
        raw["dct_artifacts"] = dct_score
        
        lap_score, lap_trig = signal_laplacian_pyramid(crops)
        raw["laplacian_pyramid"] = lap_score
        
        seam_score, seam_trig = signal_enhanced_seam(crops)
        raw["enhanced_seam"] = seam_score
        
        wavelet_score, wavelet_trig = signal_wavelet_analysis(crops)
        raw["wavelet"] = wavelet_score
        
        hist_score, hist_trig = signal_color_histogram(crops)
        raw["color_histogram"] = hist_score
    else:
        for k in ("dct_artifacts", "laplacian_pyramid", "enhanced_seam", "wavelet", "color_histogram"):
            raw[k] = 0.0
    
    # Frame-level detectors (work on full frames too)
    
    return raw
