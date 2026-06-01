"""
TrueFrame Hard Deepfake Test Suite — Extreme Cases
====================================================
Tests the detector against the hardest known deepfake scenarios:
  - High-quality lip-sync reels (Wav2Lip, SadTalker)  
  - Face-swap videos (DeepFaceLab, FaceSwap)
  - StyleGAN3 / ThisPersonDoesNotExist images
  - PhotoReal Midjourney / DALL-E portraits
  - Borderline cases (lightly-edited real faces)

Uses publicly available deepfake samples from:
  - FaceForensics++ test set clips
  - DFDC challenge public samples  
  - StyleGAN samples from thispersondoesnotexist.com
  - DeepFake Detection Challenge dataset snippets

Usage:
    python extreme_deepfake_test.py
    python extreme_deepfake_test.py --download   # download test samples first
    python extreme_deepfake_test.py --video-only  # only test videos/reels
"""

import os
import sys
import json
import time
import subprocess
import argparse
import urllib.request
import hashlib

# Force UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Paths ─────────────────────────────────────────────
DETECTOR = "verified-stream/ai_service/main.py"
TEST_DIR  = "test_assets/extreme"

# ── ANSI ──────────────────────────────────────────────
def _c(code, t):
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        return t
    return f"\033[{code}m{t}\033[0m"

GREEN  = lambda t: _c("92", t)
RED    = lambda t: _c("91", t)
YELLOW = lambda t: _c("93", t)
CYAN   = lambda t: _c("96", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)

# ─────────────────────────────────────────────────────
# EXTREME TEST MANIFEST
# Each entry: file path (relative), label, category, difficulty
# difficulty: "medium", "hard", "extreme"
# ─────────────────────────────────────────────────────

# Public domain deepfake samples for testing
# These are sourced from:
# - FaceForensics++ public test samples (c23 compression)
# - StyleGAN3 outputs (no real person)  
# - DFDC public challenge samples
EXTREME_TESTS = [

    # ── CATEGORY 1: Real videos (must pass as APPROVED) ──────────────────────
    # Test our real video in the project
    {
        "file": "176527-855920754_medium.mp4",
        "label": "real",
        "category": "real_video_stock",
        "difficulty": "medium",
        "note": "Stock footage reel — should be APPROVED"
    },

    # ── CATEGORY 2: Deepfake IMAGES — we can test these now ──────────────────
    # StyleGAN faces (if downloaded)
    {
        "file": f"{TEST_DIR}/stylegan/extreme_f01.jpg",
        "label": "fake",
        "category": "stylegan3_face",
        "difficulty": "hard",
        "note": "High-quality StyleGAN3 face — no real person exists"
    },
    {
        "file": f"{TEST_DIR}/stylegan/extreme_f02.jpg",
        "label": "fake",
        "category": "stylegan3_face",
        "difficulty": "hard",
        "note": "StyleGAN3 high-res (1024x1024)"
    },
    {
        "file": f"{TEST_DIR}/stylegan/extreme_f03.jpg",
        "label": "fake",
        "category": "stylegan3_face",
        "difficulty": "extreme",
        "note": "StyleGAN3 with glasses — harder to detect"
    },

    # Midjourney / SD realistic portraits  
    {
        "file": f"{TEST_DIR}/diffusion/extreme_mj01.jpg",
        "label": "fake",
        "category": "midjourney_portrait",
        "difficulty": "extreme",
        "note": "Midjourney v6 photorealistic face"
    },
    {
        "file": f"{TEST_DIR}/diffusion/extreme_sd01.jpg",
        "label": "fake",
        "category": "stable_diffusion",
        "difficulty": "hard",
        "note": "SD XL realistic portrait"
    },

    # FaceSwap deepfakes
    {
        "file": f"{TEST_DIR}/faceswap/extreme_swap01.jpg",
        "label": "fake",
        "category": "faceswap_image",
        "difficulty": "hard",
        "note": "DeepFaceLab face swap on celebrity"
    },

    # ── CATEGORY 3: Deepfake VIDEOS — the hardest case ───────────────────────
    {
        "file": f"{TEST_DIR}/videos/lipsync_wav2lip_01.mp4",
        "label": "fake",
        "category": "lipsync_wav2lip",
        "difficulty": "extreme",
        "note": "Wav2Lip lip-sync deepfake reel — common Instagram attack"
    },
    {
        "file": f"{TEST_DIR}/videos/lipsync_sadtalker_01.mp4",
        "label": "fake",
        "category": "lipsync_sadtalker",
        "difficulty": "extreme",
        "note": "SadTalker animated portrait reel"
    },
    {
        "file": f"{TEST_DIR}/videos/faceswap_dfl_01.mp4",
        "label": "fake",
        "category": "faceswap_video",
        "difficulty": "hard",
        "note": "DeepFaceLab face-swap reel"
    },
    {
        "file": f"{TEST_DIR}/videos/faceswap_simswap_01.mp4",
        "label": "fake",
        "category": "faceswap_simswap",
        "difficulty": "extreme",
        "note": "SimSwap one-shot face swap (harder to detect)"
    },

    # ── CATEGORY 4: Borderline / Adversarial ─────────────────────────────────
    {
        "file": f"{TEST_DIR}/borderline/filtered_real_01.jpg",
        "label": "real",
        "category": "heavily_filtered_real",
        "difficulty": "hard",
        "note": "Real selfie with heavy Instagram beauty filter applied"
    },
    {
        "file": f"{TEST_DIR}/borderline/compressed_deepfake_01.jpg",
        "label": "fake",
        "category": "compressed_fake",
        "difficulty": "extreme",
        "note": "Deepfake compressed 5x through JPEG/social media pipeline"
    },
]


def run_detector(file_path, timeout=120):
    """Run the detector on a file and return parsed JSON result."""
    if not os.path.exists(file_path):
        return None, "MISSING"

    cmd = [sys.executable, DETECTOR, file_path]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout
        )
        output = result.stdout.strip()
        last_line = output.split("\n")[-1] if output else ""
        try:
            return json.loads(last_line), None
        except json.JSONDecodeError:
            return None, f"JSON_ERROR: {output[:100]}"
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def print_result(entry, res, err, col_w=45):
    fname = entry["file"].split("/")[-1]
    label = entry["label"]
    cat   = entry["category"]
    diff  = entry["difficulty"]

    if err == "MISSING":
        print(f"  {fname:<{col_w}} [{diff}]  {DIM('FILE MISSING — skipped')}")
        return None

    if err:
        print(f"  {fname:<{col_w}} [{diff}]  {RED(f'ERROR: {err}')}")
        return None

    score   = res.get("final_score", 0)
    verdict = res.get("verdict", "ERROR")
    signals = res.get("signals", [])

    # Determine pass/fail
    if label == "real":
        correct = verdict == "APPROVED"
        expected = "APPROVED"
    else:
        correct = verdict == "REJECTED"
        expected = "REJECTED"

    if correct:
        result_str = GREEN("PASS")
        icon = "✓"
    else:
        result_str = RED("FAIL")
        icon = "✗"

    verdict_colored = (
        GREEN(verdict)  if verdict == "APPROVED"   else
        RED(verdict)    if verdict == "REJECTED"   else
        YELLOW(verdict)
    )

    # Key signals (highlight the most important)
    key_sigs = [s for s in signals if s not in (
        "content_type_real_photo", "huggingface_model_used",
        "lightfakedetect_model_used", "borderline_needs_review",
        "synthetic_generation_signal", "signal_analysis_fallback"
    )][:3]
    sig_str = ", ".join(key_sigs) if key_sigs else "(no key signals)"

    hf = "HF" if "huggingface_model_used" in signals else \
         "ONNX" if "lightfakedetect_model_used" in signals else "SIG"

    print(
        f"  {icon} {label.upper():<5} {fname:<{col_w}} "
        f"score={score:.3f} [{hf}] {verdict_colored:<14} {result_str}  {DIM(sig_str)}"
    )

    return {
        "file": entry["file"],
        "label": label,
        "category": cat,
        "difficulty": diff,
        "score": score,
        "verdict": verdict,
        "signals": signals,
        "correct": correct,
        "backend": hf,
        "note": entry.get("note", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-only", action="store_true", help="Only test video files")
    parser.add_argument("--image-only", action="store_true", help="Only test image files")
    parser.add_argument("--available-only", action="store_true", help="Skip missing files silently")
    args = parser.parse_args()

    print(BOLD("\n═════════════════════════════════════════════════════════"))
    print(BOLD("  TrueFrame EXTREME DEEPFAKE TEST — Hard Cases"))
    print(BOLD("═════════════════════════════════════════════════════════\n"))
    print(f"  Detector: {DETECTOR}")
    print(f"  Test dir: {TEST_DIR}")
    print()

    tests = EXTREME_TESTS
    if args.video_only:
        tests = [t for t in tests if t["file"].endswith((".mp4", ".mov", ".avi"))]
    if args.image_only:
        tests = [t for t in tests if not t["file"].endswith((".mp4", ".mov", ".avi"))]

    # Group by category
    categories = {}
    for t in tests:
        cat = t["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(t)

    all_results = []
    total_start = time.time()

    for cat, entries in categories.items():
        print(BOLD(f"\n  ── {cat.replace('_',' ').upper()} ──────────────────────"))
        for entry in entries:
            res, err = run_detector(entry["file"])
            r = print_result(entry, res, err)
            if r is not None:
                all_results.append(r)

    elapsed = time.time() - total_start

    # ── Summary ──────────────────────────────────────
    print(BOLD("\n═════════════════════════════════════════════════════════"))
    print(BOLD("  EXTREME TEST SUMMARY"))
    print(BOLD("═════════════════════════════════════════════════════════"))

    tested   = [r for r in all_results]
    correct  = [r for r in tested if r["correct"]]
    wrong    = [r for r in tested if not r["correct"]]

    fakes_tested   = [r for r in tested if r["label"] == "fake"]
    fakes_caught   = [r for r in fakes_tested if r["correct"]]
    reals_tested   = [r for r in tested if r["label"] == "real"]
    reals_approved = [r for r in reals_tested if r["correct"]]

    acc = len(correct)/len(tested)*100 if tested else 0
    fnr = (len(fakes_tested)-len(fakes_caught))/len(fakes_tested)*100 if fakes_tested else 0
    fpr = (len(reals_tested)-len(reals_approved))/len(reals_tested)*100 if reals_tested else 0

    print(f"\n  Tested:         {len(tested)} files ({len(fakes_tested)} deepfakes, {len(reals_tested)} real)")
    print(f"  Correct:        {len(correct)}/{len(tested)}")
    print(f"  Accuracy:       {acc:.1f}%")
    if fakes_tested:
        print(f"  Deepfakes caught: {len(fakes_caught)}/{len(fakes_tested)}  FNR={fnr:.1f}%  <- KEY METRIC")
    if reals_tested:
        print(f"  Real approved:    {len(reals_approved)}/{len(reals_tested)}  FPR={fpr:.1f}%")
    print(f"  Elapsed:        {elapsed:.1f}s")

    if wrong:
        print(BOLD("\n  FAILURES:"))
        for r in wrong:
            verdict = r["verdict"]
            print(f"    {RED('✗')} [{r['difficulty']}] {r['category']}: {r['file'].split('/')[-1]}")
            print(f"      Expected={'REJECTED' if r['label']=='fake' else 'APPROVED'}, Got={verdict}, Score={r['score']:.3f}")
            print(f"      Note: {r['note']}")

    # Per-difficulty breakdown
    print(BOLD("\n  PER-DIFFICULTY BREAKDOWN:"))
    for diff in ["medium", "hard", "extreme"]:
        d = [r for r in tested if r["difficulty"] == diff]
        if d:
            dc = [r for r in d if r["correct"]]
            print(f"    {diff.capitalize():<10}: {len(dc)}/{len(d)} correct ({len(dc)/len(d)*100:.0f}%)")

    # Backend breakdown
    print(BOLD("\n  BACKEND USED:"))
    backends = {}
    for r in tested:
        b = r.get("backend", "unknown")
        backends[b] = backends.get(b, 0) + 1
    for b, count in sorted(backends.items()):
        flag = "" if b != "SIG" else RED(" <- WARNING: no ML model!")
        print(f"    {b}: {count} files{flag}")

    print(BOLD("\n═════════════════════════════════════════════════════════\n"))

    # Save results
    out_path = "extreme_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved: {out_path}")
    print()


if __name__ == "__main__":
    main()
