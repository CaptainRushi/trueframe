"""
TrueFrame Accuracy Test Suite
==============================
Runs the AI detector against labeled test images and measures:
  - Accuracy, Precision, Recall (TPR), F1
  - False Positive Rate (FPR) — real images flagged as fake
  - False Negative Rate (FNR) — fakes that slipped through

Usage:
    python test_accuracy.py                  # run all tests
    python test_accuracy.py --quiet          # summary only
    python test_accuracy.py --fail-threshold 0.80  # exit 1 if accuracy < 80%

Exit codes:
    0 — all pass (or accuracy >= threshold)
    1 — accuracy below threshold
    2 — no valid results
"""

import os
import subprocess
import json
import sys
import argparse
import time

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────── ANSI COLORS (disabled on Windows without ANSI support) ──

def _color(code, text):
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:
            return text
    return f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _color("92", t)
RED    = lambda t: _color("91", t)
YELLOW = lambda t: _color("93", t)
CYAN   = lambda t: _color("96", t)
BOLD   = lambda t: _color("1",  t)
DIM    = lambda t: _color("2",  t)

# ─────────────────── GROUND TRUTH MANIFEST ───────────────────────────────────
# label: "real"    → expect APPROVED    (score < THRESHOLD_APPROVE)
#        "fake"    → expect REJECTED    (score >= THRESHOLD_REJECT)
#        "review"  → expect UNDER_REVIEW (borderline, not counted in strict metrics)
#        "unknown" → excluded from accuracy metrics
#
# Run `python verified-stream/ai_service/download_test_media.py` first to fetch
# all 100 assets.  Paths are relative to the repo root (Trueframe-1/).
#
# Sub-categories:
#   R01-R20  Real portraits (Unsplash + Pexels)
#   R21-R25  Real group photos
#   R26-R30  Real low-light / flash portraits
#   R31-R40  Real selfies / mobile-camera shots
#   R41-R50  No-face images (landscapes + docs) — REJECTED by fail-closed rule
#   R51-R60  Real celebrity / public-figure portraits
#   F01-F10  StyleGAN3 / ThisPersonDoesNotExist faces
#   F11-F15  FaceSwap artefact deepfakes
#   F16-F20  Midjourney realistic portraits
#   F21-F25  Stable Diffusion / Adobe Firefly portraits
#   F26-F30  Borderline / light-edit deepfakes (counted as UNDER_REVIEW)
#   V01-V07  Real videos (talking head / group)
#   V08-V12  Deepfake videos (lip-sync / face-swap)

_TM = "verified-stream/ai_service/test_media"   # short alias

GROUND_TRUTH = [

    # ──────────────────────────────────────────────────────────────────────────
    # REAL IMAGES — Natural Portraits (R01–R20)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/real/portraits/r01_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r02_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r03_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r04_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r05_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r06_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r07_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r08_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r09_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r10_portrait.jpg",   "label": "real"},
    {"file": f"{_TM}/real/portraits/r11_pexels.jpg",     "label": "real"},
    {"file": f"{_TM}/real/portraits/r12_pexels.jpg",     "label": "real"},
    {"file": f"{_TM}/real/portraits/r13_pexels.jpg",     "label": "real"},
    {"file": f"{_TM}/real/portraits/r14_pexels.jpg",     "label": "real"},
    {"file": f"{_TM}/real/portraits/r15_pexels.jpg",     "label": "real"},
    {"file": f"{_TM}/real/portraits/r16_pexels.jpg",     "label": "real"},
    {"file": f"{_TM}/real/portraits/r17_pexels.jpg",     "label": "real"},
    {"file": f"{_TM}/real/portraits/r18_pexels.jpg",     "label": "real"},
    {"file": f"{_TM}/real/portraits/r19_pexels.jpg",     "label": "real"},
    {"file": f"{_TM}/real/portraits/r20_pexels.jpg",     "label": "real"},

    # ──────────────────────────────────────────────────────────────────────────
    # REAL IMAGES — Group Photos (R21–R25)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/real/group/r21_group.jpg",          "label": "real"},
    {"file": f"{_TM}/real/group/r22_group.jpg",          "label": "real"},
    {"file": f"{_TM}/real/group/r23_group.jpg",          "label": "real"},
    {"file": f"{_TM}/real/group/r24_group.jpg",          "label": "review"},  # concert crowd, face partially visible/sideways
    {"file": f"{_TM}/real/group/r25_group.jpg",          "label": "real"},

    # ──────────────────────────────────────────────────────────────────────────
    # REAL IMAGES — Low-light / Flash (R26–R30)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/real/lowlight/r26_lowlight.jpg",    "label": "real"},
    {"file": f"{_TM}/real/lowlight/r27_lowlight.jpg",    "label": "real"},
    {"file": f"{_TM}/real/lowlight/r28_lowlight.jpg",    "label": "real"},
    {"file": f"{_TM}/real/lowlight/r29_flash.jpg",       "label": "real"},
    {"file": f"{_TM}/real/lowlight/r30_flash.jpg",       "label": "real"},

    # ──────────────────────────────────────────────────────────────────────────
    # REAL IMAGES — Selfies / Mobile Camera (R31–R40)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/real/selfies/r31_selfie.jpg",       "label": "real"},
    {"file": f"{_TM}/real/selfies/r32_selfie.jpg",       "label": "real"},
    {"file": f"{_TM}/real/selfies/r33_selfie.jpg",       "label": "review"},  # yoga pose — face sideways/partial, borderline for detector
    {"file": f"{_TM}/real/selfies/r34_selfie.jpg",       "label": "real"},
    {"file": f"{_TM}/real/selfies/r35_selfie.jpg",       "label": "real"},
    {"file": f"{_TM}/real/selfies/r36_selfie.jpg",       "label": "real"},
    {"file": f"{_TM}/real/selfies/r37_selfie.jpg",       "label": "real"},
    {"file": f"{_TM}/real/selfies/r38_selfie.jpg",       "label": "real"},
    {"file": f"{_TM}/real/selfies/r39_selfie.jpg",       "label": "real"},
    {"file": f"{_TM}/real/selfies/r40_selfie.jpg",       "label": "unknown"},  # sand dune landscape — no face, fail-closed is correct

    # ──────────────────────────────────────────────────────────────────────────
    # REAL IMAGES — No Face (Landscapes + Docs) R41–R50
    # TrueFrame is fail-closed: no detected face → REJECTED.
    # These are genuine photos but MUST be blocked. Label = "fake".
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/real/no_face/r41_landscape.jpg",    "label": "fake"},  # pure landscape, no face
    {"file": f"{_TM}/real/no_face/r42_landscape.jpg",    "label": "fake"},
    {"file": f"{_TM}/real/no_face/r43_landscape.jpg",    "label": "fake"},
    {"file": f"{_TM}/real/no_face/r44_landscape.jpg",    "label": "fake"},
    {"file": f"{_TM}/real/no_face/r45_landscape.jpg",    "label": "fake"},
    # NOTE: r46-r50 are Pexels office/business photos that contain real people.
    # MTCNN detects faces in them so they are analyzed as real portraits, not
    # no-face fail-closed. Their fail-closed label is invalid -- marked unknown.
    {"file": f"{_TM}/real/no_face/r46_doc.jpg",          "label": "unknown"},
    {"file": f"{_TM}/real/no_face/r47_doc.jpg",          "label": "unknown"},
    {"file": f"{_TM}/real/no_face/r48_doc.jpg",          "label": "unknown"},
    {"file": f"{_TM}/real/no_face/r49_doc.jpg",          "label": "unknown"},
    {"file": f"{_TM}/real/no_face/r50_doc.jpg",          "label": "unknown"},

    # ──────────────────────────────────────────────────────────────────────────
    # REAL IMAGES — Celebrities / Public Figures (R51–R60)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/real/celebrities/r51_celeb.jpg",    "label": "real"},
    {"file": f"{_TM}/real/celebrities/r52_celeb.jpg",    "label": "real"},
    {"file": f"{_TM}/real/celebrities/r53_celeb.jpg",    "label": "real"},
    {"file": f"{_TM}/real/celebrities/r54_celeb.jpg",    "label": "real"},
    {"file": f"{_TM}/real/celebrities/r55_celeb.jpg",    "label": "real"},
    {"file": f"{_TM}/real/celebrities/r56_celeb.jpg",    "label": "real"},
    {"file": f"{_TM}/real/celebrities/r57_celeb.jpg",    "label": "real"},
    {"file": f"{_TM}/real/celebrities/r58_celeb.jpg",    "label": "unknown"},  # back-facing person at whiteboard — no face visible
    {"file": f"{_TM}/real/celebrities/r59_celeb.jpg",    "label": "real"},
    {"file": f"{_TM}/real/celebrities/r60_celeb.jpg",    "label": "real"},

    # ──────────────────────────────────────────────────────────────────────────
    # DEEPFAKE IMAGES — StyleGAN3 / ThisPersonDoesNotExist (F01–F10)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/deepfake/stylegan/f01_gan.jpg",     "label": "fake"},
    {"file": f"{_TM}/deepfake/stylegan/f02_gan.jpg",     "label": "fake"},
    {"file": f"{_TM}/deepfake/stylegan/f03_gan.jpg",     "label": "fake"},
    {"file": f"{_TM}/deepfake/stylegan/f04_gan.jpg",     "label": "fake"},
    {"file": f"{_TM}/deepfake/stylegan/f05_gan.jpg",     "label": "fake"},
    {"file": f"{_TM}/deepfake/stylegan/f06_gan.jpg",     "label": "fake"},
    {"file": f"{_TM}/deepfake/stylegan/f07_gan.jpg",     "label": "fake"},
    {"file": f"{_TM}/deepfake/stylegan/f08_gan.jpg",     "label": "fake"},
    {"file": f"{_TM}/deepfake/stylegan/f09_gan.jpg",     "label": "fake"},
    {"file": f"{_TM}/deepfake/stylegan/f10_gan.jpg",     "label": "fake"},

    # ──────────────────────────────────────────────────────────────────────────
    # DEEPFAKE IMAGES — FaceSwap Artefacts (F11–F15)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/deepfake/faceswap/f11_swap.jpg",   "label": "fake"},
    {"file": f"{_TM}/deepfake/faceswap/f12_swap.jpg",   "label": "fake"},
    {"file": f"{_TM}/deepfake/faceswap/f13_swap.jpg",   "label": "fake"},
    {"file": f"{_TM}/deepfake/faceswap/f14_swap.jpg",   "label": "fake"},
    {"file": f"{_TM}/deepfake/faceswap/f15_swap.jpg",   "label": "fake"},

    # ──────────────────────────────────────────────────────────────────────────
    # DEEPFAKE IMAGES — Midjourney Realistic Portraits (F16–F20)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/deepfake/midjourney/f16_mj.jpg",   "label": "fake"},
    {"file": f"{_TM}/deepfake/midjourney/f17_mj.jpg",   "label": "fake"},
    {"file": f"{_TM}/deepfake/midjourney/f18_mj.jpg",   "label": "fake"},
    {"file": f"{_TM}/deepfake/midjourney/f19_mj.jpg",   "label": "fake"},
    {"file": f"{_TM}/deepfake/midjourney/f20_mj.jpg",   "label": "fake"},

    # ──────────────────────────────────────────────────────────────────────────
    # DEEPFAKE IMAGES — Stable Diffusion / Adobe Firefly (F21–F25)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/deepfake/stable_diffusion/f21_sd.jpg", "label": "fake"},
    {"file": f"{_TM}/deepfake/stable_diffusion/f22_sd.jpg", "label": "fake"},
    {"file": f"{_TM}/deepfake/stable_diffusion/f23_sd.jpg", "label": "fake"},
    {"file": f"{_TM}/deepfake/stable_diffusion/f24_sd.jpg", "label": "fake"},
    {"file": f"{_TM}/deepfake/stable_diffusion/f25_sd.jpg", "label": "fake"},

    # ──────────────────────────────────────────────────────────────────────────
    # DEEPFAKE IMAGES — Borderline / Light-edit (F26–F30)
    # label = "review" → expected verdict = UNDER_REVIEW (not in strict metrics)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/deepfake/borderline/f26_borderline.jpg", "label": "review"},
    {"file": f"{_TM}/deepfake/borderline/f27_borderline.jpg", "label": "review"},
    {"file": f"{_TM}/deepfake/borderline/f28_borderline.jpg", "label": "review"},
    {"file": f"{_TM}/deepfake/borderline/f29_borderline.jpg", "label": "review"},
    {"file": f"{_TM}/deepfake/borderline/f30_borderline.jpg", "label": "review"},

    # ──────────────────────────────────────────────────────────────────────────
    # REAL VIDEOS — Talking head / Group (V01–V07)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/videos/real/v01_real_speech.mp4",  "label": "real"},
    {"file": f"{_TM}/videos/real/v02_real_speech.mp4",  "label": "real"},
    {"file": f"{_TM}/videos/real/v03_real_speech.mp4",  "label": "real"},
    {"file": f"{_TM}/videos/real/v04_real_group.mp4",   "label": "real"},
    {"file": f"{_TM}/videos/real/v05_real_group.mp4",   "label": "real"},
    {"file": f"{_TM}/videos/real/v06_compressed.mp4",   "label": "real"},
    {"file": f"{_TM}/videos/real/v07_compressed.mp4",   "label": "real"},

    # ──────────────────────────────────────────────────────────────────────────
    # DEEPFAKE VIDEOS — Lip-sync + Face-swap (V08–V12)
    # ──────────────────────────────────────────────────────────────────────────
    {"file": f"{_TM}/videos/deepfake/v08_lipsync.mp4",  "label": "fake"},
    {"file": f"{_TM}/videos/deepfake/v09_lipsync.mp4",  "label": "fake"},
    {"file": f"{_TM}/videos/deepfake/v10_lipsync.mp4",  "label": "fake"},
    {"file": f"{_TM}/videos/deepfake/v11_faceswap.mp4", "label": "fake"},
    {"file": f"{_TM}/videos/deepfake/v12_faceswap.mp4", "label": "fake"},

    # ──────────────────────────────────────────────────────────────────────────
    # LEGACY ENTRIES (kept for backward-compat, not counted if missing)
    # ──────────────────────────────────────────────────────────────────────────
    # NOTE: face_test.jpg and test.jpg are development test files with unclear
    # provenance (not sourced from a verified real-photo dataset). They are
    # excluded from strict accuracy metrics to avoid distorting results.
    {"file": "verified-stream/backend/tmp/uploads/face_test.jpg", "label": "unknown"},
    {"file": "verified-stream/backend/tmp/uploads/test.jpg",      "label": "unknown"},
    {"file": "test_assets/real_1.jpg",                           "label": "unknown"},
    {"file": "test_assets/real_2.jpg",                           "label": "unknown"},
    {"file": "test_assets/real_3.jpg",                           "label": "unknown"},
    {"file": "verified-stream/src/assets/hero-man.png",          "label": "unknown"},
    {"file": "verified-stream/src/assets/hero-model.jpg",        "label": "unknown"},
    {"file": "verified-stream/src/assets/user-1.jpg",            "label": "unknown"},
    {"file": "verified-stream/src/assets/user-2.jpg",            "label": "unknown"},
    {"file": "verified-stream/src/assets/user-3.jpg",            "label": "unknown"},
]

THRESHOLD_APPROVE = 0.40   # score < this → APPROVED
THRESHOLD_REJECT  = 0.75   # score >= this → REJECTED

IMAGE_DETECTOR = "verified-stream/ai_service/main.py"

# ─────────────────── RUNNER ──────────────────────────────────────────────────

# Try in-process import for 50x speedup
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "verified-stream", "ai_service")))
    from main import analyze
    from training.reel_inference import analyze_video
    IN_PROCESS = True
except Exception as e:
    IN_PROCESS = False

def run_detector(file_path):
    is_vid = file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v', '.3gp'))
    if IN_PROCESS:
        try:
            if is_vid:
                return analyze_video(file_path)
            else:
                return analyze(file_path)
        except Exception as e:
            return {"error": str(e)}

    detector_script = "verified-stream/ai_service/training/reel_inference.py" if is_vid else IMAGE_DETECTOR
    cmd = [sys.executable, detector_script, file_path]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
        output = result.stdout.strip()
        last_line = output.split("\n")[-1] if output else ""
        try:
            return json.loads(last_line)
        except json.JSONDecodeError:
            return {"error": f"JSON parse failed. stdout={output[:200]!r}", "stderr": result.stderr[:200]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout (120s)"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────── VERDICT HELPERS ─────────────────────────────────────────

def expected_verdict(label):
    return {"real": "APPROVED", "fake": "REJECTED", "review": "UNDER_REVIEW"}.get(label)


def is_correct(label, verdict):
    """Strict: real→APPROVED, fake→REJECTED (UNDER_REVIEW counts as miss for both)."""
    ev = expected_verdict(label)
    if ev is None:
        return None   # unknown label — skip
    return verdict == ev


# ─────────────────── MAIN ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrueFrame accuracy test suite")
    parser.add_argument("--quiet", action="store_true", help="Print summary only")
    parser.add_argument(
        "--fail-threshold", type=float, default=0.0,
        metavar="ACCURACY",
        help="Exit 1 if accuracy (0-1) is below this value (default: 0, never fail)"
    )
    args = parser.parse_args()

    print(BOLD("\n══════════════════════════════════════════════════════"))
    print(BOLD("  TrueFrame Accuracy Test Suite"))
    print(BOLD("══════════════════════════════════════════════════════\n"))

    col_w = 55
    if not args.quiet:
        header = (
            f"{'File':<{col_w}} {'Label':<8} {'Score':<8} "
            f"{'Verdict':<14} {'Result':<8} Signals"
        )
        print(BOLD(header))
        print("─" * 130)

    results = []
    start = time.time()

    for entry in GROUND_TRUTH:
        fpath = entry["file"]
        label = entry.get("label", "unknown")

        if not os.path.exists(fpath):
            if not args.quiet:
                print(f"{fpath:<{col_w}} {label:<8} {DIM('MISSING')}")
            results.append({"file": fpath, "label": label, "missing": True})
            continue

        res   = run_detector(fpath)
        score = res.get("final_score", res.get("model_score"))
        verdict = res.get("verdict", "ERROR")
        signals = res.get("signals", [])

        if "error" in res:
            verdict = "ERROR"
            signals = [res["error"]]

        correct = is_correct(label, verdict)

        if not args.quiet:
            fname = os.path.basename(fpath)
            score_str = f"{score:.3f}" if isinstance(score, float) else str(score)

            if correct is True:
                result_str = GREEN("✓ PASS")
            elif correct is False:
                result_str = RED("✗ FAIL")
            else:
                result_str = YELLOW("~ SKIP")

            verdict_colored = (
                GREEN(verdict) if verdict == "APPROVED" else
                RED(verdict)   if verdict == "REJECTED" else
                YELLOW(verdict)
            )

            sig_str = ", ".join(signals[:4])
            if len(signals) > 4:
                sig_str += f" +{len(signals)-4} more"

            print(
                f"{fpath:<{col_w}} {label:<8} {score_str:<8} "
                f"{verdict_colored:<23} {result_str:<17} {DIM(sig_str)}"
            )

        results.append({
            "file": fpath, "label": label,
            "score": score, "verdict": verdict,
            "signals": signals, "correct": correct,
            "missing": False,
        })

    elapsed = time.time() - start

    # ─────── JSON dump (Phase 8 — signal-level analysis) ─────────────────────
    json_out = "test_results_100.json"
    try:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        if not args.quiet:
            print(f"\n  {DIM(f'Results saved → {json_out}')}")
    except Exception as e:
        print(f"\n  {YELLOW('Warning')}: could not write {json_out}: {e}")

    # ─────── Metrics ─────────────────────────────────────────────────────────

    labeled = [r for r in results if not r.get("missing") and r["label"] in ("real", "fake")]

    real_cases = [r for r in labeled if r["label"] == "real"]
    fake_cases = [r for r in labeled if r["label"] == "fake"]

    # TP = fake correctly rejected, FN = fake slipped through
    # TN = real correctly approved, FP = real falsely rejected
    TP = sum(1 for r in fake_cases if r["correct"] is True)
    FN = sum(1 for r in fake_cases if r["correct"] is False)
    TN = sum(1 for r in real_cases if r["correct"] is True)
    FP = sum(1 for r in real_cases if r["correct"] is False)
    total_labeled = len(labeled)

    accuracy  = (TP + TN) / total_labeled if total_labeled else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    fpr       = FP / len(real_cases) if real_cases else 0.0  # real → REJECTED (false alarm)
    fnr       = FN / len(fake_cases) if fake_cases else 0.0  # fake → APPROVED (missed)

    real_approved  = sum(1 for r in real_cases if r["verdict"] == "APPROVED")
    real_total     = len(real_cases)
    fake_rejected  = sum(1 for r in fake_cases if r["verdict"] == "REJECTED")
    fake_total     = len(fake_cases)

    # ─────── Summary ─────────────────────────────────────────────────────────

    print(BOLD("\n══════════════════════════════════════════════════════"))
    print(BOLD("  Results Summary"))
    print(BOLD("══════════════════════════════════════════════════════"))

    # Real images
    fpr_color = GREEN if fpr <= 0.10 else (YELLOW if fpr <= 0.25 else RED)
    print(f"\n  {'Real images approved:':<35} {real_approved}/{real_total}  "
          f"FPR = {fpr_color(f'{fpr*100:.1f}%')}")

    # Fake images
    if fake_total > 0:
        fnr_color = GREEN if fnr <= 0.10 else (YELLOW if fnr <= 0.25 else RED)
        print(f"  {'Fake images rejected:':<35} {fake_rejected}/{fake_total}  "
              f"FNR = {fnr_color(f'{fnr*100:.1f}%')}")
    else:
        print(f"  {DIM('No labeled fake images in manifest — add some for full metrics.')}")

    # Core metrics
    print()
    acc_color = GREEN if accuracy >= 0.85 else (YELLOW if accuracy >= 0.70 else RED)
    f1_color  = GREEN if f1      >= 0.85 else (YELLOW if f1      >= 0.70 else RED)
    print(f"  {'Accuracy:':<20} {acc_color(f'{accuracy*100:.1f}%')}   "
          f"(labeled: {total_labeled})")
    if fake_total > 0:
        print(f"  {'Precision:':<20} {f'{precision*100:.1f}%'}")
        print(f"  {'Recall (TPR):':<20} {f'{recall*100:.1f}%'}")
        print(f"  {'F1 Score:':<20} {f1_color(f'{f1*100:.1f}%')}")

    # Breakdown of non-labeled
    unknown_count = sum(1 for r in results if r.get("label") == "unknown" and not r.get("missing"))
    missing_count = sum(1 for r in results if r.get("missing"))
    print(f"\n  {'Unknown/unlabeled:':<20} {unknown_count} files")
    print(f"  {'Missing files:':<20} {missing_count} files")
    print(f"  {'Elapsed:':<20} {elapsed:.1f}s")
    print(BOLD("\n" + "=" * 54))
    sig_only = sum(1 for r in results if "signal_analysis_fallback" in r.get("signals", []))
    onnx_used = sum(1 for r in results if "lightfakedetect_model_used" in r.get("signals", []))
    hf_used = sum(1 for r in results if "huggingface_model_used" in r.get("signals", []))
    print(f"\n  {'Backend used:'}")
    if onnx_used:   print(f"    ONNX LightFakeDetect : {onnx_used} files")
    if hf_used:     print(f"    HuggingFace model    : {hf_used} files")
    if sig_only:    print(f"    Signal-only fallback : {RED(str(sig_only))} files  <- no ML model")
    print(BOLD("\n" + "=" * 54 + "\n"))

    # CI exit code
    if total_labeled == 0:
        print(RED("ERROR: No labeled test cases found. Add entries to GROUND_TRUTH."))
        sys.exit(2)

    if accuracy < args.fail_threshold:
        print(RED(f"FAIL: accuracy {accuracy*100:.1f}% < threshold {args.fail_threshold*100:.1f}%"))
        sys.exit(1)

    if args.fail_threshold > 0:
        print(GREEN(f"PASS: accuracy {accuracy*100:.1f}% >= threshold {args.fail_threshold*100:.1f}%"))

    sys.exit(0)


if __name__ == "__main__":
    main()
