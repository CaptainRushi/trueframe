"""
TrueFrame Platform Testing Suite
=================================
Executes all 10 test cases from the platform_testing_plan.md
and generates a detailed pass/fail report with remediation hints.

Run from the verified-stream/ directory:
    python ai_service/run_platform_tests.py
"""

import subprocess
import json
import sys
import os
import time
import io

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

# -- colour helpers (works on modern Windows terminals) ----------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _c(color, text):
    return f"{color}{text}{RESET}"


# -- locate repo root --------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent           # .../ai_service/
REPO_ROOT    = SCRIPT_DIR.parent                          # .../verified-stream/
AI_DIR       = SCRIPT_DIR
REAL_DIR     = AI_DIR / "test_media" / "real"
FAKE_DIR     = AI_DIR / "test_media" / "deepfake"

# -- test matrix ------------------------------------------------------------
# (id, label, asset_path, expected_verdict, description)
TESTS = [
    (
        "01", "Real Image  -  Natural Portrait",
        REAL_DIR / "natural_face.png",
        "APPROVED",
        "Standard portrait of a real person; should score low on all signals.",
    ),
    (
        "02", "Real Image  -  Bright/Overexposed Portrait",
        REAL_DIR / "bright_face.png",
        "APPROVED",
        "Portrait under strong overexposed lighting.",
    ),
    (
        "03", "Real Image  -  Dark/Low-light Portrait",
        REAL_DIR / "dark_portrait.png",
        "APPROVED",
        "Portrait under low-light environment; uses brightened fallback path.",
    ),
    (
        "04", "Real Image  -  Landscape (No Face)",
        REAL_DIR / "landscape.png",
        "REJECTED",
        "Scenery image without any human face; must be rejected (fail-closed).",
    ),
    (
        "05", "Deepfake  -  Blending Seam Face",
        FAKE_DIR / "seam_face.png",
        "REJECTED",
        "Face with artificial blending borders (seam artefact).",
    ),
    (
        "06", "Deepfake  -  Channel-Decoupled Face",
        FAKE_DIR / "channel_decoupled_face.png",
        "REJECTED",
        "Face showing decoupled R/G/B colour channels.",
    ),
    (
        "07", "Deepfake  -  Oversmoothed Skin Face",
        FAKE_DIR / "oversmoothed_face.png",
        "REJECTED",
        "GAN-generated face with unnaturally smooth skin texture.",
    ),
    (
        "08", "Deepfake  -  Heavy JPEG Artefacts",
        FAKE_DIR / "heavy_jpeg_face.png",
        "REJECTED",
        "Deepfake with DCT grid re-encoding block artefacts.",
    ),
    (
        "09", "Real Video  -  Person Speaking",
        REAL_DIR / "real_frame_0.png",   # use real frame as proxy if no mp4
        "APPROVED",
        "Clean still-frame proxy for a real video recording.",
    ),
    (
        "10", "Fail-Closed  -  Non-Existent File",
        Path("non_existent_file_xyzabc.mp4"),
        "REJECTED",
        "Missing / invalid file; must be rejected with fail_closed signal.",
    ),
]


# -- runner ------------------------------------------------------------------

def run_test(test_id, label, asset_path, expected, description):
    """Run main.py on one asset, return (passed, result_dict, elapsed_s)."""
    abs_path = str(asset_path.resolve()) if asset_path.is_absolute() else str(asset_path)

    cmd = [sys.executable, str(AI_DIR / "main.py"), abs_path]
    t0  = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        elapsed = round(time.perf_counter() - t0, 2)
        stdout  = proc.stdout.strip()
        stderr  = proc.stderr.strip()

        # parse JSON output
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            result = {
                "verdict":     "PARSE_ERROR",
                "final_score": -1,
                "signals":     [f"bad_json: {stdout[:200]}"],
                "error":       stderr[:400] if stderr else "",
            }
        result["_elapsed"] = elapsed
        result["_stderr"]  = stderr[-600:] if stderr else ""

        actual  = result.get("verdict", "PARSE_ERROR")
        passed  = (actual == expected)
        return passed, result

    except subprocess.TimeoutExpired:
        elapsed = round(time.perf_counter() - t0, 2)
        return False, {
            "verdict":     "TIMEOUT",
            "final_score": -1,
            "signals":     ["process_timeout"],
            "_elapsed":    elapsed,
            "_stderr":     "",
        }
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 2)
        return False, {
            "verdict":     "EXCEPTION",
            "final_score": -1,
            "signals":     [str(exc)],
            "_elapsed":    elapsed,
            "_stderr":     "",
        }


def print_result(test_id, label, expected, passed, result):
    actual  = result.get("verdict", "?")
    score   = result.get("final_score", -1)
    signals = result.get("signals", [])
    elapsed = result.get("_elapsed", "?")
    onnx    = result.get("onnx_model_score")

    status_str = _c(GREEN, "PASS OK") if passed else _c(RED, "FAIL XX")
    verdict_col = GREEN if actual == "APPROVED" else (RED if actual == "REJECTED" else YELLOW)

    print(f"\n{'='*68}")
    print(f"  Case {test_id}: {_c(BOLD, label)}")
    print(f"  Status  : {status_str}  ({elapsed}s)")
    print(f"  Expected: {expected}   Got: {_c(verdict_col, actual)}")
    print(f"  Score   : {score:.4f}" if isinstance(score, float) else f"  Score   : {score}")
    if onnx is not None:
        print(f"  ONNX    : {onnx:.4f}")
    if signals:
        print(f"  Signals : {', '.join(signals[:5])}")
    if not passed:
        stderr = result.get("_stderr", "")
        if stderr:
            print(f"  {_c(YELLOW, 'stderr')}  : {stderr[:300]}")


def main():
    print(_c(BOLD + CYAN, "\n==================================================================="))
    print(_c(BOLD + CYAN, "   TrueFrame Platform Testing Suite -- 10 Media Asset Verification  "))
    print(_c(BOLD + CYAN, "==================================================================="))
    print(f"  Repo root : {REPO_ROOT}")
    print(f"  Python    : {sys.executable}")
    print()

    passed_ids = []
    failed_ids = []
    results    = []

    for (tid, label, asset, expected, desc) in TESTS:
        print(f"  Running Case {tid}: {label} ...", end=" ", flush=True)
        ok, res = run_test(tid, label, asset, expected, desc)
        print(_c(GREEN, "done") if ok else _c(RED, "FAILED"))
        results.append((tid, label, asset, expected, ok, res))

    # -- detailed results ----------------------------------------------------
    print()
    print(_c(BOLD, "--- DETAILED RESULTS -----------------------------------------------------------"))
    for (tid, label, asset, expected, ok, res) in results:
        print_result(tid, label, expected, ok, res)
        if ok:
            passed_ids.append(tid)
        else:
            failed_ids.append(tid)

    # -- summary -------------------------------------------------------------
    total  = len(TESTS)
    passed = len(passed_ids)
    failed = len(failed_ids)

    print(f"\n{'='*68}")
    print(_c(BOLD, f"  SUMMARY: {passed}/{total} tests passed"))
    if passed_ids:
        print(_c(GREEN,  f"  PASSED : {', '.join(passed_ids)}"))
    if failed_ids:
        print(_c(RED,    f"  FAILED : {', '.join(failed_ids)}"))

    # -- remediation hints for failures --------------------------------------
    if failed_ids:
        print()
        print(_c(BOLD + YELLOW, "  REMEDIATION HINTS:"))
        for (tid, label, asset, expected, ok, res) in results:
            if ok:
                continue
            actual  = res.get("verdict", "?")
            signals = res.get("signals", [])
            score   = res.get("final_score", -1)

            if actual == "APPROVED" and expected == "REJECTED":
                print(f"\n  Case {tid} ({label})  -  False Negative (deepfake slipped through):")
                print(f"    -> Score {score:.4f} is below THRESHOLD_REJECT.")
                print(f"    -> Signals detected: {signals}")
                print(f"    -> Action: Raise signal boost coefficients in main.py for these signals.")
                print(f"              Check WEIGHT_ARTIFACT / WEIGHT_COMPRESSION / WEIGHT_MODEL.")

            elif actual == "REJECTED" and expected == "APPROVED":
                print(f"\n  Case {tid} ({label})  -  False Positive (real media blocked):")
                print(f"    -> Score {score:.4f} triggered reject threshold.")
                print(f"    -> Signals: {signals}")
                print(f"    -> Action: Tune sensitivity of triggered signals in _signal_* helpers.")
                print(f"              Or lower the weight of noisy signal paths.")

            elif actual in ("PARSE_ERROR", "TIMEOUT", "EXCEPTION"):
                print(f"\n  Case {tid} ({label})  -  Engine Error:")
                print(f"    -> {res.get('signals', [])}")
                print(f"    -> stderr: {res.get('_stderr', '')[:300]}")
                print(f"    -> Action: Check Python dependencies (pip install -r requirements.txt).")

    print(f"\n{'='*68}\n")

    # exit code reflects overall pass/fail for CI integration
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
