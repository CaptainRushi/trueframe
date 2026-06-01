"""
generate_and_test_deepfakes.py
==============================
Downloads real StyleGAN faces from thispersondoesnotexist.com and
tests the detector on them. Also tests the detector on any deepfake
images already in the test_media directory.

These are REAL deepfakes:
- thispersondoesnotexist.com = StyleGAN3 (no real person exists)
- generated_ai_faces/ = locally generated AI portraits

Usage:
    python generate_and_test_deepfakes.py              # download + test 10 TPDNE faces
    python generate_and_test_deepfakes.py --count 20   # test 20 faces
    python generate_and_test_deepfakes.py --test-only  # only test already-downloaded faces
"""

import os, sys, json, time, subprocess, argparse, random
import urllib.request, urllib.error
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

DETECTOR  = "verified-stream/ai_service/main.py"
OUT_DIR   = Path("test_assets/stylegan_tpdne")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GREEN  = lambda t: f"\033[92m{t}\033[0m"
RED    = lambda t: f"\033[91m{t}\033[0m"
YELLOW = lambda t: f"\033[93m{t}\033[0m"
BOLD   = lambda t: f"\033[1m{t}\033[0m"
DIM    = lambda t: f"\033[2m{t}\033[0m"

TPDNE_URLS = [
    "https://thispersondoesnotexist.com",
    "https://thispersondoesnotexist.com/image",
]

ALT_GAN_SOURCES = [
    # Generated.Photos — free CC0 AI faces (different GAN)
    ("https://generated.photos/faces/6f4b3d2e-face-00001.jpg", "genphoto_01.jpg"),
    ("https://generated.photos/faces/7a1c9e3f-face-00002.jpg", "genphoto_02.jpg"),
    # This X Does Not Exist alternatives
    ("https://www.this-person-does-not-exist.com/img/avatar-gen0.jpg", "tpdne_alt_01.jpg"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/avif,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://thispersondoesnotexist.com/",
    "Cache-Control": "no-cache",
}


def download_tpdne(dest: Path, attempt=0) -> tuple[bool, str]:
    """Download one TPDNE face. TPDNE returns a new face on each request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = TPDNE_URLS[attempt % len(TPDNE_URLS)]
    try:
        # Add random cache-buster
        req = urllib.request.Request(
            f"{url}?r={random.randint(100000, 999999)}",
            headers={**HEADERS, "Cache-Control": f"no-cache, max-age={random.randint(0,99)}"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        if len(data) < 10000:
            return False, f"too small ({len(data)} bytes)"
        dest.write_bytes(data)
        return True, f"{len(data)//1024}KB"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)[:60]


def run_detector(path: str, timeout=90) -> tuple[dict | None, str | None]:
    if not os.path.exists(path):
        return None, "MISSING"
    try:
        r = subprocess.run(
            [sys.executable, DETECTOR, path],
            capture_output=True, text=True, check=False, timeout=timeout
        )
        last = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
        return json.loads(last), None
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def test_file(path: str, label: str):
    data, err = run_detector(path)
    fname = Path(path).name

    if err:
        print(f"  {'SKIP':<6} {fname:<45} [{err}]")
        return None

    score   = data.get("final_score", 0)
    verdict = data.get("verdict", "?")
    is_video = data.get("video_threshold_reject") is not None
    thr = data.get("video_threshold_reject", 0.75)

    correct = (verdict == "REJECTED") if label == "fake" else (verdict == "APPROVED")
    icon = GREEN("✓") if correct else RED("✗")
    v_str = GREEN(verdict) if verdict == "APPROVED" else \
            RED(verdict)   if verdict == "REJECTED" else \
            YELLOW(verdict)
    thr_str = DIM(f"[vid thr={thr:.2f}]") if is_video else ""

    print(f"  {icon} [{label.upper():<5}] {fname:<45} score={score:.4f}  {v_str} {thr_str}")
    return {"file": path, "label": label, "score": score, "verdict": verdict, "correct": correct}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",     type=int, default=10, help="Number of TPDNE faces to fetch")
    parser.add_argument("--test-only", action="store_true",  help="Skip download, test existing files")
    parser.add_argument("--no-real",   action="store_true",  help="Skip testing real images")
    args = parser.parse_args()

    print(BOLD("\n══════════════════════════════════════════════════════════"))
    print(BOLD("  TrueFrame — StyleGAN Deepfake Detection Test"))
    print(BOLD("══════════════════════════════════════════════════════════\n"))

    fake_files = []

    # ── Step 1: Download TPDNE faces ────────────────────────────────────────
    if not args.test_only:
        print(BOLD(f"  Downloading {args.count} StyleGAN faces from thispersondoesnotexist.com...\n"))
        downloaded = 0
        attempt = 0
        while downloaded < args.count and attempt < args.count * 3:
            fname = OUT_DIR / f"tpdne_{downloaded+1:02d}.jpg"
            if fname.exists():
                print(f"  {DIM('[SKIP]')}  {fname.name}")
                fake_files.append(str(fname))
                downloaded += 1
                attempt += 1
                continue

            ok, info = download_tpdne(fname, attempt)
            if ok:
                print(f"  {GREEN('[OK]  ')}  {fname.name}  ({info})")
                fake_files.append(str(fname))
                downloaded += 1
            else:
                print(f"  {YELLOW('[RETRY]')}  Attempt {attempt+1}: {info}")
            attempt += 1
            time.sleep(1.5)  # polite crawl delay

        print(f"\n  Downloaded: {downloaded}/{args.count} TPDNE faces\n")
    else:
        fake_files = sorted(str(p) for p in OUT_DIR.glob("*.jpg"))
        print(f"  Found {len(fake_files)} existing TPDNE files\n")

    # Also include any deepfakes already in test_media
    for p in Path("verified-stream/ai_service/test_media/deepfake").rglob("*.jpg"):
        fake_files.append(str(p))

    # ── Step 2: Test all fake files ─────────────────────────────────────────
    print(BOLD("  ── DEEPFAKE DETECTION TEST ─────────────────────────────"))
    all_results = []

    for fpath in fake_files:
        r = test_file(fpath, "fake")
        if r:
            all_results.append(r)

    # ── Step 3: Test real images as control ─────────────────────────────────
    if not args.no_real:
        real_samples = [
            ("verified-stream/ai_service/test_media/real/portraits/r01_portrait.jpg", "real"),
            ("verified-stream/ai_service/test_media/real/portraits/r05_portrait.jpg", "real"),
            ("verified-stream/ai_service/test_media/real/portraits/r10_portrait.jpg", "real"),
            ("verified-stream/ai_service/test_media/real/selfies/r31_selfie.jpg",     "real"),
            ("verified-stream/ai_service/test_media/real/selfies/r35_selfie.jpg",     "real"),
            ("176527-855920754_medium.mp4",                                           "real"),
        ]
        print(BOLD("\n  ── REAL IMAGE CONTROL (must be APPROVED) ──────────────"))
        for fpath, label in real_samples:
            r = test_file(fpath, label)
            if r:
                all_results.append(r)

    # ── Summary ─────────────────────────────────────────────────────────────
    fakes   = [r for r in all_results if r["label"] == "fake"]
    reals   = [r for r in all_results if r["label"] == "real"]
    tp      = sum(1 for r in fakes if r["correct"])
    fn      = len(fakes) - tp
    tn      = sum(1 for r in reals if r["correct"])
    fp      = len(reals) - tn
    total   = len(all_results)
    acc     = (tp + tn) / total * 100 if total else 0
    fnr     = fn / len(fakes) * 100 if fakes else 0
    fpr     = fp / len(reals) * 100 if reals else 0

    print(BOLD("\n══════════════════════════════════════════════════════════"))
    print(BOLD("  DEEPFAKE DETECTION RESULTS"))
    print(BOLD("══════════════════════════════════════════════════════════"))
    print(f"\n  StyleGAN faces tested : {len(fakes)}")
    print(f"  Real images (control) : {len(reals)}")
    print(f"  Total                 : {total}")
    print()
    if fakes:
        scores = [r["score"] for r in fakes]
        verdicts = {v: sum(1 for r in fakes if r["verdict"]==v) for v in ["REJECTED","UNDER_REVIEW","APPROVED"]}
        print(f"  Deepfake scores  — min={min(scores):.3f}  max={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}")
        print(f"  REJECTED         : {verdicts.get('REJECTED',0)}/{len(fakes)}  ← caught")
        print(f"  UNDER_REVIEW     : {verdicts.get('UNDER_REVIEW',0)}/{len(fakes)}  ← needs human review")
        print(f"  APPROVED         : {verdicts.get('APPROVED',0)}/{len(fakes)}  ← MISSED (false negative)")
        print(f"  FNR              : {fnr:.1f}%  (deepfakes that slipped through)")
    if reals:
        r_scores = [r["score"] for r in reals]
        print(f"\n  Real image FPR   : {fpr:.1f}%  ({fp}/{len(reals)} wrongly blocked)")
        print(f"  Real scores      — min={min(r_scores):.3f}  max={max(r_scores):.3f}  mean={sum(r_scores)/len(r_scores):.3f}")
    print(f"\n  Overall accuracy : {acc:.1f}%")

    # Save
    out = {"summary": {"total": total, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                        "accuracy": round(acc,1), "fnr": round(fnr,1), "fpr": round(fpr,1)},
           "results": all_results}
    with open("deepfake_test_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved: deepfake_test_results.json")
    print(BOLD("\n══════════════════════════════════════════════════════════\n"))


if __name__ == "__main__":
    main()
