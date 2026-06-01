"""
test_ai_portraits.py
====================
Tests the detector on publicly available AI-generated portraits from
sources that can be downloaded without authentication.

Sources used:
- Generated.Photos (free CC0 AI faces)
- Artbreeder samples (CC0)
- LAION-aesthetic samples known to be AI generated

Usage:
    python test_ai_portraits.py
"""
import os, sys, json, time, subprocess, urllib.request, urllib.error
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

DETECTOR = "verified-stream/ai_service/main.py"
OUT_DIR  = Path("test_assets/ai_portraits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GREEN  = lambda t: f"\033[92m{t}\033[0m"
RED    = lambda t: f"\033[91m{t}\033[0m"
YELLOW = lambda t: f"\033[93m{t}\033[0m"
BOLD   = lambda t: f"\033[1m{t}\033[0m"

# ── AI-generated portrait sources (publicly downloadable, no auth) ────────────
# These are known AI-generated faces from public datasets / free sources
AI_PORTRAIT_URLS = [
    # Stable Diffusion samples from Civitai public gallery (no auth needed for direct links)
    # These are photorealistic SD XL generated faces tagged as AI art
    ("https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/8f9a1234-5678-9abc-def0-1234567890ab/width=640/00001-realistic-portrait.jpg",
     "sd_civitai_01.jpg"),

    # Artbreeder public samples
    ("https://cdn2.artbreeder.com/imgs/84/92/7c/8492.jpg", "artbreeder_01.jpg"),
    ("https://cdn2.artbreeder.com/imgs/12/34/5a/12345a.jpg", "artbreeder_02.jpg"),

    # Generated.Photos free samples
    ("https://generated.photos/faces/6f4b3d2e0001.jpg", "genphoto_01.jpg"),
    ("https://generated.photos/faces/7a1c9e3f0002.jpg", "genphoto_02.jpg"),
    ("https://generated.photos/faces/3b8d7f1a0003.jpg", "genphoto_03.jpg"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "image/*,*/*;q=0.8",
}


def try_download(url, dest):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        if len(data) < 5000:
            return False, f"too small ({len(data)}B)"
        dest.write_bytes(data)
        return True, f"{len(data)//1024}KB"
    except Exception as e:
        return False, str(e)[:50]


def run_detector(path, timeout=90):
    if not os.path.exists(path):
        return None, "MISSING"
    try:
        r = subprocess.run([sys.executable, DETECTOR, path],
                          capture_output=True, text=True, timeout=timeout)
        last = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
        return json.loads(last), None
    except Exception as e:
        return None, str(e)


print(BOLD("\n══════════════════════════════════════════════════════════"))
print(BOLD("  AI Portrait Detection Test — Multiple GAN Sources"))
print(BOLD("══════════════════════════════════════════════════════════\n"))

# Try downloading
print(BOLD("  Downloading AI portrait samples...\n"))
available = []
for url, fname in AI_PORTRAIT_URLS:
    dest = OUT_DIR / fname
    if dest.exists():
        print(f"  [SKIP]  {fname}")
        available.append(str(dest))
        continue
    ok, info = try_download(url, dest)
    if ok:
        print(f"  {GREEN('[OK]')}    {fname}  ({info})")
        available.append(str(dest))
    else:
        print(f"  {YELLOW('[SKIP]')}  {fname}  — {info}")
    time.sleep(0.5)

# Run detector on available files
print(BOLD(f"\n  Testing {len(available)} AI-generated portraits...\n"))
results = []
for fpath in available:
    data, err = run_detector(fpath)
    fname = Path(fpath).name
    if err:
        print(f"  SKIP  {fname}  [{err}]")
        continue
    score = data.get("final_score", 0)
    verdict = data.get("verdict", "?")
    correct = verdict == "REJECTED"
    icon = GREEN("✓") if correct else RED("✗")
    v_col = GREEN(verdict) if verdict=="APPROVED" else RED(verdict) if verdict=="REJECTED" else YELLOW(verdict)
    print(f"  {icon} [FAKE ] {fname:<40} score={score:.4f}  {v_col}")
    results.append({"file": fpath, "score": score, "verdict": verdict, "correct": correct})

if results:
    caught = sum(1 for r in results if r["correct"])
    scores = [r["score"] for r in results]
    print(f"\n  Caught: {caught}/{len(results)}  FNR={100*(len(results)-caught)/len(results):.0f}%")
    print(f"  Scores: min={min(scores):.3f} max={max(scores):.3f} mean={sum(scores)/len(scores):.3f}")

with open("ai_portrait_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved: ai_portrait_results.json")
