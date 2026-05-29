"""
download_test_media.py
======================
Downloads ~80 auto-fetchable test media assets from free/CC sources.
The remaining ~20 (StyleGAN, Midjourney, SD, deepfake videos) must be
downloaded manually -- this script prints exact instructions for those.

Usage (from repo root):
    python verified-stream/ai_service/download_test_media.py
    python verified-stream/ai_service/download_test_media.py --dry-run
    python verified-stream/ai_service/download_test_media.py --verify-only

Requirements: No external packages -- uses stdlib urllib only.
"""

import sys
import os

# Force UTF-8 on Windows to avoid cp1252 errors with box-drawing chars
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import time
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # .../ai_service/
BASE       = SCRIPT_DIR / "test_media"                # .../ai_service/test_media/

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def c(color, text):
    return f"{color}{text}{RESET}"


# =============================================================================
# DOWNLOAD MANIFEST
# Each entry: {"url": ..., "dest": Path(...), "label": ..., "manual": True/False}
# label: "real" | "fake" | "review"
# =============================================================================

DOWNLOADS = []

def add(url, dest_parts, label, manual=False):
    dest = BASE.joinpath(*dest_parts)
    DOWNLOADS.append({"url": url, "dest": dest, "label": label, "manual": manual})


# ── REAL IMAGES: Natural Portraits R01–R10 (Unsplash CDN) ────────────────────
UNSPLASH = [
    ("1500648767791-00dcc994a43e", "r01_portrait.jpg"),
    ("1506794778202-cad84cf45f1d", "r02_portrait.jpg"),
    ("1507003211169-0a1dd7228f2d", "r03_portrait.jpg"),
    ("1494790108377-be9c29b29330", "r04_portrait.jpg"),
    ("1531746020798-e6953c6e8e04", "r05_portrait.jpg"),
    ("1499946576-45153-153030-random", "r06_portrait.jpg"),  # will be overridden below
    ("1508214751196-bcfd4ca60f91", "r07_portrait.jpg"),
    ("1544725176-7c40e5a71c5e",   "r08_portrait.jpg"),
    ("1534528741775-53994a69daeb", "r09_portrait.jpg"),  # alt: working Unsplash ID
    ("1590086782957-93c06ef21604", "r10_portrait.jpg"),
]
for uid, fname in UNSPLASH:
    add(
        f"https://images.unsplash.com/photo-{uid}?w=640&q=85&fm=jpg&fit=crop&crop=face",
        ["real", "portraits", fname], "real"
    )

# Override r06 with a reliable Pexels fallback (Unsplash blocked this slot)
add(
    "https://images.pexels.com/photos/415829/pexels-photo-415829.jpeg?auto=compress&cs=tinysrgb&w=640",
    ["real", "portraits", "r06_portrait.jpg"], "real"
)

# ── REAL IMAGES: Natural Portraits R11–R20 (Pexels) ──────────────────────────
PEXELS_PORTRAITS = [
    (774909,  "r11_pexels.jpg"),
    (1239291, "r12_pexels.jpg"),
    (1181686, "r13_pexels.jpg"),
    (1484794, "r14_pexels.jpg"),
    (3785079, "r15_pexels.jpg"),
    (1040881, "r16_pexels.jpg"),
    (2379004, "r17_pexels.jpg"),
    (1043474, "r18_pexels.jpg"),
    (3764119, "r19_pexels.jpg"),
    (1036623, "r20_pexels.jpg"),
]
for pid, fname in PEXELS_PORTRAITS:
    add(
        f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=640",
        ["real", "portraits", fname], "real"
    )

# ── REAL IMAGES: Group Photos R21–R25 (Pexels) ───────────────────────────────
PEXELS_GROUPS = [
    (1267696, "r21_group.jpg"),
    (3184299, "r22_group.jpg"),
    (3184418, "r23_group.jpg"),
    (2833037, "r24_group.jpg"),
    (1181533, "r25_group.jpg"),
]
for pid, fname in PEXELS_GROUPS:
    add(
        f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=640",
        ["real", "group", fname], "real"
    )

# ── REAL IMAGES: Low-light / Flash R26–R30 (Pexels) ─────────────────────────
PEXELS_LOWLIGHT = [
    (1820559, "r26_lowlight.jpg"),
    (2269872, "r27_lowlight.jpg"),
    (1906802, "r28_lowlight.jpg"),
    (1382731, "r29_flash.jpg"),
    (1197132, "r30_flash.jpg"),
]
for pid, fname in PEXELS_LOWLIGHT:
    add(
        f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=640",
        ["real", "lowlight", fname], "real"
    )

# ── REAL IMAGES: Selfies R31–R40 (Pexels) ────────────────────────────────────
PEXELS_SELFIES = [
    (3586798, "r31_selfie.jpg"),
    (4307869, "r32_selfie.jpg"),
    (3823039, "r33_selfie.jpg"),
    (2080938, "r34_selfie.jpg"),
    (1212984, "r35_selfie.jpg"),
    (1382734, "r36_selfie.jpg"),
    (3771839, "r37_selfie.jpg"),
    (2726111, "r38_selfie.jpg"),
    (2955305, "r39_selfie.jpg"),
    (3307618, "r40_selfie.jpg"),
]
for pid, fname in PEXELS_SELFIES:
    add(
        f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=640",
        ["real", "selfies", fname], "real"
    )

# ── REAL IMAGES: Landscapes (No Face) R41–R45 ────────────────────────────────
# label is "fake" because fail-closed: no face → REJECTED
PEXELS_LANDSCAPES = [
    (414612,  "r41_landscape.jpg"),
    (1366919, "r42_landscape.jpg"),
    (624015,  "r43_landscape.jpg"),
    (210186,  "r44_landscape.jpg"),
    (1624600, "r45_landscape.jpg"),
]
for pid, fname in PEXELS_LANDSCAPES:
    add(
        f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=640",
        ["real", "no_face", fname], "fake"   # expected verdict: REJECTED (fail-closed)
    )

# ── REAL IMAGES: Documents / Abstract (No Face) R46–R50 ─────────────────────
PEXELS_DOCS = [
    (95916,  "r46_doc.jpg"),
    (590016, "r47_doc.jpg"),
    (669619, "r48_doc.jpg"),
    (261763, "r49_doc.jpg"),
    (357514, "r50_doc.jpg"),  # alt Pexels doc photo
]
for pid, fname in PEXELS_DOCS:
    add(
        f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=640",
        ["real", "no_face", fname], "fake"   # expected verdict: REJECTED (fail-closed)
    )

# ── REAL IMAGES: Celebrities / Public Figures R51–R60 (Pexels) ───────────────
PEXELS_CELEBS = [
    (1468379, "r51_celeb.jpg"),
    (1102341, "r52_celeb.jpg"),
    (2182970, "r53_celeb.jpg"),
    (2065195, "r54_celeb.jpg"),
    (3785077, "r55_celeb.jpg"),
    (1587009, "r56_celeb.jpg"),
    (3757942, "r57_celeb.jpg"),
    (1181345, "r58_celeb.jpg"),
    (1516680, "r59_celeb.jpg"),
    (2406949, "r60_celeb.jpg"),
]
for pid, fname in PEXELS_CELEBS:
    add(
        f"https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=640",
        ["real", "celebrities", fname], "real"
    )

# ── DEEPFAKE IMAGES: StyleGAN / TPDNE F01–F10 ────────────────────────────────
# thispersondoesnotexist.com refreshes on each reload — must save manually.
for i in range(1, 11):
    add(
        "https://thispersondoesnotexist.com",
        ["deepfake", "stylegan", f"f{i:02d}_gan.jpg"], "fake",
        manual=True
    )

# ── DEEPFAKE IMAGES: FaceSwap F11–F15 ────────────────────────────────────────
# FaceForensics++ requires academic registration. Manual download.
for i in range(11, 16):
    add(
        "https://github.com/ondyari/FaceForensics (academic access required)",
        ["deepfake", "faceswap", f"f{i:02d}_swap.jpg"], "fake",
        manual=True
    )

# ── DEEPFAKE IMAGES: Midjourney F16–F20 ──────────────────────────────────────
# Midjourney showcase — must be saved manually from https://www.midjourney.com/showcase
for i in range(16, 21):
    add(
        "https://www.midjourney.com/showcase (save realistic portrait images)",
        ["deepfake", "midjourney", f"f{i:02d}_mj.jpg"], "fake",
        manual=True
    )

# ── DEEPFAKE IMAGES: Stable Diffusion / Adobe Firefly F21–F25 ────────────────
# Generate from Adobe Firefly or SD WebUI — save manually.
for i in range(21, 26):
    add(
        "https://firefly.adobe.com (generate a photorealistic portrait)",
        ["deepfake", "stable_diffusion", f"f{i:02d}_sd.jpg"], "fake",
        manual=True
    )

# ── DEEPFAKE IMAGES: Borderline / Light-edit F26–F30 ─────────────────────────
# DeepFaceLab samples or lightly edited real photos — manual.
for i in range(26, 31):
    add(
        "https://github.com/iperov/DeepFaceLab (use output samples)",
        ["deepfake", "borderline", f"f{i:02d}_borderline.jpg"], "review",
        manual=True
    )

# ── REAL VIDEOS V01–V07 ───────────────────────────────────────────────────────
# Pexels videos require browser download — mark manual.
PEXELS_VIDEOS_REAL = [
    (3195394, "v01_real_speech.mp4"),
    (3195399, "v02_real_speech.mp4"),
    (3195396, "v03_real_speech.mp4"),
    (3209828, "v04_real_group.mp4"),
    (3209832, "v05_real_group.mp4"),
    (2795750, "v06_compressed.mp4"),
    (3194625, "v07_compressed.mp4"),
]
for vid, fname in PEXELS_VIDEOS_REAL:
    add(
        f"https://www.pexels.com/video/{vid}/ (download HD MP4)",
        ["videos", "real", fname], "real",
        manual=True
    )

# ── DEEPFAKE VIDEOS V08–V12 ───────────────────────────────────────────────────
DEEPFAKE_VIDEOS = [
    ("FaceForensics++ dataset — DeepFakes method", "v08_lipsync.mp4"),
    ("FaceForensics++ dataset — FaceSwap method",  "v09_lipsync.mp4"),
    ("FaceForensics++ dataset — NeuralTextures",   "v10_lipsync.mp4"),
    ("DF-TIMIT dataset — high quality",            "v11_faceswap.mp4"),
    ("DF-TIMIT dataset — low quality",             "v12_faceswap.mp4"),
]
for src, fname in DEEPFAKE_VIDEOS:
    add(
        src,
        ["videos", "deepfake", fname], "fake",
        manual=True
    )


# =============================================================================
# GROUND TRUTH MANIFEST (printed for copy-paste into test_accuracy.py)
# =============================================================================

def print_manifest():
    print(c(BOLD + CYAN, "\n# GROUND_TRUTH manifest — paste into test_accuracy.py:\n"))
    print("GROUND_TRUTH = [")
    for d in DOWNLOADS:
        rel = d["dest"].relative_to(SCRIPT_DIR.parent.parent)  # relative to repo root
        rel_str = str(rel).replace("\\", "/")
        print(f'    {{"file": "verified-stream/ai_service/test_media/{"/".join(d["dest"].parts[-3:])}", "label": "{d["label"]}"}},')
    print("]")


# =============================================================================
# DOWNLOADER
# =============================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}


def download_file(url: str, dest: Path) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 5000:
            return False, f"suspiciously small ({len(data)} bytes) — may be error page"
        dest.write_bytes(data)
        return True, f"{len(data) // 1024} KB"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)[:80]


def verify_files():
    """Check which files exist and which are missing."""
    auto   = [d for d in DOWNLOADS if not d["manual"]]
    manual = [d for d in DOWNLOADS if d["manual"]]
    present = [d for d in DOWNLOADS if d["dest"].exists()]
    missing_auto   = [d for d in auto   if not d["dest"].exists()]
    missing_manual = [d for d in manual if not d["dest"].exists()]

    print(c(BOLD, f"\n{'='*60}"))
    print(c(BOLD, "  TrueFrame Test Media — File Verification"))
    print(c(BOLD, f"{'='*60}"))
    print(f"\n  Total in manifest  : {len(DOWNLOADS)}")
    print(c(GREEN,  f"  Present on disk    : {len(present)}"))
    print(c(YELLOW, f"  Missing (auto)     : {len(missing_auto)}"))
    print(c(YELLOW, f"  Missing (manual)   : {len(missing_manual)}"))

    if missing_auto:
        print(c(YELLOW, "\n  Auto-download missing:"))
        for d in missing_auto:
            print(f"    ✗  {d['dest'].name}")

    if missing_manual:
        print(c(CYAN, "\n  Manual download missing:"))
        for d in missing_manual:
            print(f"    ⊙  {d['dest'].name}  ←  {d['url'][:70]}")

    coverage = len(present) / len(DOWNLOADS) * 100
    color = GREEN if coverage >= 80 else (YELLOW if coverage >= 50 else RED)
    print(c(color, f"\n  Coverage: {coverage:.1f}%"))
    print()
    return len(missing_auto), len(missing_manual)


def main():
    # Enable ANSI on Windows
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Download TrueFrame 100-media test suite"
    )
    parser.add_argument("--dry-run",      action="store_true", help="Print URLs, don't download")
    parser.add_argument("--verify-only",  action="store_true", help="Check which files exist")
    parser.add_argument("--manifest",     action="store_true", help="Print GROUND_TRUTH manifest")
    parser.add_argument("--delay",        type=float, default=0.6, help="Seconds between requests (default 0.6)")
    args = parser.parse_args()

    auto   = [d for d in DOWNLOADS if not d["manual"]]
    manual = [d for d in DOWNLOADS if d["manual"]]

    print(c(BOLD + CYAN, "\n╔══════════════════════════════════════════════╗"))
    print(c(BOLD + CYAN,  "║   TrueFrame 100-Media Test Suite Downloader  ║"))
    print(c(BOLD + CYAN,  "╚══════════════════════════════════════════════╝"))
    print(f"\n  Manifest     : {len(DOWNLOADS)} assets total")
    print(f"  Auto-fetch   : {len(auto)} images")
    print(f"  Manual-only  : {len(manual)} files (GAN, MJ, videos)")
    print(f"  Output dir   : {BASE}\n")

    if args.manifest:
        print_manifest()
        return

    if args.verify_only:
        verify_files()
        return

    if args.dry_run:
        print(c(BOLD, "  DRY RUN — no files will be downloaded\n"))
        for d in DOWNLOADS:
            tag = c(YELLOW, "[MANUAL]") if d["manual"] else c(GREEN, "[AUTO]  ")
            print(f"  {tag}  {d['dest'].parts[-1]}  ←  {d['url'][:70]}")
        return

    # ── Auto-download ─────────────────────────────────────────────────────────
    print(c(BOLD, f"  Downloading {len(auto)} auto-fetchable images...\n"))

    ok = err = skip = 0
    for d in auto:
        dest = d["dest"]
        if dest.exists():
            print(f"  {c(CYAN, '[SKIP]')}  {dest.name}")
            skip += 1
            continue

        ok_flag, info = download_file(d["url"], dest)
        if ok_flag:
            print(f"  {c(GREEN, '[OK]  ')}  {dest.name}  ({info})")
            ok += 1
        else:
            print(f"  {c(RED,   '[ERR] ')}  {dest.name}  — {info}")
            err += 1

        time.sleep(args.delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(c(BOLD, f"\n{'='*60}"))
    print(c(BOLD,  "  Download Summary"))
    print(c(BOLD, f"{'='*60}"))
    print(c(GREEN,  f"  Downloaded : {ok}"))
    print(c(CYAN,   f"  Skipped    : {skip}  (already present)"))
    print(c(RED,    f"  Errors     : {err}"))

    # ── Manual instructions ───────────────────────────────────────────────────
    print(c(BOLD + YELLOW, f"\n  ⚠  {len(manual)} files MUST be downloaded manually:\n"))

    categories = {}
    for d in manual:
        cat = d["dest"].parts[-3]  # e.g. "stylegan", "videos"
        categories.setdefault(cat, []).append(d)

    for cat, items in categories.items():
        print(c(BOLD, f"  [{cat.upper()}]"))
        for d in items:
            print(f"    → {d['dest'].name}")
            print(f"      Source: {d['url']}")
        print()

    print(c(BOLD, "  Instructions for manual assets:"))
    print("  1. StyleGAN (f01–f10): Visit https://thispersondoesnotexist.com")
    print("     Refresh the page 10 times and save each image with the filename shown above.")
    print()
    print("  2. FaceForensics++ (f11-f15, v08-v10): Request free academic access at")
    print("     https://github.com/ondyari/FaceForensics#access")
    print("     Download FaceSwap and DeepFakes splits, extract frames as JPEG.")
    print()
    print("  3. Midjourney (f16-f20): Go to https://www.midjourney.com/showcase")
    print("     Filter by 'Realistic' and save 5 portrait images.")
    print()
    print("  4. Adobe Firefly / SD (f21-f25): Go to https://firefly.adobe.com")
    print("     Prompt: 'photorealistic portrait photo of a person, high quality'")
    print("     Save 5 images. Alternatively use SD WebUI locally.")
    print()
    print("  5. Pexels Videos (v01-v07): Go to https://www.pexels.com/videos/")
    print("     Search 'person speaking', download HD MP4 files.")
    print()
    print("  6. DF-TIMIT Videos (v11-v12): Request access at")
    print("     https://www.idiap.ch/dataset/df-timit")
    print()

    # ── Verify coverage ───────────────────────────────────────────────────────
    verify_files()

    print(c(BOLD + CYAN, "  Next steps:"))
    print("  1. Complete manual downloads above")
    print("  2. Run: python test_accuracy.py  (from repo root)")
    print("  3. Run: python test_accuracy.py --fail-threshold 0.80")
    print()


if __name__ == "__main__":
    main()
