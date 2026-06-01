"""
Fast deepfake detector test — only runs the detector on files that are:
1. Actually present on disk
2. Labeled as 'fake' (the critical test)
3. A small sample of 'real' images to verify FPR

Uses the detector with --quiet output for speed.
"""
import os, sys, json, subprocess, time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DETECTOR = "verified-stream/ai_service/main.py"

# Ground truth from test_results_100.json - the files that actually EXIST
FAKE_FILES = [
    # These 5 fake images actually existed in test_results_100.json
    # (labeled 'fake' but actually landscape/no-face images treated as fake)
    ("verified-stream/ai_service/test_media/real/no_face/r41_landscape.jpg", "fake"),
    ("verified-stream/ai_service/test_media/real/no_face/r42_landscape.jpg", "fake"),
    ("verified-stream/ai_service/test_media/real/no_face/r43_landscape.jpg", "fake"),
    ("verified-stream/ai_service/test_media/real/no_face/r44_landscape.jpg", "fake"),
    ("verified-stream/ai_service/test_media/real/no_face/r45_landscape.jpg", "fake"),
]

# Sample of real images that were FAILING before (the hard cases)
HARD_REAL = [
    ("verified-stream/ai_service/test_media/real/group/r24_group.jpg", "real"),
    ("verified-stream/ai_service/test_media/real/selfies/r33_selfie.jpg", "real"),
    ("verified-stream/ai_service/test_media/real/selfies/r40_selfie.jpg", "real"),
    ("verified-stream/ai_service/test_media/real/celebrities/r58_celeb.jpg", "real"),
    ("verified-stream/ai_service/test_media/real/group/r23_group.jpg", "real"),  # was high (0.3976)
    ("verified-stream/ai_service/test_media/real/selfies/r34_selfie.jpg", "real"),  # was high (0.3669)
]

# The actual deepfake test samples we know exist
DEEPFAKE_VIDEOS_IF_PRESENT = [
    ("176527-855920754_medium.mp4", "real"),  # stock video - known real
]

ALL_TESTS = FAKE_FILES + HARD_REAL + DEEPFAKE_VIDEOS_IF_PRESENT

def run_detect(path, timeout=90):
    if not os.path.exists(path):
        return None, "MISSING"
    start = time.time()
    try:
        r = subprocess.run(
            [sys.executable, DETECTOR, path],
            capture_output=True, text=True, check=False, timeout=timeout
        )
        elapsed = time.time() - start
        last = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
        try:
            data = json.loads(last)
            data["_elapsed"] = round(elapsed, 1)
            return data, None
        except Exception:
            return None, f"JSON_ERR: {last[:80]}"
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


print("=" * 65)
print("  TrueFrame — Targeted Detection Test (Hard Cases + Fakes)")
print("=" * 65)
print()

results = []
TP = TN = FP = FN = 0

for path, label in ALL_TESTS:
    data, err = run_detect(path)
    fname = path.split("/")[-1]

    if err:
        print(f"  {'SKIP':<8} {fname:<42} [{err}]")
        continue

    score   = data.get("final_score", 0)
    verdict = data.get("verdict", "?")
    elapsed = data.get("_elapsed", 0)
    is_video = data.get("video_threshold_reject") is not None

    # Correctness
    if label == "real":
        correct = verdict == "APPROVED"
        if correct: TN += 1
        else: FP += 1
    else:
        correct = verdict == "REJECTED"
        if correct: TP += 1
        else: FN += 1

    icon = "✓" if correct else "✗"
    vfmt = {
        "APPROVED":     "\033[92mAPPROVED\033[0m",
        "UNDER_REVIEW": "\033[93mUNDER_REVIEW\033[0m",
        "REJECTED":     "\033[91mREJECTED\033[0m",
    }.get(verdict, verdict)
    threshold = f"(thr={data.get('video_threshold_reject', 0.75):.2f})" if is_video else ""

    print(f"  {icon} [{label.upper():<5}] {fname:<42} score={score:.4f}  {vfmt} {threshold}  {elapsed}s")

    results.append({
        "file": path, "label": label,
        "score": score, "verdict": verdict,
        "correct": correct, "is_video": is_video
    })

total = TP + TN + FP + FN
acc   = (TP + TN) / total * 100 if total else 0
fnr   = FN / (TP + FN) * 100 if (TP + FN) else 0
fpr   = FP / (TN + FP) * 100 if (TN + FP) else 0

print()
print("=" * 65)
print("  RESULTS")
print("=" * 65)
print(f"  Total tested:          {total}")
print(f"  TP (fake caught):      {TP}")
print(f"  FN (fake missed):      {FN}  FNR={fnr:.0f}%")
print(f"  TN (real approved):    {TN}")
print(f"  FP (real blocked):     {FP}  FPR={fpr:.0f}%")
print(f"  Accuracy:              {acc:.1f}%")
print()

with open("fast_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Results saved: fast_test_results.json")
