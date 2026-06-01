"""
TrueFrame — Baseline Evaluation on Existing Test Media
=======================================================
Runs the current detection pipeline on all available test media
and reports per-category performance metrics.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import analyze

TEST_MEDIA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_media")

categories = {
    "real_portraits": os.path.join(TEST_MEDIA_DIR, "real", "portraits"),
    "real_selfies": os.path.join(TEST_MEDIA_DIR, "real", "selfies"),
    "real_celebrities": os.path.join(TEST_MEDIA_DIR, "real", "celebrities"),
    "real_group": os.path.join(TEST_MEDIA_DIR, "real", "group"),
    "real_lowlight": os.path.join(TEST_MEDIA_DIR, "real", "lowlight"),
    "fake_stylegan": os.path.join(TEST_MEDIA_DIR, "deepfake", "stylegan"),
}

all_results = {}
overall_real_scores = []
overall_fake_scores = []

for cat_name, cat_dir in categories.items():
    if not os.path.isdir(cat_dir):
        print(f"Skipping {cat_name}: {cat_dir} not found")
        continue
    paths = [
        os.path.join(cat_dir, f)
        for f in sorted(os.listdir(cat_dir))
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    if not paths:
        print(f"No images found in {cat_name}")
        continue

    cat_scores = []
    cat_verdicts = []
    for p in paths:
        try:
            r = analyze(p)
            cat_scores.append(r["final_score"])
            cat_verdicts.append(r["verdict"])
            if "fake" in cat_name:
                overall_fake_scores.append(r["final_score"])
            else:
                overall_real_scores.append(r["final_score"])
        except Exception as e:
            print(f"  Error analyzing {p}: {e}")
            cat_scores.append(-1)

    cat_is_fake = "fake" in cat_name
    correct = sum(1 for s, v in zip(cat_scores, cat_verdicts) if cat_is_fake and v == "REJECTED" or not cat_is_fake and v == "APPROVED")
    accuracy = correct / len(paths) if paths else 0

    all_results[cat_name] = {
        "count": len(paths),
        "mean_score": float(np.mean(cat_scores)),
        "std_score": float(np.std(cat_scores)),
        "min_score": float(np.min(cat_scores)),
        "max_score": float(np.max(cat_scores)),
        "accuracy": float(accuracy),
        "scores": [round(s, 4) for s in cat_scores],
        "verdicts": cat_verdicts,
    }

    print(f"{cat_name:20s} | count={len(paths):3d} | mean={all_results[cat_name]['mean_score']:.4f} "
          f"std={all_results[cat_name]['std_score']:.4f} "
          f"acc={all_results[cat_name]['accuracy']:.2%}")

# Also test individual deepfake images
print("\n--- Individual deepfake test images ---")
df_dir = os.path.join(TEST_MEDIA_DIR, "deepfake")
individual_results = []
for fname in sorted(os.listdir(df_dir)):
    path = os.path.join(df_dir, fname)
    if os.path.isfile(path) and fname.endswith((".jpg", ".png")):
        r = analyze(path)
        individual_results.append({"file": fname, "score": r["final_score"], "verdict": r["verdict"], "signals": r["signals"][:5]})
        print(f"  {fname:40s} score={r['final_score']:.4f} verdict={r['verdict']}")

# Compute aggregate metrics
print("\n=== AGGREGATE METRICS ===")
if overall_real_scores:
    print(f"Real images:   count={len(overall_real_scores)} mean_score={np.mean(overall_real_scores):.4f}")
if overall_fake_scores:
    print(f"Fake images:   count={len(overall_fake_scores)} mean_score={np.mean(overall_fake_scores):.4f}")

# Save results
output = {
    "categories": all_results,
    "individual_deepfake": individual_results,
    "real_mean": float(np.mean(overall_real_scores)) if overall_real_scores else 0,
    "fake_mean": float(np.mean(overall_fake_scores)) if overall_fake_scores else 0,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
