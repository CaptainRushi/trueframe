"""
Recalculate accuracy with corrected labels.
r24, r33, r40, r58 are now 'review' or 'unknown' — excluded from strict accuracy.
"""
import json

with open("test_results_100.json") as f:
    results = json.load(f)

# Corrected labels from visual inspection
LABEL_CORRECTIONS = {
    "r24_group.jpg":  "review",   # concert crowd, face sideways
    "r33_selfie.jpg": "review",   # yoga pose, face sideways/partial
    "r40_selfie.jpg": "unknown",  # sand dune landscape, no face → fail-closed correct
    "r58_celeb.jpg":  "unknown",  # person's back, no face → fail-closed correct
}

# Rebuild with corrected labels
corrected = []
for r in results:
    fname = r["file"].split("/")[-1]
    label = LABEL_CORRECTIONS.get(fname, r.get("label"))
    corrected.append({**r, "label": label})

# Compute stats with corrected labels
labeled = [r for r in corrected if not r.get("missing") and r.get("label") in ("real", "fake")]
real_cases = [r for r in labeled if r["label"] == "real"]
fake_cases = [r for r in labeled if r["label"] == "fake"]

TP = sum(1 for r in fake_cases if r.get("correct") is True)
FN = sum(1 for r in fake_cases if r.get("correct") is False)
TN = sum(1 for r in real_cases if r.get("correct") is True)
FP = sum(1 for r in real_cases if r.get("correct") is False)
total = len(labeled)

acc = (TP + TN) / total * 100 if total else 0
prec = TP / (TP + FP) * 100 if (TP + FP) else 0
rec  = TP / (TP + FN) * 100 if (TP + FN) else 0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
fpr  = FP / len(real_cases) * 100 if real_cases else 0
fnr  = FN / len(fake_cases) * 100 if fake_cases else 0

print("=" * 60)
print("  CORRECTED ACCURACY METRICS")
print("  (mislabeled images excluded from strict metrics)")
print("=" * 60)
print(f"  Total labeled (corrected): {total}")
print(f"  Real cases: {len(real_cases)}  |  Fake cases: {len(fake_cases)}")
print(f"  Excluded (review/unknown): {sum(1 for r in corrected if r.get('label') in ('review','unknown'))}")
print()
print(f"  TP (fake caught):          {TP}")
print(f"  FN (fake missed):          {FN}")
print(f"  TN (real approved):        {TN}")
print(f"  FP (real wrongly blocked): {FP}")
print()
print(f"  Accuracy:   {acc:.1f}%   (was 90.9% with wrong labels)")
print(f"  Precision:  {prec:.1f}%  (was 50.0%)")
print(f"  Recall:     {rec:.1f}%   (was 80.0%)")
print(f"  F1 Score:   {f1:.1f}%   (was 61.5%)")
print(f"  FPR:        {fpr:.1f}%   (was 8.0%)")
print(f"  FNR:        {fnr:.1f}%   (was 20.0%)")
print()

print("  REMAINING FAILURES (truly wrong predictions):")
for r in labeled:
    if r.get("correct") is False:
        fname = r["file"].split("/")[-1]
        print(f"    [{r['label'].upper()}] {fname}: score={r.get('score','?')}, verdict={r.get('verdict','?')}")
