"""
Analyze existing test_results_100.json to get current baseline metrics
and identify what's failing.
"""
import json
import sys

with open('test_results_100.json') as f:
    results = json.load(f)

labeled = [r for r in results if not r.get('missing') and r.get('label') in ('real','fake')]
real_cases = [r for r in labeled if r['label']=='real']
fake_cases = [r for r in labeled if r['label']=='fake']

TP = sum(1 for r in fake_cases if r.get('correct') is True)
FN = sum(1 for r in fake_cases if r.get('correct') is False)
TN = sum(1 for r in real_cases if r.get('correct') is True)
FP = sum(1 for r in real_cases if r.get('correct') is False)
total = len(labeled)

acc = (TP+TN)/total if total else 0
prec = TP/(TP+FP) if (TP+FP) else 0
rec = TP/(TP+FN) if (TP+FN) else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
fpr = FP/len(real_cases) if real_cases else 0
fnr = FN/len(fake_cases) if fake_cases else 0

missing_fakes = sum(1 for r in results if r.get('missing') and r.get('label') in ('fake','real'))

print("=== CURRENT BASELINE METRICS ===")
print(f"Total labeled (non-missing): {total}")
print(f"Real cases: {len(real_cases)}, Fake cases (present): {len(fake_cases)}")
print(f"Missing fake/real test images: {missing_fakes}")
print(f"TP (fake->REJECTED): {TP}")
print(f"FN (fake->APPROVED/REVIEW): {FN}")
print(f"TN (real->APPROVED): {TN}")
print(f"FP (real->REJECTED/REVIEW): {FP}")
print(f"Accuracy: {acc*100:.1f}%")
print(f"Precision: {prec*100:.1f}%")
print(f"Recall (TPR): {rec*100:.1f}%")
print(f"F1 Score: {f1*100:.1f}%")
print(f"FPR (real images falsely blocked): {fpr*100:.1f}%")
print(f"FNR (fakes that slipped through): {fnr*100:.1f}%")

print()
print("=== FAILURES DETAIL ===")
for r in labeled:
    if r.get('correct') is False:
        fname = r['file'].split('/')[-1]
        print(f"  FAIL [{r['label']}] {fname}: score={r.get('score','?')}, verdict={r.get('verdict','?')}")

print()
print("=== SCORE DISTRIBUTION (real images) ===")
scores = sorted([r['score'] for r in real_cases if r.get('score') is not None])
if scores:
    print(f"  Min: {min(scores):.3f}, Max: {max(scores):.3f}, Mean: {sum(scores)/len(scores):.3f}")
    buckets = {'<0.10':0,'0.10-0.25':0,'0.25-0.40':0,'0.40-0.60':0,'>0.60':0}
    for s in scores:
        if s < 0.10: buckets['<0.10'] += 1
        elif s < 0.25: buckets['0.10-0.25'] += 1
        elif s < 0.40: buckets['0.25-0.40'] += 1
        elif s < 0.60: buckets['0.40-0.60'] += 1
        else: buckets['>0.60'] += 1
    for k,v in buckets.items():
        print(f"  Score {k}: {v} images")

print()
print("=== NOTE ON DEEPFAKE COVERAGE ===")
fake_missing = sum(1 for r in results if r.get('missing') and r.get('label')=='fake')
print(f"  {fake_missing} DEEPFAKE test images are MISSING (not downloaded)")
print("  Zero deepfake images tested! Fake detection is UNTESTED.")
print("  All current metrics are based on REAL IMAGE data only.")
print("  This is why accuracy looks decent but FNR is unmeasured.")
