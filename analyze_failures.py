import json

with open('test_results_100.json') as f:
    results = json.load(f)

print('=== REAL IMAGES INCORRECTLY REJECTED (False Positives) ===')
fps = [r for r in results
       if r.get('label') == 'real' and r.get('correct') == False and not r.get('missing')]
fps.sort(key=lambda r: r['score'], reverse=True)
for r in fps:
    fname = r['file'].split('/')[-1].split('\\')[-1]
    bad_sigs = [s for s in r.get('signals', [])
                if s not in ('content_type_real_photo', 'huggingface_model_used',
                             'borderline_needs_review', 'synthetic_generation_signal')]
    print(f"  score={r['score']:.3f}  {r['verdict']:<15}  {fname:<28}  {', '.join(bad_sigs[:4])}")

print(f"\nTotal FP: {len(fps)}/52 real images wrongly rejected")

print('\n=== FAKE IMAGES NOT REJECTED (False Negatives) ===')
fns = [r for r in results
       if r.get('label') == 'fake' and r.get('correct') == False and not r.get('missing')]
fns.sort(key=lambda r: r['score'])
for r in fns:
    fname = r['file'].split('/')[-1].split('\\')[-1]
    sigs = r.get('signals', [])[:4]
    print(f"  score={r['score']:.3f}  {r['verdict']:<15}  {fname:<28}  {', '.join(sigs)}")

print(f"\nTotal FN: {len(fns)}/10 fake images not rejected")

print('\n=== SIGNAL FREQUENCY (top offenders in FP cases) ===')
sig_count = {}
for r in fps:
    for s in r.get('signals', []):
        if s not in ('content_type_real_photo', 'huggingface_model_used',
                     'borderline_needs_review', 'synthetic_generation_signal',
                     'face_detection_missed_portrait', 'fail_closed'):
            sig_count[s] = sig_count.get(s, 0) + 1

for sig, cnt in sorted(sig_count.items(), key=lambda x: -x[1]):
    print(f"  {cnt:3d}x  {sig}")
