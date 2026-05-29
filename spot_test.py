import subprocess, json

tests = [
    ('verified-stream/ai_service/test_media/real/no_face/r41_landscape.jpg', 'fake', 'landscape r41'),
    ('verified-stream/ai_service/test_media/real/no_face/r42_landscape.jpg', 'fake', 'landscape r42'),
    ('verified-stream/ai_service/test_media/real/no_face/r43_landscape.jpg', 'fake', 'landscape r43'),
    ('verified-stream/ai_service/test_media/real/portraits/r01_portrait.jpg', 'real', 'portrait r01'),
    ('verified-stream/ai_service/test_media/real/celebrities/r60_celeb.jpg', 'real', 'celeb r60'),
    ('verified-stream/ai_service/test_media/real/celebrities/r54_celeb.jpg', 'real', 'celeb r54'),
    ('verified-stream/ai_service/test_media/real/selfies/r34_selfie.jpg',    'real', 'selfie r34'),
]

pass_count = fail_count = 0
for path, expected, desc in tests:
    r = subprocess.run(['python','verified-stream/ai_service/main.py', path],
                       capture_output=True, text=True, timeout=300)
    stdout = r.stdout.strip()
    try:
        d = json.loads(stdout.split('\n')[-1])
        verdict = d.get('verdict', 'ERR')
        score   = d.get('final_score', 0)
        ok = (verdict == 'APPROVED' and expected == 'real') or \
             (verdict == 'REJECTED' and expected == 'fake')
        status = 'PASS' if ok else 'FAIL'
        if ok: pass_count += 1
        else:  fail_count += 1
        print(f'{status}  score={score:.3f}  {verdict:<15}  {desc}')
    except Exception as e:
        print(f'ERR   {desc}: {e}')
        fail_count += 1

print(f'\n{pass_count}/{pass_count+fail_count} spot tests passed')
