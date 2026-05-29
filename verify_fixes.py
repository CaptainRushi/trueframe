with open('verified-stream/ai_service/main.py', encoding='utf-8') as f:
    content = f.read()

checks = [
    ('MTCNN-only has_faces', '_get_mtcnn_crops(frames)'),
    ('HF confidence penalty', 'hf_confidence_penalty_applied'),
    ('No-face base 0.75 (skin-tone)', 'no_face_base = 0.75'),
    ('No-face base 0.80 (no-skin)', 'no_face_base = 0.80'),
    ('Split boost when HF active', 'if _model_active:'),
    ('dima806 fake_idx override', 'dima806/deepfake_vs_real_image_detection'),
]
for name, token in checks:
    found = token in content
    status = 'OK' if found else 'MISSING'
    print(f'  {status}  {name}')
