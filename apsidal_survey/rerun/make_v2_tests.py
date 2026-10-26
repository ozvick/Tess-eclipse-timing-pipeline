"""Write run_v2_tests.sh: the patched pipeline on the test set, each system's period fixed (+/-0.1%)
at its current value, results in v2_test/ so nothing existing is overwritten."""
import json, os

TICS = [197755658, 289044378, 257737073, 220420534, 341029467,
        125857387, 381857817, 89522181, 160328766]
os.makedirs('v2_test', exist_ok=True)
lines = ['cd ~/Desktop']
for t in TICS:
    jp = f'rerun_fixedP/TIC_{t}_out/TIC{t}_summary.json'
    jp = jp if os.path.exists(jp) else f'TIC_{t}_out/TIC{t}_summary.json'
    data = next((d for d in (f'TIC_{t}_fits', f'TIC{t}_fits') if os.path.isdir(d)), None)
    if not os.path.exists(jp) or data is None:
        print(f'{t}: missing summary or data folder, skipped')
        continue
    P = json.load(open(jp))['period_days']
    out = f'v2_test/TIC_{t}_out'
    lines.append(f'echo "=== TIC {t} ==="')
    lines.append(f'python3 apsidal_pipeline.py --tic {t} --data-dir {data} --outdir {out} '
                 f'--period-min {P * 0.999:.8f} --period-max {P * 1.001:.8f} > {out}.log 2>&1')
open('run_v2_tests.sh', 'w').write('\n'.join(lines) + '\n')
print(f'wrote run_v2_tests.sh with {(len(lines) - 1) // 2} systems')
