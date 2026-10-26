"""Write rerun_v2.sh: the patched pipeline on every analyzed system, period fixed (+/-0.1%) at its
confirmed value, results in rerun_v2/. Systems that already have a summary there are skipped,
so the script can be re-run to resume an interrupted batch."""
import glob, json, os, re

EXCLUDE = {408651399,             # duplicate of 408651394
           240122720, 142047616}  # likely contact binaries (P < 0.3 d)

tics = set()
for d in glob.glob('TIC_*_out'):
    m = re.fullmatch(r'TIC_(\d+)_out', d)
    if m:
        tics.add(int(m.group(1)))

os.makedirs('rerun_v2', exist_ok=True)
lines, skipped, missing = ['cd ~/Desktop'], 0, []
for t in sorted(tics - EXCLUDE):
    out = f'rerun_v2/TIC_{t}_out'
    if os.path.exists(f'{out}/TIC{t}_summary.json'):
        skipped += 1
        continue
    jp = f'rerun_fixedP/TIC_{t}_out/TIC{t}_summary.json'
    jp = jp if os.path.exists(jp) else f'TIC_{t}_out/TIC{t}_summary.json'
    data = next((d for d in (f'TIC_{t}_fits', f'TIC{t}_fits') if os.path.isdir(d)), None)
    if not os.path.exists(jp) or data is None:
        missing.append(t)
        continue
    P = json.load(open(jp))['period_days']
    lines.append(f'echo "=== TIC {t} ==="')
    lines.append(f'python3 apsidal_pipeline.py --tic {t} --data-dir {data} --outdir {out} '
                 f'--period-min {P * 0.999:.8f} --period-max {P * 1.001:.8f} > {out}.log 2>&1')

open('rerun_v2.sh', 'w').write('\n'.join(lines) + '\n')
print(f'systems found: {len(tics)}, excluded: {len(tics & EXCLUDE)}, already done: {skipped}')
print(f'wrote rerun_v2.sh with {(len(lines) - 1) // 2} systems to run')
if missing:
    print(f'no summary or data folder for: {missing}')
