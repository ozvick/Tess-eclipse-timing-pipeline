import json, os
tics = [329770815, 257737073, 358232450, 18271803, 155540827, 224308601]
os.makedirs('timing_dump', exist_ok=True)
lines = ['cd ~/Desktop']
for t in tics:
    s = json.load(open(f'TIC_{t}_out/TIC{t}_summary.json'))
    P = s['period_days']
    data = next((d for d in (f'TIC_{t}_fits', f'TIC{t}_fits') if os.path.isdir(d)), None)
    if data is None:
        print("missing data folder for", t)
        continue
    out = f'timing_dump/TIC_{t}_out'
    lines.append(f'echo "=== TIC {t} ==="')
    lines.append(f'python3 apsidal_pipeline.py --tic {t} --data-dir {data} --outdir {out} '
                 f'--period-min {P*0.999:.8f} --period-max {P*1.001:.8f} > {out}.log 2>&1')
open('dump_runs.sh', 'w').write('\n'.join(lines) + '\n')
print(f"wrote dump_runs.sh with {(len(lines) - 1) // 2} systems")
