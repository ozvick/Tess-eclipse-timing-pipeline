import csv, os
rows = [r for r in csv.DictReader(open('period_arbitration.csv')) if r['verdict'] == 'Penev']
print(f"systems to re-run: {len(rows)}")
os.makedirs('rerun_fixedP', exist_ok=True)
lines, missing = ['cd ~/Desktop'], []
for r in rows:
    t, P = r['tic'], float(r['P_prsa'])
    data = next((d for d in (f'TIC_{t}_fits', f'TIC{t}_fits') if os.path.isdir(d)), None)
    if data is None:
        missing.append(t)
        continue
    out = f'rerun_fixedP/TIC_{t}_out'
    lines.append(f'echo "=== TIC {t}  P={P:.6f} ==="')
    lines.append(f'python3 apsidal_pipeline.py --tic {t} --data-dir {data} --outdir {out} '
                 f'--period-min {P*0.999:.8f} --period-max {P*1.001:.8f} > {out}.log 2>&1')
open('rerun_fixedP.sh', 'w').write('\n'.join(lines) + '\n')
print(f"commands written: {len(rows) - len(missing)};  missing data folders: {missing}")
