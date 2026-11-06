"""Recompute primary and secondary eclipse half-widths for every system in rerun_v2/ with the
v2.2 (sub-bin) width measurement, without rerunning the pipeline.

Uses the pipeline's own loader and measure_eclipse_half_width() from the patched
apsidal_pipeline.py in the current folder, with each system's final period, epoch and secondary
phase from its summary, exactly as the final pipeline pass does. rerun_v2/ is not modified.
Writes widths_v22.csv. Then: python3 compare_ecc_components.py --widths widths_v22.csv
"""
import csv, glob, json, os, sys, contextlib, io
sys.path.insert(0, os.getcwd())
import numpy as np
import apsidal_pipeline as ap

src = open('apsidal_pipeline.py').read()
if 'Sub-bin precision (v2.2)' not in src:
    sys.exit('apsidal_pipeline.py here is not v2.2 -- run patch_pipeline_v2_2.py first')

rows, skipped = [], []
paths = sorted(glob.glob('rerun_v2/TIC_*_out/TIC*_summary.json'))
for k, p in enumerate(paths, 1):
    s = json.load(open(p))
    t = int(s['tic'])
    P, T0, ph = s.get('period_days'), s.get('t0_btjd'), s.get('secondary_phase')
    old_p, old_s = s.get('primary_half_width_phase'), s.get('secondary_half_width_phase')
    data = next((d for d in (f'TIC_{t}_fits', f'TIC{t}_fits') if os.path.isdir(d)), None)
    if None in (P, T0, ph, old_p, old_s) or data is None:
        skipped.append(t); continue
    with contextlib.redirect_stdout(io.StringIO()):          # silence the loader's messages
        time, flux, _, _ = ap.load_and_stitch(data)
    hw_p = ap.measure_eclipse_half_width(time, flux, P, T0)
    hw_s = ap.measure_eclipse_half_width(time, flux, P, T0 + ph * P)
    rows.append(dict(tic=t, P=P, hw_p_old=old_p, hw_s_old=old_s, hw_p=hw_p, hw_s=hw_s,
                     ratio_old=old_s / old_p, ratio_new=hw_s / hw_p))
    if k % 20 == 0:
        print(f'{k}/{len(paths)} done', flush=True)

with open('widths_v22.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
d = np.array([r['ratio_new'] - r['ratio_old'] for r in rows])
rel_p = np.array([r['hw_p'] / r['hw_p_old'] - 1 for r in rows])
print(f'\nrecomputed {len(rows)} systems (skipped {len(skipped)}: no secondary or no data) -> widths_v22.csv')
print(f'primary half-width change: median {100 * np.median(rel_p):+.1f}%, 90% within {100 * np.percentile(np.abs(rel_p), 90):.1f}%')
print(f'width-ratio change: median {np.median(d):+.4f}, 90% within {np.percentile(np.abs(d), 90):.4f}')
