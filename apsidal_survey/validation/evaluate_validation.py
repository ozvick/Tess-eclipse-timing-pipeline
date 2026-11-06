"""How well does the pipeline recover |e sin w| (eclipse-duration ratio) and |e cos w| (secondary
phase) on synthetic light curves with known values? Uses exactly the formulas of
compare_ecc_components.py:  |e cos w| ~ (pi/2)|phi_sec - 0.5|,  |e sin w| ~ |r - 1|/(r + 1).
Reads val/truth.csv and val/out/TIC_*_out/TIC*_summary.json. Writes val/validation_results.csv.
"""
import csv, json, os
from math import pi
import numpy as np

truth = list(csv.DictReader(open('val/truth.csv')))
rows = []
print(f"{'TIC':>10} {'P':>4} {'shape':>6} {'sb':>4} | {'esinw true':>10} {'est':>7} {'err':>8} | "
      f"{'ecosw true':>10} {'est':>7} {'err':>8} | {'w_s/w_p':>7}")
for t in truth:
    p = f"val/out/TIC_{t['tic']}_out/TIC{t['tic']}_summary.json"
    if not os.path.exists(p):
        print(f"{t['tic']:>10}  no result (see val/out/TIC_{t['tic']}.log)"); continue
    s = json.load(open(p))
    phs, wp, ws = s.get('secondary_phase'), s.get('primary_half_width_phase'), s.get('secondary_half_width_phase')
    if None in (phs, wp, ws) or not wp:
        print(f"{t['tic']:>10}  missing secondary phase or widths"); continue
    r = ws / wp
    esw_est, ecw_est = abs((r - 1) / (r + 1)), abs(pi / 2 * (phs - 0.5))
    esw_t, ecw_t = abs(float(t['esinw'])), abs(float(t['ecosw']))
    row = dict(tic=t['tic'], P=float(t['P']), shape=t['shape'], sbratio=float(t['sbratio']),
               esinw_true=esw_t, esinw_est=esw_est, esinw_err=esw_est - esw_t,
               ecosw_true=ecw_t, ecosw_est=ecw_est, ecosw_err=ecw_est - ecw_t, width_ratio=r,
               depth_ratio=(s.get('secondary_depth') or np.nan) / (s.get('primary_depth') or np.nan))
    rows.append(row)
    print(f"{row['tic']:>10} {row['P']:4.1f} {row['shape']:>6} {row['sbratio']:4.1f} | {esw_t:10.3f} {esw_est:7.4f} "
          f"{row['esinw_err']:+8.4f} | {ecw_t:10.3f} {ecw_est:7.4f} {row['ecosw_err']:+8.4f} | {r:7.3f}")

if rows:
    with open('val/validation_results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('\nSummary (median |error| and worst |error|):')
    groups = [('all', lambda r: True),
              ('spherical, similar depths', lambda r: r['shape'] == 'sphere' and r['sbratio'] > 0.5),
              ('spherical, unequal depths', lambda r: r['shape'] == 'sphere' and r['sbratio'] < 0.5),
              ('distorted (Roche), P = 2.5 d', lambda r: r['shape'] == 'roche')]
    for name, sel in groups:
        g = [r for r in rows if sel(r)]
        if g:
            es = np.abs([r['esinw_err'] for r in g]); ec = np.abs([r['ecosw_err'] for r in g])
            print(f"  {name:30s} n={len(g):2d}  |e sin w| error: median {np.median(es):.4f}, worst {es.max():.4f}   "
                  f"|e cos w| error: median {np.median(ec):.4f}, worst {ec.max():.4f}")
    print('wrote val/validation_results.csv')
