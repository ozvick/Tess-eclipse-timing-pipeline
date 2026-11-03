"""Compare the patched pipeline's drifts (paul_v2/) with Penev's ETV drifts for Paul's systems,
and write own_drifts.csv (days per epoch, errors x1.2) for: python3 paul_k2.py own_drifts.csv binary_radii_paul.csv
Needs binary_radii_paul.csv and paul_etv_drifts.csv (both from kartof).
"""
import csv, json, os
from math import hypot

ERR_SCALE = 1.2
P = {int(r['tic']): float(r['per']) for r in csv.DictReader(open('binary_radii_paul.csv'))}
pen = {}
if os.path.exists('paul_etv_drifts.csv'):
    for r in csv.DictReader(open('paul_etv_drifts.csv')):
        if r.get('sp') and int(r['tic']) in P:
            conv = 1440 / P[int(r['tic'])] * 1000
            pen[int(r['tic'])] = (-float(r['sp']) * conv, float(r['sp_err']) * conv)   # his C-O -> O-C

rows = []
print(f"{'TIC':>10} {'verdict':>22} {'n_sec':>5} {'yours [min/1000d]':>18} {'Penev':>15} {'z':>6} {'curv. p':>8}")
for t in sorted(P):
    p = f'paul_v2/TIC_{t}_out/TIC{t}_summary.json'
    if not os.path.exists(p):
        print(f"{t:>10}  no result (see paul_v2/TIC_{t}_out.log)"); continue
    s = json.load(open(p))
    S, Se = s.get('apsidal_slope_min_per_1000d'), s.get('apsidal_slope_err_min_per_1000d')
    cp = s.get('primary_curvature_sector_p')
    pv = pen.get(t)
    z = (S - pv[0]) / hypot(Se * ERR_SCALE, pv[1]) if S is not None and Se and pv else None
    print(f"{t:>10} {str(s.get('verdict')):>22} {str(s.get('n_secondary_qc')):>5} "
          f"{'--' if S is None else f'{S:+8.2f} +/- {Se:5.2f}':>18} "
          f"{'--' if not pv else f'{pv[0]:+7.2f} +/- {pv[1]:4.2f}':>15} "
          f"{'--' if z is None else f'{z:+6.1f}':>6} {'--' if cp is None else f'{cp:8.4f}':>8}")
    if S is not None and Se:
        conv = 1440 / P[t] * 1000
        rows.append(dict(tic=t, sp=S / conv, sp_err=Se * ERR_SCALE / conv))

with open('own_drifts.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['tic', 'sp', 'sp_err'])
    w.writeheader(); w.writerows(rows)
print(f"\nwrote own_drifts.csv ({len(rows)} systems) -> python3 paul_k2.py own_drifts.csv binary_radii_paul.csv")
