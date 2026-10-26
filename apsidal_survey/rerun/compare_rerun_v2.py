"""Compare the full patched re-run (rerun_v2/) with the old results, Penev's ETVs and the mirror method.

Prints: verdict transitions, the systems whose drift changed most, and how agreement with
Penev's ETVs and the mirror method changed. Writes rerun_v2_comparison.csv.
"""
import csv, glob, json, os
from collections import Counter
from math import hypot

def old_summary(t):
    for p in (f'rerun_fixedP/TIC_{t}_out/TIC{t}_summary.json', f'TIC_{t}_out/TIC{t}_summary.json'):
        if os.path.exists(p):
            return json.load(open(p))
    return {}

def load_csv(fname, key_val):
    if not os.path.exists(fname):
        return {}
    return {int(r['tic']): key_val(r) for r in csv.DictReader(open(fname))}

penev = load_csv('etv_comparison.csv', lambda r: (float(r['pen']), float(r['pen_err'])))
mirror = load_csv('retime_all.csv', lambda r: (float(r['mirror']), float(r['mirror_err'])))

rows = []
for p in sorted(glob.glob('rerun_v2/TIC_*_out/TIC*_summary.json')):
    n = json.load(open(p))
    t = int(n['tic'])
    o = old_summary(t)
    rows.append(dict(
        tic=t,
        old_verdict=o.get('verdict'), new_verdict=n.get('verdict'),
        old_S=o.get('apsidal_slope_min_per_1000d'), old_Se=o.get('apsidal_slope_err_min_per_1000d'),
        new_S=n.get('apsidal_slope_min_per_1000d'), new_Se=n.get('apsidal_slope_err_min_per_1000d'),
        passes=n.get('n_passes'), conv=n.get('converged_shift_min'), tol=n.get('converged_tolerance_min'),
        width_ratio=(n['secondary_half_width_phase'] / n['primary_half_width_phase']
                     if n.get('primary_half_width_phase') else None),
        bound_hits=(n.get('n_primary_at_shift_bound') or 0) + (n.get('n_secondary_at_shift_bound') or 0),
        P0_shift_min=abs(n['first_pass_period_days'] - n['period_days']) * 1440
        if n.get('first_pass_period_days') else None,
    ))

print(f"systems with patched results: {len(rows)}\n")
print("verdict transitions (old -> new):")
for (a, b), c in Counter((r['old_verdict'], r['new_verdict']) for r in rows).most_common():
    print(f"  {c:4d}  {a} -> {b}")

def z(a, ae, b, be):
    return (a - b) / hypot(ae or 0, be or 0) if None not in (a, b) and hypot(ae or 0, be or 0) > 0 else None

both = [r for r in rows if r['old_S'] is not None and r['new_S'] is not None]
for r in both:
    r['change_z'] = z(r['new_S'], r['new_Se'], r['old_S'], r['old_Se'])
big = sorted([r for r in both if r['change_z'] is not None and abs(r['change_z']) > 3],
             key=lambda r: -abs(r['change_z']))
print(f"\ndrift changed by more than 3 sigma: {len(big)} of {len(both)}")
print(f"{'TIC':>10} {'old':>16} {'new':>16} {'z':>6} {'passes':>6} {'w_s/w_p':>7}  verdict old > new")
for r in big[:40]:
    print(f"{r['tic']:>10} {r['old_S']:+8.2f}+/-{r['old_Se'] or 0:5.2f} {r['new_S']:+8.2f}+/-{r['new_Se'] or 0:5.2f} "
          f"{r['change_z']:+6.1f} {r['passes'] or 0:>6} {r['width_ratio'] or 0:7.2f}  {r['old_verdict']} > {r['new_verdict']}")

for label, ref in (("Penev's ETVs", penev), ("mirror method", mirror)):
    zo, zn = [], []
    for r in both:
        if r['tic'] in ref:
            v, ve = ref[r['tic']]
            a, b = z(r['old_S'], r['old_Se'], v, ve), z(r['new_S'], r['new_Se'], v, ve)
            if a is not None and b is not None:
                zo.append(abs(a)); zn.append(abs(b))
    if zo:
        frac = lambda zs, k: sum(1 for x in zs if x < k)
        print(f"\nagreement with {label} (n={len(zo)}):  within 2 sigma  old {frac(zo, 2)}  new {frac(zn, 2)}   "
              f"| within 3 sigma  old {frac(zo, 3)}  new {frac(zn, 3)}")

unconv = [r for r in rows if r['conv'] is not None and r['tol'] is not None and r['conv'] > r['tol']]
hits = [r['tic'] for r in rows if r['bound_hits']]
print(f"\nnot converged within tolerance: {[r['tic'] for r in unconv]}")
print(f"systems with shift-bound hits in the final pass: {hits}")

with open('rerun_v2_comparison.csv', 'w', newline='') as f:
    keys = list(rows[0].keys()) + ['change_z']
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
print("\nwrote rerun_v2_comparison.csv")
