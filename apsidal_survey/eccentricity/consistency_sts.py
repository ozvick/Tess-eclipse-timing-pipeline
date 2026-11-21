"""Apsidal-motion consistency test (tides + GR -> implied k2) using STAR SHADOW eccentricities.

Same physics and Monte Carlo as consistency_modelfree.py (implied_k2 copied verbatim), same drifts and
stellar parameters; only the orbit shape changes: e cos w and e sin w come from STAR SHADOW v1.1.7a
(IJspeert et al. 2024 method) run per TESS cycle and combined in sts_combined.csv.

  --source phys : physical eclipse-model fit (default; not limited to the edge-on duration formula)
  --source form : STAR SHADOW timing formulae (duration-based, comparable to our model-free method)

Uncertainties: e cos w from STAR SHADOW's formal error (floor 0.001); e sin w from the cycle-to-cycle
scatter of the chosen estimate divided by sqrt(n_cycles), with a floor of 0.01 (the miss seen on the
simulated test light curve).

Writes consistency_sts.csv (same columns as consistency_modelfree.csv, so theory_k2_bands.py can read it
with --obs consistency_sts.csv).
"""
import argparse, csv, os
from math import pi, sqrt, sin, cos, atan2, hypot
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--source', choices=['phys', 'form'], default='phys')
ap.add_argument('--sts', default='sts_combined.csv')
ap.add_argument('--out', default='consistency_sts.csv')
args = ap.parse_args()

GM_SUN, C, R_SUN = 1.32712440018e20, 2.99792458e8, 6.957e8
ERR_SCALE, N = 1.2, 3000
ECW_FLOOR, ESW_FLOOR = 0.001, 0.01
rng = np.random.default_rng(5)

par = {}
for f in ('binary_radii.csv', 'binary_radii_more.csv', 'binary_radii_paul.csv', 'binary_radii_ecc.csv'):
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            par[int(float(r['tic']))] = r

mf = {int(r['tic']): r for r in csv.DictReader(open('consistency_modelfree.csv'))}


def M_anom(nu, e):
    E = 2 * atan2(sqrt(1 - e) * sin(nu / 2), sqrt(1 + e) * cos(nu / 2))
    return E - e * sin(E)


def dD_dw(w, e, P, h=1e-5):
    sep = lambda w: P * ((M_anom(1.5 * pi - w, e) - M_anom(0.5 * pi - w, e)) % (2 * pi)) / (2 * pi)
    return (sep(w + h) - sep(w - h)) / (2 * h)


def implied_k2(S, Se, ecw, esw, ecw_err, esw_err, p, P):
    M1, M2, R1, R2, a = (float(p[k]) for k in ('M1', 'M2', 'R1', 'R2', 'a_Rsun'))
    prot = float(p.get('prot') or 0)
    rot = P / prot if prot > 0 else 1.0
    out = []
    for _ in range(N):
        ec, es = ecw + ecw_err * rng.standard_normal(), esw + esw_err * rng.standard_normal()
        e = min(hypot(ec, es), 0.95)
        if e < 1e-4:
            continue
        w = atan2(es, ec)
        gr = 3 * GM_SUN * (M1 + M2) / (C ** 2 * a * R_SUN * (1 - e * e))
        f = (1 + 1.5 * e ** 2 + 0.125 * e ** 4) / (1 - e ** 2) ** 5
        g = 1 / (1 - e ** 2) ** 2
        tid = sum((Ri / a) ** 5 * (rot ** 2 * (1 + mj / mi) * g + 15 * (mj / mi) * f)
                  for mi, mj, Ri in ((M1, M2, R1), (M2, M1, R2)))
        Sd = abs(S + Se * ERR_SCALE * rng.standard_normal())
        wdot = (Sd / 1000 / 1440) / abs(dD_dw(w, e, P))
        out.append((wdot * P / (2 * pi) - gr) / tid)
    return np.percentile(out, [2.5, 50, 97.5]) if out else (np.nan, np.nan, np.nan)


rows = []
print(f"STAR SHADOW e sin w source: {args.source}\n")
print(f"{'TIC':>10} {'P':>7} | {'e mf':>6} {'k2 (mf)':>8} | {'e STS':>6} {'ecosw':>7} {'esinw':>7} "
      f"{'k2 (STS) [95%]':>26} | note")
for s in csv.DictReader(open(args.sts)):
    t = int(s['tic'])
    if int(s['n_cycles']) == 0 or t not in mf or t not in par:
        print(f"{t:>10}  skipped (no STAR SHADOW result, drift or stellar parameters)")
        continue
    m, p = mf[t], par[t]
    n = int(s['n_cycles'])
    ecw = float(s[f'ecosw_{args.source}'])
    esw = float(s[f'esinw_{args.source}'])
    ecw_err = max(float(s['ecosw_form_err']), ECW_FLOOR)
    sd_key = 'esinw_phys_cycle_std' if args.source == 'phys' else 'esinw_cycle_std'
    sd = float(s[sd_key]) if s[sd_key] not in ('', 'nan') else np.nan
    esw_err = max(sd / sqrt(n) if np.isfinite(sd) else 0.0, ESW_FLOOR)
    S, Se, P = float(m['drift']), float(m['drift_err']), float(m['P'])
    k = implied_k2(S, Se, ecw, esw, ecw_err, esw_err, p, P)
    e = hypot(ecw, esw)
    notes = []
    if e < 0.02:
        notes.append('near-circular: k2 poorly defined')
    if k[0] > 0.75:
        notes.append('k2 > physical max')
    elif k[2] < 0:
        notes.append('slower than GR')
    elif 0.001 <= k[1] <= 0.05:
        notes.append('normal k2')
    notes.append(f'STAR SHADOW {args.source}, {n} cycles')
    rows.append(dict(tic=t, P=P, drift=S, drift_err=Se, e_penev=m['e_penev'], k2_penev=m['k2_penev'],
                     ecosw_mf=ecw, esinw_mf=esw, e_mf=e, k2_mf_lo=k[0], k2_mf=k[1], k2_mf_hi=k[2],
                     notes='; '.join(notes)))
    print(f"{t:>10} {P:7.3f} | {float(m['e_mf']):6.3f} {float(m['k2_mf']):8.4f} | {e:6.3f} {ecw:+7.3f} {esw:+7.3f} "
          f"{k[1]:9.4f} [{k[0]:.4f}, {k[2]:.4f}] | {'; '.join(notes)}", flush=True)

with open(args.out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\n{len(rows)} systems -> {args.out}   (columns named *_mf hold the STAR SHADOW values, for theory_k2_bands.py)")
