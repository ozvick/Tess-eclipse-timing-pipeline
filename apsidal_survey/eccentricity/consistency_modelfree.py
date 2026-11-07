"""Apsidal-motion consistency test (tides + GR -> implied k2) with MODEL-FREE eccentricity.

Penev's fitted e and omega are unreliable (fitting bug; also inclination). Here the orbit shape comes
from the pipeline's own measurements:
  e cos w : exact Kepler relation for the secondary-eclipse phase (solved numerically)
  e sin w : eclipse-duration ratio r = d_sec/d_pri = (1 + e sin w)/(1 - e sin w)  (exact for edge-on)
using v2.2 sub-bin half-widths computed here with the pipeline's own functions.
Masses, radii, separation and rotation still come from Penev's fits (PROVISIONAL: the bug may also
have affected the star sizes). Each system's k2 is computed with Penev's e, w and with the model-free
e, w, so the effect of the switch is visible.

Usage: python3 consistency_modelfree.py [TIC ...]     (default: the candidate list + Paul's systems)
Needs: apsidal_pipeline.py (v2.2) here, rerun_v2/ or paul_v2/ summaries, TIC_*_fits/ light curves,
binary_radii*.csv. Writes consistency_modelfree.csv.
"""
import csv, contextlib, glob, io, json, os, sys
from math import pi, sqrt, sin, cos, atan2, hypot
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.getcwd())
import apsidal_pipeline as ap

if 'Sub-bin precision (v2.2)' not in open('apsidal_pipeline.py').read():
    sys.exit('apsidal_pipeline.py here is not v2.2 -- run patch_pipeline_v2_2.py first')

GM_SUN, C, R_SUN = 1.32712440018e20, 2.99792458e8, 6.957e8
ERR_SCALE, ECW_ERR, ESW_ERR, N = 1.2, 0.001, 0.004, 3000
rng = np.random.default_rng(5)

# stellar parameters from every radius file available (later files win)
par = {}
for f in ('binary_radii.csv', 'binary_radii_more.csv', 'binary_radii_paul.csv', 'binary_radii_ecc.csv'):
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            par[int(float(r['tic']))] = r

PAUL = [117175528, 286566392, 468792279, 417728778]
if len(sys.argv) > 1:
    TICS = [int(x) for x in sys.argv[1:]]
else:
    cands = [int(r['tic']) for r in csv.DictReader(open('candidates_v2.csv'))] if os.path.exists('candidates_v2.csv') else []
    TICS = list(dict.fromkeys(cands + PAUL))


def M_anom(nu, e):
    E = 2 * atan2(sqrt(1 - e) * sin(nu / 2), sqrt(1 + e) * cos(nu / 2))
    return E - e * sin(E)


def sec_phase(ecw, esw):
    e, w = hypot(ecw, esw), atan2(esw, ecw)
    return ((M_anom(1.5 * pi - w, e) - M_anom(0.5 * pi - w, e)) % (2 * pi)) / (2 * pi)


def solve_ecw(phi, esw):
    f = lambda x: sec_phase(x, esw) - phi
    lim = sqrt(max(0.95 ** 2 - esw ** 2, 1e-6))
    try:
        return brentq(f, -lim, lim)
    except ValueError:
        return pi / 2 * (phi - 0.5)                 # fall back to first order


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
print(f"{'TIC':>10} {'P':>7} | {'Penev e':>7} {'k2 (Penev e,w)':>16} | {'e (model-free)':>14} {'ecosw':>7} {'esinw':>7} "
      f"{'k2 (model-free) [95%]':>26} | note")
for t in TICS:
    sp = next((f for f in (f'rerun_v2/TIC_{t}_out/TIC{t}_summary.json', f'paul_v2/TIC_{t}_out/TIC{t}_summary.json')
               if os.path.exists(f)), None)
    data = next((d for d in (f'TIC_{t}_fits', f'TIC{t}_fits') if os.path.isdir(d)), None)
    if sp is None or data is None or t not in par:
        continue
    s = json.load(open(sp))
    S, Se = s.get('apsidal_slope_min_per_1000d'), s.get('apsidal_slope_err_min_per_1000d')
    P, T0, phi = s.get('period_days'), s.get('t0_btjd'), s.get('secondary_phase')
    if None in (S, Se, P, T0, phi):
        continue
    with contextlib.redirect_stdout(io.StringIO()):
        time, flux, _, _ = ap.load_and_stitch(data)
    hw_p = ap.measure_eclipse_half_width(time, flux, P, T0)
    hw_s = ap.measure_eclipse_half_width(time, flux, P, T0 + phi * P)
    r = hw_s / hw_p
    esw = (r - 1) / (r + 1)
    ecw = solve_ecw(phi, esw)
    e_mf = hypot(ecw, esw)
    p = par[t]
    e_pen, w_pen = float(p['ecc']), np.radians(float(p['w_deg']))
    k_pen = implied_k2(S, Se, e_pen * cos(w_pen), e_pen * sin(w_pen), 0.0005, 0.0005, p, P)
    k_mf = implied_k2(S, Se, ecw, esw, ECW_ERR, ESW_ERR, p, P)
    notes = []
    if e_mf < 0.02:
        notes.append('near-circular: k2 poorly defined')
    if abs(esw) > 0.05 or e_mf > 0.2:
        notes.append('edge-on duration formula approximate')
    if k_mf[0] > 0.75:
        notes.append('k2 > physical max')
    elif k_mf[2] < 0:
        notes.append('slower than GR')
    elif 0.001 <= k_mf[1] <= 0.05:
        notes.append('normal k2')
    rows.append(dict(tic=t, P=P, drift=S, drift_err=Se, e_penev=e_pen, k2_penev=k_pen[1],
                     ecosw_mf=ecw, esinw_mf=esw, e_mf=e_mf, k2_mf_lo=k_mf[0], k2_mf=k_mf[1], k2_mf_hi=k_mf[2],
                     notes='; '.join(notes)))
    print(f"{t:>10} {P:7.3f} | {e_pen:7.3f} {k_pen[1]:16.4f} | {e_mf:14.3f} {ecw:+7.3f} {esw:+7.3f} "
          f"{k_mf[1]:9.4f} [{k_mf[0]:.4f}, {k_mf[2]:.4f}] | {'; '.join(notes)}", flush=True)

if rows:
    with open('consistency_modelfree.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} systems -> consistency_modelfree.csv  (radii/masses from Penev's fits: provisional)")
