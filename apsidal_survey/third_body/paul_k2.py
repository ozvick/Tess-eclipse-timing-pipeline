"""On kartof: is the apsidal motion seen in Penev's ETVs consistent with tides + GR?

Reads ~/paul_etv_drifts.csv (from etv_drifts_v2.py) and ~/binary_radii_paul.csv
(from extract_radii_list.py), which also supplies e and omega from Penev's fits.
For each system: the secondary-minus-primary drift, the implied apsidal-motion constant k2
(tides + GR, Monte Carlo over the drift uncertainty), the observed apsidal period U, and the
U that GR alone would give. Typical stellar k2 is ~0.002-0.01; above ~0.75 is impossible,
which would point to an extra driver such as a third body.
"""
import csv, os
from math import pi, sqrt, sin, cos, atan2, radians
import numpy as np

GM_SUN, C, R_SUN = 1.32712440018e20, 2.99792458e8, 6.957e8
rng = np.random.default_rng(1)
import sys
home = os.path.expanduser('~')
f_etv = sys.argv[1] if len(sys.argv) > 1 else f'{home}/paul_etv_drifts.csv'
f_par = sys.argv[2] if len(sys.argv) > 2 else f'{home}/binary_radii_paul.csv'
etv = {int(r['tic']): r for r in csv.DictReader(open(f_etv)) if r.get('sp')}
par = {int(r['tic']): r for r in csv.DictReader(open(f_par))}


def mean_anom(nu, e):
    E = 2 * atan2(sqrt(1 - e) * sin(nu / 2), sqrt(1 + e) * cos(nu / 2))
    return E - e * sin(E)


def dD_dw(w, e, P, h=1e-5):
    """d(secondary - primary separation)/d(omega), in days per radian."""
    def sep(w):
        return P * ((mean_anom(1.5 * pi - w, e) - mean_anom(0.5 * pi - w, e)) % (2 * pi)) / (2 * pi)
    return (sep(w + h) - sep(w - h)) / (2 * h)


def tidal_per_k2(e, M1, M2, R1, R2, a, rot):
    """Tidal + rotational apsidal advance per orbit / 2pi, per unit k2 (both stars, equal k2)."""
    f = (1 + 1.5 * e**2 + 0.125 * e**4) / (1 - e**2) ** 5
    g = 1 / (1 - e**2) ** 2
    return sum((Ri / a) ** 5 * (rot**2 * (1 + mj / mi) * g + 15 * (mj / mi) * f)
               for mi, mj, Ri in ((M1, M2, R1), (M2, M1, R2)))


print(f"{'TIC':>10} {'P[d]':>7} {'e':>5} {'w':>5} {'drift[min/1000d]':>18} {'U_obs[yr]':>10} {'U_GR[yr]':>9}  k2 (median, 95%)")
for t in sorted(etv):
    if t not in par:
        print(f"{t:>10}  no fitted parameters"); continue
    p = par[t]
    P, e, w = float(p['per']), float(p['ecc']), radians(float(p['w_deg']))
    M1, M2, R1, R2, a = (float(p[k]) for k in ('M1', 'M2', 'R1', 'R2', 'a_Rsun'))
    prot = float(p['prot'])
    rot = P / prot if prot > 0 else 1.0
    conv = 1440 / P * 1000                       # days per epoch -> min per 1000 d
    S, Se = abs(float(etv[t]['sp'])) * conv, float(etv[t]['sp_err']) * conv
    dd = dD_dw(w, e, P)
    gr = 3 * GM_SUN * (M1 + M2) / (C**2 * a * R_SUN * (1 - e**2))   # advance per orbit / 2pi
    tid = tidal_per_k2(e, M1, M2, R1, R2, a, rot)
    Sd = np.abs(S + Se * rng.standard_normal(4000))
    wdot = (Sd / 1000 / 1440) / abs(dd)           # rad per day
    k2 = (wdot * P / (2 * pi) - gr) / tid
    U_obs = 2 * pi / ((S / 1000 / 1440) / abs(dd)) / 365.25
    U_gr = P / gr / 365.25
    lo, med, hi = np.percentile(k2, [2.5, 50, 97.5])
    note = ''
    if e < 0.02:
        note = '  near-circular: k2 unreliable'
    elif lo > 0.75:
        note = '  ABOVE physical maximum -> extra driver (third body?)'
    elif hi < 0:
        note = '  slower than GR alone'
    print(f"{t:>10} {P:7.3f} {e:5.3f} {np.degrees(w) % 360:5.0f} {S:9.2f} +/- {Se:5.2f} {U_obs:10.1f} {U_gr:9.0f}  "
          f"{med:.4f} [{lo:.4f}, {hi:.4f}]{note}")
