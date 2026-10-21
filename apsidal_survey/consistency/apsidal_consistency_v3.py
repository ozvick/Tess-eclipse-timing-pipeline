import csv, json, os, re
import numpy as np
from math import pi, sqrt, sin, cos, atan2, hypot

GM_SUN, C, R_SUN, DAY = 1.32712440018e20, 2.99792458e8, 6.957e8, 86400.0
ERR_SCALE, N = 1.2, 2000
rng = np.random.default_rng(1)

def load(f):
    return {int(float(r['tic_id'])): r for r in csv.DictReader(open(f))}
med, trig = load('tic_medians_table_add_med_flipped.csv'), load('tic_etrigwquants.csv')
cols = list(next(iter(med.values())).keys())
cM = next(c for c in cols if 'M_1+M_2' in c)
cq = next(c for c in cols if 'M_2/M_1' in c)
cP = next(c for c in cols if 'P_{orb}' in c)
radii = {int(r['tic']): r for r in csv.DictReader(open('binary_radii.csv'))}

src = re.sub(r'#.*', '', open('group_statistics.py').read())
sets = {n: {int(x) for x in re.findall(r'\b\d{5,10}\b', b)}
        for n, b in re.findall(r'^([A-Z_]+)\s*=\s*\{(.*?)\}', src, re.S | re.M)}
NEW = {396903513, 331183881, 381857817, 427312571, 272822330, 184246521, 89522181, 372758722}
targets = sets.get('PRECESSION', set()) | sets.get('PRECESSION_LTTE', set()) | NEW

def summary(t):
    for p in (f'rerun_fixedP/TIC_{t}_out/TIC{t}_summary.json', f'TIC_{t}_out/TIC{t}_summary.json'):
        if os.path.exists(p):
            return json.load(open(p))
    return None

def mean_anom(nu, e):
    E = 2 * atan2(sqrt(1 - e) * sin(nu / 2), sqrt(1 + e) * cos(nu / 2))
    return E - e * sin(E)

def dD_dw(w, e, P, h=1e-5):
    def sep(w):
        return P * ((mean_anom(1.5 * pi - w, e) - mean_anom(0.5 * pi - w, e)) % (2 * pi)) / (2 * pi)
    return (sep(w + h) - sep(w - h)) / (2 * h)

def radius_ms(m):
    return m ** 0.8 if m < 1 else m ** 0.57

def tidal_pu(e, M1, M2, R1, R2, a, k2, rot):
    f = (1 + 1.5 * e**2 + 0.125 * e**4) / (1 - e**2) ** 5
    g = 1 / (1 - e**2) ** 2
    return sum(k2 * (Ri / a) ** 5 * (rot**2 * (1 + mj / mi) * g + 15 * (mj / mi) * f)
               for mi, mj, Ri in ((M1, M2, R1), (M2, M1, R2)))

rows = []
for t in sorted(targets):
    s = summary(t)
    if s is None or t not in med or t not in trig or s.get('apsidal_slope_min_per_1000d') is None:
        continue
    P, M, q = float(med[t][cP]), float(med[t][cM]), float(med[t][cq])
    if t in radii:
        r = radii[t]
        M1, M2, R1, R2 = (float(r[k]) for k in ('M1', 'M2', 'R1', 'R2'))
        a_R = float(r['a_Rsun'])
        prot = float(r['prot'])
        rot = P / prot if prot > 0 else 1.0
        rsrc = 'Penev'
    else:
        M1, M2 = M / (1 + q), M * q / (1 + q)
        R1, R2 = radius_ms(M1), radius_ms(M2)
        a_R = (GM_SUN * M * (P * DAY) ** 2 / (4 * pi**2)) ** (1 / 3) / R_SUN
        rot, rsrc = 1.0, 'MS'
    a_m = a_R * R_SUN
    Q = {k: float(v) for k, v in trig[t].items() if k.startswith('e')}
    ec, es = Q['ecosw_Q0.5'], Q['esinw_Q0.5']
    sec = (Q['ecosw_Q0.841'] - Q['ecosw_Q0.159']) / 2
    ses = (Q['esinw_Q0.841'] - Q['esinw_Q0.159']) / 2
    S, Se = s['apsidal_slope_min_per_1000d'], s.get('apsidal_slope_err_min_per_1000d') or 0
    e0 = hypot(ec, es)
    pu_gr = 3 * GM_SUN * M / (C**2 * a_m * (1 - e0**2))
    tid1 = tidal_pu(e0, M1, M2, R1, R2, a_R, 1.0, rot)
    pu_lo, pu_hi = pu_gr + 0.002 * tid1, pu_gr + 0.010 * tid1

    Sd = S + Se * ERR_SCALE * rng.standard_normal(N)
    ecd, esd = ec + sec * rng.standard_normal(N), es + ses * rng.standard_normal(N)
    wdot = np.array([(Sd[i] / 1000 / 1440) / dD_dw(atan2(esd[i], ecd[i]), min(hypot(ecd[i], esd[i]), 0.99), P)
                     for i in range(N)])
    pu = np.abs(wdot) * P / (2 * pi)
    lo, mid, hi = np.percentile(pu, [2.5, 50, 97.5])
    k2 = (pu - pu_gr) / tid1
    k2lo, k2mid, k2hi = np.percentile(k2, [2.5, 50, 97.5])

    if hi < pu_gr:
        verdict = 'SLOWER THAN GR'
    elif lo > 10 * pu_hi:
        verdict = 'TOO FAST'
    elif lo > pu_hi:
        verdict = 'fast (k2 > 0.01)'
    elif hi < pu_lo:
        verdict = 'slow (k2 < 0.002)'
    else:
        verdict = 'consistent'
    U = lambda x: P / x / 365.25
    rows.append(dict(tic=t, radii=rsrc, e=e0, R1=R1, R2=R2, a_Rsun=a_R, U_impl=U(mid),
                     U_GR=U(pu_gr), U_exp_fast=U(pu_hi), U_exp_slow=U(pu_lo),
                     ratio_med=mid / pu_hi, k2_med=k2mid, k2_lo=k2lo, k2_hi=k2hi,
                     p_neg=float(np.mean(wdot < 0)), verdict=verdict))

print(f"{'TIC':>10} {'radii':5} {'e':>6} {'R1':>6} {'R2':>6} {'U_impl':>8} {'U_GR':>8} "
      f"{'U_exp range':>15} {'impl/max':>8} {'implied k2 [95%]':>26}  verdict")
for r in sorted(rows, key=lambda r: (r['verdict'], -r['ratio_med'])):
    print(f"{r['tic']:>10} {r['radii']:5} {r['e']:6.3f} {r['R1']:6.2f} {r['R2']:6.2f} "
          f"{r['U_impl']:8.0f} {r['U_GR']:8.0f} {r['U_exp_fast']:7.0f}-{r['U_exp_slow']:<7.0f} "
          f"{r['ratio_med']:8.2f}  {r['k2_med']:8.4f} [{r['k2_lo']:8.4f},{r['k2_hi']:8.4f}]  {r['verdict']}")

with open('apsidal_consistency_v3.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nwrote apsidal_consistency_v3.csv")
