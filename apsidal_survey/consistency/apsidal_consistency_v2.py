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

def radius(m):
    return m ** 0.8 if m < 1 else m ** 0.57

def tidal_pu(e, m1, m2, a, k2):
    f = (1 + 1.5 * e**2 + 0.125 * e**4) / (1 - e**2) ** 5
    g = 1 / (1 - e**2) ** 2
    return sum(k2 * (radius(mi) * R_SUN / a) ** 5 * ((1 + mj / mi) * g + 15 * (mj / mi) * f)
               for mi, mj in ((m1, m2), (m2, m1)))

rows = []
for t in sorted(targets):
    s = summary(t)
    if s is None or t not in med or t not in trig or s.get('apsidal_slope_min_per_1000d') is None:
        continue
    P, M, q = float(med[t][cP]), float(med[t][cM]), float(med[t][cq])
    m1, m2 = M / (1 + q), M * q / (1 + q)
    a = (GM_SUN * M * (P * DAY) ** 2 / (4 * pi**2)) ** (1 / 3)
    Q = {k: float(v) for k, v in trig[t].items() if k.startswith('e')}
    ec, es = Q['ecosw_Q0.5'], Q['esinw_Q0.5']
    sec = (Q['ecosw_Q0.841'] - Q['ecosw_Q0.159']) / 2
    ses = (Q['esinw_Q0.841'] - Q['esinw_Q0.159']) / 2
    S, Se = s['apsidal_slope_min_per_1000d'], s.get('apsidal_slope_err_min_per_1000d') or 0
    e0 = hypot(ec, es)
    pu_gr = 3 * GM_SUN * M / (C**2 * a * (1 - e0**2))
    pu_hi = pu_gr + tidal_pu(e0, m1, m2, a, 0.010)

    Sd = S + Se * ERR_SCALE * rng.standard_normal(N)
    ecd = ec + sec * rng.standard_normal(N)
    esd = es + ses * rng.standard_normal(N)
    wdot = np.array([(Sd[i] / 1000 / 1440) / dD_dw(atan2(esd[i], ecd[i]), min(hypot(ecd[i], esd[i]), 0.99), P)
                     for i in range(N)])
    pu = np.abs(wdot) * P / (2 * pi)
    lo, med_pu, hi = np.percentile(pu, [2.5, 50, 97.5])
    p_neg = float(np.mean(wdot < 0))

    if hi < pu_gr:
        verdict = 'SLOWER THAN GR'
    elif lo > 10 * pu_hi:
        verdict = 'TOO FAST'
    elif lo > pu_hi:
        verdict = 'fast'
    else:
        verdict = 'consistent'

    v_ec = s.get('implied_ecosw')
    swap = ''
    if v_ec is not None and abs(ec) > 3 * sec and abs(v_ec) > 0.01 and v_ec * ec < 0:
        swap = 'ecosw signs differ: SWAP likely'
    elif v_ec is not None and (abs(ec) <= 3 * sec or abs(v_ec) <= 0.01):
        swap = 'ecosw ~0: inconclusive'
    pd, sd = s.get('primary_depth'), s.get('secondary_depth')
    ddep = abs(pd - sd) / pd * 100 if pd and sd else float('nan')
    U = lambda x: P / x / 365.25
    rows.append(dict(tic=t, e=e0, S=S, Se=Se, U_med=U(med_pu), U_fast=U(hi), U_slow=U(lo),
                     U_GR=U(pu_gr), ratio_lo=lo / pu_hi, ratio_med=med_pu / pu_hi, ratio_hi=hi / pu_hi,
                     p_neg=p_neg, v_ecosw=v_ec, p_ecosw=ec, depth_diff_pct=ddep,
                     verdict=verdict, swap=swap))

print(f"{'TIC':>10} {'e':>6} {'S':>13} {'U_impl [95%] yr':>24} {'U_GR':>7} "
      f"{'impl/max [95%]':>20} {'P(neg)':>6}  verdict")
for r in sorted(rows, key=lambda r: (r['verdict'], -r['ratio_med'])):
    print(f"{r['tic']:>10} {r['e']:6.3f} {r['S']:+7.2f}+/-{r['Se']:4.2f} "
          f"{r['U_med']:8.0f} [{r['U_fast']:6.0f}-{r['U_slow']:<7.0f}] {r['U_GR']:7.0f} "
          f"{r['ratio_med']:7.2f} [{r['ratio_lo']:.2f}-{r['ratio_hi']:.2f}] {r['p_neg']:6.2f}  {r['verdict']}")

print("\nSign check (systems with P(neg) > 0.5):")
print(f"{'TIC':>10} {'P(neg)':>6} {'Victor ecosw':>12} {'Penev ecosw':>11} {'depth diff %':>12}  diagnosis")
for r in sorted(rows, key=lambda r: -r['p_neg']):
    if r['p_neg'] > 0.5:
        ve = f"{r['v_ecosw']:+.4f}" if r['v_ecosw'] is not None else 'n/a'
        print(f"{r['tic']:>10} {r['p_neg']:6.2f} {ve:>12} {r['p_ecosw']:+11.4f} {r['depth_diff_pct']:12.1f}  {r['swap']}")

with open('apsidal_consistency_v2.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nwrote apsidal_consistency_v2.csv")
