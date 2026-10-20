import csv, json, glob, os, re
from math import pi, sqrt, sin, cos, atan2, hypot

GM_SUN = 1.32712440018e20   # m^3 s^-2
C = 2.99792458e8            # m/s
R_SUN = 6.957e8             # m
DAY = 86400.0

def load(f):
    return {int(float(r['tic_id'])): r for r in csv.DictReader(open(f))}
med = load('tic_medians_table_add_med_flipped.csv')
trig = load('tic_etrigwquants.csv')
cols = list(next(iter(med.values())).keys())
cM = next(c for c in cols if 'M_1+M_2' in c)
cq = next(c for c in cols if 'M_2/M_1' in c)
cP = next(c for c in cols if 'P_{orb}' in c)

src = re.sub(r'#.*', '', open('group_statistics.py').read())
sets = {n: {int(x) for x in re.findall(r'\b\d{5,10}\b', b)}
        for n, b in re.findall(r'^([A-Z_]+)\s*=\s*\{(.*?)\}', src, re.S | re.M)}
NEW = {396903513, 331183881, 381857817, 427312571, 272822330, 184246521,
       89522181, 372758722, 399860444}
targets = sets.get('PRECESSION', set()) | sets.get('PRECESSION_LTTE', set()) | NEW

def summary(t):
    for p in (f'rerun_fixedP/TIC_{t}_out/TIC{t}_summary.json', f'TIC_{t}_out/TIC{t}_summary.json'):
        if os.path.exists(p):
            return json.load(open(p)), ('rerun' if 'rerun' in p else 'orig')
    return None, None

def mean_anom(nu, e):
    E = 2 * atan2(sqrt(1 - e) * sin(nu / 2), sqrt(1 + e) * cos(nu / 2))
    return E - e * sin(E)

def sep(w, e, P):
    dM = (mean_anom(1.5 * pi - w, e) - mean_anom(0.5 * pi - w, e)) % (2 * pi)
    return P * dM / (2 * pi)

def radius(m):  # rough main-sequence radius, solar units
    return m ** 0.8 if m < 1 else m ** 0.57

def tidal_P_over_U(e, m1, m2, a, k2):  # Sterne (1939), synchronous rotation
    f = (1 + 1.5 * e**2 + 0.125 * e**4) / (1 - e**2) ** 5
    g = 1 / (1 - e**2) ** 2
    tot = 0.0
    for mi, mj in ((m1, m2), (m2, m1)):
        r = radius(mi) * R_SUN / a
        tot += k2 * r**5 * ((1 + mj / mi) * g + 15 * (mj / mi) * f)
    return tot

rows = []
for t in sorted(targets):
    s, where = summary(t)
    if s is None or t not in med or t not in trig or s.get('apsidal_slope_min_per_1000d') is None:
        print(f"{t}: skipped (missing results or catalog entry)")
        continue
    P = float(med[t][cP]); M = float(med[t][cM]); q = float(med[t][cq])
    m1, m2 = M / (1 + q), M * q / (1 + q)
    ec, es = float(trig[t]['ecosw_Q0.5']), float(trig[t]['esinw_Q0.5'])
    e, w = hypot(ec, es), atan2(es, ec)
    a = (GM_SUN * M * (P * DAY) ** 2 / (4 * pi**2)) ** (1 / 3)
    S = s['apsidal_slope_min_per_1000d']
    h = 1e-5
    dDdw = (sep(w + h, e, P) - sep(w - h, e, P)) / (2 * h)       # days per radian
    wdot = (S / 1000 / 1440) / dDdw                                # rad per day, signed
    pu_imp = abs(wdot) * P / (2 * pi)
    pu_gr = 3 * GM_SUN * M / (C**2 * a * (1 - e**2))
    pu_lo = pu_gr + tidal_P_over_U(e, m1, m2, a, 0.002)
    pu_hi = pu_gr + tidal_P_over_U(e, m1, m2, a, 0.010)
    U = lambda pu: P / pu / 365.25
    flags = []
    if e < 0.02:
        flags.append('near-circular')
    if wdot < 0:
        flags.append('sign?')
    if pu_imp < 0.5 * pu_gr:
        verdict = 'SLOWER THAN GR'
    elif pu_imp > 10 * pu_hi:
        verdict = 'TOO FAST'
    else:
        verdict = 'consistent'
    rows.append(dict(tic=t, src=where, P=P, e=e, w_deg=w * 180 / pi, M=M, q=q, S=S,
                     U_implied=U(pu_imp), U_GR=U(pu_gr), U_exp_fast=U(pu_hi), U_exp_slow=U(pu_lo),
                     ratio_to_max=pu_imp / pu_hi, verdict=verdict, flags=' '.join(flags)))

print(f"\n{'TIC':>10} {'src':5} {'P[d]':>7} {'e':>6} {'M':>5} {'S':>8} {'U_impl[yr]':>11} "
      f"{'U_GR[yr]':>10} {'U_exp range[yr]':>17} {'impl/max':>8}  verdict  flags")
for r in sorted(rows, key=lambda r: r['verdict']):
    print(f"{r['tic']:>10} {r['src']:5} {r['P']:7.3f} {r['e']:6.3f} {r['M']:5.2f} {r['S']:+8.2f} "
          f"{r['U_implied']:11.0f} {r['U_GR']:10.0f} {r['U_exp_fast']:8.0f}-{r['U_exp_slow']:<8.0f} "
          f"{r['ratio_to_max']:8.2f}  {r['verdict']}  {r['flags']}")
with open('apsidal_consistency.csv', 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
print("\nwrote apsidal_consistency.csv")
