import csv, glob, json, re
from math import atan2, sin, cos, sqrt, pi, hypot
from collections import Counter

def load(fname):
    with open(fname) as f:
        return {int(float(r['tic_id'])): r for r in csv.DictReader(f)}

med = load('tic_medians_table_add_med_flipped.csv')
trig = load('tic_etrigwquants.csv')
trigf = load('tic_etrigwquants_flipped.csv')
pcol = next(c for c in next(iter(med.values())) if 'P_{orb}' in c)

src = re.sub(r'#.*', '', open('group_statistics.py').read())
sets = {n: {int(x) for x in re.findall(r'\b\d{5,10}\b', b)}
        for n, b in re.findall(r'^([A-Z_]+)\s*=\s*\{(.*?)\}', src, re.S | re.M)}
def cls(t):
    for n in ('PRECESSION_LTTE', 'PRECESSION', 'TENTATIVE', 'DEMOTED', 'NULL_CONSTRAINED'):
        if t in sets.get(n, ()):
            return n
    return 'other'

def pcat(x):
    if x is None:
        return 'n/a'
    for name, v in (('1', 1), ('2', 2), ('1/2', 0.5), ('3', 3), ('1/3', 1/3)):
        if abs(x / v - 1) < 0.01:
            return name
    return 'other'

def mean_anom(nu, e):
    E = 2 * atan2(sqrt(1 - e) * sin(nu / 2), sqrt(1 + e) * cos(nu / 2))
    return E - e * sin(E)

def sep(w, e, P):  # secondary minus primary, days, i = 90 deg
    dM = (mean_anom(1.5 * pi - w, e) - mean_anom(0.5 * pi - w, e)) % (2 * pi)
    return P * dM / (2 * pi)

rows = []
for path in sorted(glob.glob('TIC_*_out/TIC*_summary.json')):
    if '_out2/' in path or '_early_out/' in path:
        continue
    s = json.load(open(path))
    t = int(s['tic'])
    if t in sets.get('DUPLICATE', ()):
        continue
    r = dict(tic=t, cls=cls(t), P_v=s.get('period_days'), ecosw_v=s.get('implied_ecosw'),
             S=s.get('apsidal_slope_min_per_1000d'), S_err=s.get('apsidal_slope_err_min_per_1000d'))
    if t in med and t in trig:
        q = trig[t]
        P_k = float(med[t][pcol])
        ec, es = float(q['ecosw_Q0.5']), float(q['esinw_Q0.5'])
        r.update(P_k=P_k, ratio=(r['P_v'] / P_k) if r['P_v'] else None, ecosw_k=ec, esinw_k=es,
                 esinw_err=(float(q['esinw_Q0.841']) - float(q['esinw_Q0.159'])) / 2)
        if t in trigf:
            r['flip_diff'] = abs(float(trigf[t]['ecosw_Q0.5']) - ec) + abs(float(trigf[t]['esinw_Q0.5']) - es)
        if r['S'] and es:
            rel = hypot(r['esinw_err'] / abs(es), (r['S_err'] or 0) / abs(r['S']))
            U1 = 4 * P_k * abs(es) / (abs(r['S']) / 1000 / 1440) / 365.25 / 1440 * 1440
            U1 = 4 * P_k * 1440 * abs(es) / (abs(r['S']) / 1000) / 365.25
            e, w, h = hypot(ec, es), atan2(es, ec), 1e-5
            dDdw = abs(sep(w + h, e, P_k) - sep(w - h, e, P_k)) / (2 * h)
            wdot = (abs(r['S']) / 1000 / 1440) / dDdw
            Ux = 2 * pi / wdot / 365.25
            r.update(e_k=e, U_first=U1, U_exact=Ux, U_err=Ux * rel)
    rows.append(r)

inK = [r for r in rows if 'P_k' in r]
print(f"analyzed: {len(rows)}   in Penev tables: {len(inK)}")
print("period ratio (yours / Penev):", dict(Counter(pcat(r['ratio']) for r in inK)))
print("\nperiod mismatches:")
for r in inK:
    if pcat(r['ratio']) != '1':
        print(f"  {r['tic']:>10}  {r['cls']:16s} P_v {r['P_v']:.5f}  P_k {r['P_k']:.5f}  ratio {r['ratio']:.4f}")

same = [r for r in inK if pcat(r['ratio']) == '1' and r['ecosw_v'] is not None]
bad = [r for r in same if abs(abs(r['ecosw_v']) - abs(r['ecosw_k'])) > 0.02]
print(f"\nperiods agree and e.cosw available: {len(same)};  |e.cosw| differs by >0.02: {len(bad)}")
for r in bad:
    print(f"  {r['tic']:>10}  {r['cls']:16s} yours {r['ecosw_v']:+.4f}  Penev {r['ecosw_k']:+.4f}")

print("\nDETECTIONS")
print(f"{'TIC':>10} {'class':15s} {'ratio':>5} {'ecosw_v':>8} {'ecosw_k':>8} {'esinw_k':>8} {'e_k':>6} "
      f"{'S':>8} {'U1st':>8} {'Uexact':>8} {'err':>6} {'flip':>7}")
for r in sorted(rows, key=lambda r: r['tic']):
    if r['cls'] not in ('PRECESSION', 'PRECESSION_LTTE'):
        continue
    if 'P_k' not in r:
        print(f"{r['tic']:>10} {r['cls']:15s}  -- not in Penev tables")
        continue
    ev = f"{r['ecosw_v']:+.3f}" if r['ecosw_v'] is not None else 'n/a'
    U1 = f"{r['U_first']:.0f}" if 'U_first' in r else 'n/a'
    Ux = f"{r['U_exact']:.0f}" if 'U_exact' in r else 'n/a'
    Ue = f"{r['U_err']:.0f}" if 'U_err' in r else ''
    print(f"{r['tic']:>10} {r['cls']:15s} {pcat(r['ratio']):>5} {ev:>8} {r['ecosw_k']:+8.3f} "
          f"{r['esinw_k']:+8.3f} {r['e_k']:6.3f} {r['S']:+8.2f} {U1:>8} {Ux:>8} {Ue:>6} "
          f"{r.get('flip_diff', 0):7.1e}")

keys = ['tic', 'cls', 'P_v', 'P_k', 'ratio', 'ecosw_v', 'ecosw_k', 'esinw_k', 'esinw_err',
        'e_k', 'S', 'S_err', 'U_first', 'U_exact', 'U_err', 'flip_diff']
with open('crossmatch_penev_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
print("\nwrote crossmatch_penev_results.csv")
