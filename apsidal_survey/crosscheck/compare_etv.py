import csv, json, glob, re
from math import hypot

def conv(x, P):
    return x * 1440 / P * 1000

pen = {}
for r in csv.DictReader(open('penev_etv_drifts.csv')):
    if r['sp']:
        pen.setdefault(int(r['tic']), {})[r['variant']] = (float(r['sp']), float(r['sp_err']))

arb = {int(r['tic']): r for r in csv.DictReader(open('period_arbitration.csv'))}
src = re.sub(r'#.*', '', open('group_statistics.py').read())
sets = {n: {int(x) for x in re.findall(r'\b\d{5,10}\b', b)}
        for n, b in re.findall(r'^([A-Z_]+)\s*=\s*\{(.*?)\}', src, re.S | re.M)}
def cls(t):
    for n in ('PRECESSION_LTTE', 'PRECESSION', 'TENTATIVE', 'DEMOTED', 'NULL_CONSTRAINED'):
        if t in sets.get(n, ()):
            return n
    return 'other'

vic = {}
for p in glob.glob('TIC_*_out/TIC*_summary.json'):
    if '_out2/' in p or '_early_out/' in p:
        continue
    s = json.load(open(p))
    t = int(s['tic'])
    if t in sets.get('DUPLICATE', ()):
        continue
    vic[t] = (cls(t), s)
for p in glob.glob('rerun_fixedP/TIC_*_out/TIC*_summary.json'):
    s = json.load(open(p))
    vic[int(s['tic'])] = ('rerun:' + str(s.get('verdict')), s)

rows, nosec = [], []
for t, (c, s) in vic.items():
    if t not in pen or 'none' not in pen[t]:
        continue
    P = float(arb[t]['P_prsa']) if t in arb and arb[t]['P_prsa'] else s['period_days']
    sp, spe = pen[t]['none']
    pv, pve = -conv(sp, P), conv(spe, P)
    spl = -conv(pen[t]['spline'][0], P) if 'spline' in pen[t] else None
    S, Se = s.get('apsidal_slope_min_per_1000d'), s.get('apsidal_slope_err_min_per_1000d')
    if S is None or not Se:
        nosec.append((t, c, pv, pve))
        continue
    z = (S - pv) / hypot(Se, pve)
    rows.append(dict(tic=t, cls=c, S=S, Se=Se, pen=pv, pen_err=pve, z=z,
                     spl_diff=(spl - pv) if spl is not None else None))

print(f"systems in both with a Victor slope: {len(rows)};  Penev drift but no Victor secondary: {len(nosec)}")
b = [sum(1 for r in rows if lo <= abs(r['z']) < hi) for lo, hi in ((0, 2), (2, 3), (3, 1e9))]
print(f"agreement |z|<2: {b[0]}   2-3: {b[1]}   >3: {b[2]}")
both = [r for r in rows if abs(r['S'] / r['Se']) > 3 and abs(r['pen'] / r['pen_err']) > 3]
same = sum(1 for r in both if r['S'] * r['pen'] > 0)
print(f"both significant (>3 sigma): {len(both)};  same sign (after flipping Penev): {same}")

print(f"\n{'TIC':>10} {'class':22s} {'Victor':>16} {'Penev (-1x)':>16} {'z':>6} {'spl-none':>8}")
for r in sorted(rows, key=lambda r: (r['cls'], r['tic'])):
    if abs(r['S'] / r['Se']) > 3 or abs(r['pen'] / r['pen_err']) > 3:
        sd = f"{r['spl_diff']:+8.2f}" if r['spl_diff'] is not None else '     n/a'
        print(f"{r['tic']:>10} {r['cls']:22s} {r['S']:+8.2f}+/-{r['Se']:5.2f} "
              f"{r['pen']:+8.2f}+/-{r['pen_err']:5.2f} {r['z']:+6.1f} {sd}")

print("\nPenev drift >3 sigma where Victor has no secondary:")
for t, c, pv, pve in sorted(nosec):
    if abs(pv / pve) > 3:
        print(f"  {t:>10} {c:22s} {pv:+8.2f}+/-{pve:5.2f}")

with open('etv_comparison.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['tic', 'cls', 'S', 'Se', 'pen', 'pen_err', 'z', 'spl_diff'])
    w.writeheader()
    w.writerows(rows)
print("\nwrote etv_comparison.csv")
