import csv
from collections import Counter
from astroquery.vizier import Vizier

cats = Vizier(row_limit=-1, columns=['**']).get_catalogs('J/ApJS/258/16')
for t in cats:
    print(f"table {t.meta.get('name')}: {len(t)} rows, cols {t.colnames[:12]}")
t = max(cats, key=len)
tic_col = next(c for c in t.colnames if 'TIC' in c.upper())
per_col = next(c for c in t.colnames if c.lower().startswith('per'))
print(f"using table {t.meta.get('name')}, TIC column '{tic_col}', period column '{per_col}'\n")

prsa = {}
for r in t:
    try:
        p = float(r[per_col])
        if p > 0:
            prsa[int(r[tic_col])] = p
    except (ValueError, TypeError):
        pass

def near(a, b):
    return abs(a / b - 1) < 0.01

rows = [r for r in csv.DictReader(open('crossmatch_penev_results.csv')) if r['P_k'] and r['P_v']]
out, tally = [], {'match': Counter(), 'mismatch': Counter()}
for r in rows:
    tic, Pv, Pk = int(r['tic']), float(r['P_v']), float(r['P_k'])
    group = 'match' if near(Pv, Pk) else 'mismatch'
    Pp = prsa.get(tic)
    if Pp is None:
        v = 'not in Prsa'
    elif near(Pp, Pk) and near(Pp, Pv):
        v = 'all agree'
    elif near(Pp, Pk):
        v = 'Penev'
    elif near(Pp, Pv):
        v = 'Victor'
    else:
        v = 'neither'
    tally[group][v] += 1
    out.append(dict(tic=tic, cls=r['cls'], P_v=Pv, P_k=Pk, P_prsa=Pp, verdict=v, group=group))

for g in ('match', 'mismatch'):
    print(f"{g}: {dict(tally[g])}")

print("\nmismatches:")
for o in sorted(out, key=lambda o: (o['group'] != 'mismatch', o['verdict'], o['cls'])):
    if o['group'] != 'mismatch':
        continue
    pp = f"{o['P_prsa']:.5f}" if o['P_prsa'] else '   --  '
    print(f"  {o['tic']:>10} {o['cls']:16s} P_v {o['P_v']:9.5f}  P_k {o['P_k']:9.5f}  "
          f"P_prsa {pp:>9}  -> {o['verdict']}")

with open('period_arbitration.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print("\nwrote period_arbitration.csv")
