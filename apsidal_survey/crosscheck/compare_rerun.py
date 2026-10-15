import csv, json, glob, os
from collections import Counter

arb = {int(r['tic']): r for r in csv.DictReader(open('period_arbitration.csv'))}

def f(x, spec):
    return 'n/a' if x is None else format(x, spec)

trans, pbad = Counter(), []
print(f"{'TIC':>10} {'class':16s} {'P_new/P_prsa':>12} {'sec old>new':>11} "
      f"{'old slope':>9} {'new slope':>9} {'err':>6} {'ecosw':>7}  verdict old > new")
for path in sorted(glob.glob('rerun_fixedP/TIC_*_out/TIC*_summary.json')):
    new = json.load(open(path))
    t = int(new['tic'])
    oldp = f'TIC_{t}_out/TIC{t}_summary.json'
    old = json.load(open(oldp)) if os.path.exists(oldp) else {}
    ratio = new['period_days'] / float(arb[t]['P_prsa'])
    if abs(ratio - 1) > 0.01:
        pbad.append(t)
    so = 'Y' if old.get('has_secondary') else 'N'
    sn = 'Y' if new.get('has_secondary') else 'N'
    trans[(old.get('verdict'), new.get('verdict'))] += 1
    print(f"{t:>10} {arb[t]['cls']:16s} {ratio:12.4f} {so + '>' + sn:>11} "
          f"{f(old.get('apsidal_slope_min_per_1000d'), '+9.2f'):>9} "
          f"{f(new.get('apsidal_slope_min_per_1000d'), '+9.2f'):>9} "
          f"{f(new.get('apsidal_slope_err_min_per_1000d'), '6.2f'):>6} "
          f"{f(new.get('implied_ecosw'), '+7.3f'):>7}  {old.get('verdict')} > {new.get('verdict')}")

print(f"\nperiod not matching input (possible alias doubling): {pbad or 'none'}")
print("verdict transitions (old -> new):")
for (a, b), n in trans.most_common():
    print(f"  {a} -> {b}: {n}")
