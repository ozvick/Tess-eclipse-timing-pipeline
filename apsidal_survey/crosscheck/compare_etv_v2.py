"""Compare Penev's (new) ETV drifts with the patched pipeline (rerun_v2/).

Usage: python3 compare_etv_v2.py [penev_etv_drifts_v2.csv]
- converts his drifts (days per epoch) to min/1000 d using each system's period,
- detects the sign convention from the data instead of assuming it,
- reports overall agreement and the status of every candidate group,
- writes etv_comparison_v2.csv.
"""
import csv, json, os, sys
from math import hypot

ERR_SCALE = 1.2
fname = sys.argv[1] if len(sys.argv) > 1 else 'penev_etv_drifts_v2.csv'

CONFIRMED = [197755658, 12915124, 220430912, 89522181, 396903513, 160328766, 161988269, 319936228,
             331183881, 289044378, 71531463, 184246521, 341029467, 55497281, 455824124, 148914403,
             356015357]
PENDING = [272822330, 424719589, 272532173]
REVIEW = [381857817, 358232450, 220420534, 427312571]
RETIRED = [140659980, 140659978]

# systems where Penev's period is not the one the pipeline used (his epochs would not match)
skip = set()
if os.path.exists('period_arbitration.csv'):
    skip = {int(r['tic']) for r in csv.DictReader(open('period_arbitration.csv')) if r['verdict'] == 'Victor'}

pen = {int(r['tic']): r for r in csv.DictReader(open(fname)) if r.get('sp')}
rows = []
for t, r in pen.items():
    p = f'rerun_v2/TIC_{t}_out/TIC{t}_summary.json'
    if t in skip or not os.path.exists(p):
        continue
    s = json.load(open(p))
    S, Se = s.get('apsidal_slope_min_per_1000d'), s.get('apsidal_slope_err_min_per_1000d')
    if S is None or not Se:
        continue
    conv = 1440 / s['period_days'] * 1000
    rows.append(dict(tic=t, S=S, Se=Se * ERR_SCALE, raw=float(r['sp']) * conv,
                     raw_err=float(r['sp_err']) * conv, verdict=s.get('verdict')))

both = [r for r in rows if abs(r['S'] / r['Se']) > 3 and abs(r['raw'] / r['raw_err']) > 3]
same = sum(1 for r in both if r['S'] * r['raw'] > 0)
print(f"systems compared: {len(rows)} (skipped {len(skip)} with a different period)")
if len(both) < 3:
    flip = -1
    print(f"sign check: only {len(both)} significant pairs -- undetermined; keeping the "
          f"established convention (his values are C-O, flipping)")
else:
    flip = -1 if same < len(both) / 2 else 1
    print(f"sign check: {same} of {len(both)} significant pairs have the same sign as stored "
          f"-> {'his values are C-O, flipping' if flip < 0 else 'his values are O-C, no flip'}")

for r in rows:
    r['pen'] = flip * r['raw']
    r['pen_err'] = r['raw_err']
    r['z'] = (r['S'] - r['pen']) / hypot(r['Se'], r['pen_err'])

within = lambda k: sum(1 for r in rows if abs(r['z']) < k)
print(f"agreement: within 2 sigma {within(2)}/{len(rows)}, within 3 sigma {within(3)}/{len(rows)}")

by = {r['tic']: r for r in rows}
def show(label, tics):
    print(f"\n{label}")
    for t in tics:
        r = by.get(t)
        if r is None:
            print(f"  {t:>10}  not in his ETV results")
        else:
            print(f"  {t:>10}  yours {r['S']:+8.2f}+/-{r['Se']:5.2f}   his {r['pen']:+8.2f}+/-{r['pen_err']:5.2f}"
                  f"   z {r['z']:+5.1f}  {'AGREE' if abs(r['z']) < 3 else 'DISAGREE'}")

show("PENDING (his check decides these):", PENDING)
show("UNDER REVIEW:", REVIEW)
show("CONFIRMED (should stay consistent):", CONFIRMED)
show("RETIRED PAIR:", RETIRED)

bad = sorted((r for r in rows if abs(r['z']) > 3), key=lambda r: -abs(r['z']))
print(f"\nall systems disagreeing by more than 3 sigma ({len(bad)}):")
for r in bad[:30]:
    print(f"  {r['tic']:>10}  yours {r['S']:+8.2f}  his {r['pen']:+8.2f}  z {r['z']:+6.1f}  [{r['verdict']}]")

with open('etv_comparison_v2.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['tic', 'S', 'Se', 'pen', 'pen_err', 'z', 'verdict'], extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
print("\nwrote etv_comparison_v2.csv")
