import csv, json, os
def drift(t):
    p = f'rerun_v2/TIC_{t}_out/TIC{t}_summary.json'
    if not os.path.exists(p):
        return 'n/a'
    s = json.load(open(p))
    S, Se = s.get('apsidal_slope_min_per_1000d'), s.get('apsidal_slope_err_min_per_1000d')
    return f"{S:+.2f}+/-{Se:.2f} ({s.get('verdict')})" if S is not None else 'n/a'
for r in csv.DictReader(open('candidates_v2.csv')):
    if r['co_drift'] or r['opposite']:
        print(f"{r['tic']:>10} {float(r['S']):+8.2f}  [{r['new_verdict']}]")
        for kind in ('co_drift', 'opposite'):
            for u in r[kind].split():
                print(f"      {kind:9s} {u:>10}  {drift(u)}")
