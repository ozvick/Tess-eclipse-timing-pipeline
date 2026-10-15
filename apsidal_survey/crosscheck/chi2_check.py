import csv, statistics as st
for f in ('penev_etv_drifts.csv', 'penev_etv_drifts_sqrt2.csv'):
    rows = [r for r in csv.DictReader(open(f)) if r['variant'] == 'none']
    cp = [float(r['chi2_p']) for r in rows if r['chi2_p']]
    cs = [float(r['chi2_s']) for r in rows if r['chi2_s']]
    print(f"{f}: median reduced chi2  primary {st.median(cp):.2f} (n={len(cp)})   "
          f"secondary {st.median(cs):.2f} (n={len(cs)})")
