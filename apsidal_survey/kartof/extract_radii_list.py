"""On kartof: build Penev's Binary from the max-likelihood MCMC sample for each TIC in a list file,
and save masses, radii, separation and rotation to ~/binary_radii_more.csv.
Usage: ~/venv_etv/bin/python ~/extract_radii_list.py ~/need_radii.txt 2>/dev/null
"""
import sys, csv, math, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '/home/victor/code')
sys.path.insert(0, '/home/kpenev/projects/git/TESS_EB_Bayesian/scripts')
from chain_analysis import get_max_likelihood_params
from binary import Binary

TICS = [int(x) for x in open(sys.argv[1]).read().split() if x.strip()]
GM_SUN, R_SUN = 1.32712440018e20, 6.957e8
rows = []
for t in TICS:
    try:
        best = get_max_likelihood_params(f'/mnt/md2/TESS_EBs/first1000/tess{t}_samples.h5')[0]
        b = Binary(from_mcmc=best)
        P, M, q = b.per, b.mtotal, b.mratio
        a_kep = (GM_SUN * M * (P * 86400) ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / R_SUN
        R1 = b.rstar
        rows.append(dict(tic=t, per=P, ecc=b.ecc, w_deg=b.w % 360, inc=b.inc, mtotal=M, mratio=q,
                         M1=M / (1 + q), M2=M * q / (1 + q), R1=R1, R2=b.rp * R1, a_Rsun=b.a * R1,
                         a_check=b.a * R1 / a_kep, teff_ratio=b.teff_ratio, prot=b.prot,
                         inverted=b.inverted))
        r = rows[-1]
        print(f"{t:>10}  M1 {r['M1']:5.2f}  M2 {r['M2']:5.2f}  R1 {r['R1']:6.2f}  R2 {r['R2']:6.2f}  "
              f"a_check {r['a_check']:.3f}", flush=True)
    except Exception as e:
        print(f"{t:>10}  FAILED: {e!r}", flush=True)

if rows:
    with open('/home/victor/binary_radii_more.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
print(f"\nwrote {len(rows)} of {len(TICS)} systems to ~/binary_radii_more.csv")
