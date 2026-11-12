"""Observed vs theoretical k2 WITH uncertainty bands.

Monte Carlo per system: draws each star's radius and mass (default 5% each, from Penev's fits),
the metallicity (Claret 2023 grids Z = 0.004237 / 0.0134 / 0.04237 with weights 0.25/0.5/0.25), and the
observed-k2 measurement error (from its 95% range). For every draw BOTH sides are recomputed: the
theoretical k2 (track at the drawn mass, entered at the drawn radius) and the observed k2, which scales
with the stars' (R/a)^5 through the tidal term (k2_obs' = k2_obs * T(R)/T(R')). A radius error therefore
moves the two sides together, as it should.

Usage: python3 theory_k2_bands.py [--sigR 0.05] [--sigM 0.05] [--n 400]
Needs theory_k2.py (its track handling), consistency_modelfree.csv, binary_radii*.csv.
Downloads the two non-solar grids once (cached as claret2023_Z*.csv). Writes theory_k2_bands.csv/.png.
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theory_k2 as T

ap = argparse.ArgumentParser()
ap.add_argument('--sigR', type=float, default=0.05, help='fractional radius uncertainty (1 sigma)')
ap.add_argument('--sigM', type=float, default=0.05, help='fractional mass uncertainty (1 sigma)')
ap.add_argument('--n', type=int, default=400, help='Monte Carlo draws per system')
args = ap.parse_args()
rng = np.random.default_rng(3)

GRIDS = {0.004237: (24, 'claret2023_Z0.004237.csv'), 0.0134: (1, T.CACHE), 0.04237: (47, 'claret2023_Z0.04237.csv')}
ZW = {0.004237: 0.25, 0.0134: 0.5, 0.04237: 0.25}


def load_grid(z):
    first, cache = GRIDS[z]
    if not os.path.exists(cache):
        from astroquery.vizier import Vizier
        v = Vizier(columns=['**'], row_limit=-1, timeout=300); v.VIZIER_SERVER = 'vizier.cfa.harvard.edu'
        print(f'downloading Z = {z} tracks...')
        with open(cache, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['grid_mass', 'age', 'mass', 'logg', 'logk2'])
            for i, m in enumerate(T.MASSES):
                t = v.get_catalogs(f'J/A+A/674/A67/table{first + i}')[0]
                for r in t:
                    w.writerow([m, float(r['Age']), float(r['Mass']), float(r['log(g)']), float(r['log(k2)'])])
                print(f'  table{first + i} (M = {m})', flush=True)
    tr = {}
    for r in csv.DictReader(open(cache)):
        tr.setdefault(float(r['grid_mass']), []).append((float(r['age']), float(r['mass']), float(r['logg']), float(r['logk2'])))
    # precompute each track's main-sequence segment once (speed)
    return {m: T.ms_segment(np.array(sorted(v))) for m, v in tr.items()}


SEG = {z: load_grid(z) for z in GRIDS}


def logk2_star(z, M, R):
    seg = SEG[z]; grid = np.array(sorted(seg))
    if M < grid[0] or M > grid[-1]:
        return np.nan
    hi = int(np.searchsorted(grid, M)); lo = max(hi - 1, 0)
    if grid[hi] == M:
        lo = hi

    def on(m):
        logR, logk2, _ = seg[m]
        lr = np.log10(R)
        if lr <= logR[0]:
            return logk2[0]
        above = np.where(logR >= lr)[0]
        if not len(above):
            return logk2[-1]
        j = above[0]
        f = (lr - logR[j - 1]) / (logR[j] - logR[j - 1]) if logR[j] != logR[j - 1] else 0.0
        return logk2[j - 1] + f * (logk2[j] - logk2[j - 1])
    a = on(grid[lo])
    if lo == hi:
        return a
    f = (np.log10(M) - np.log10(grid[lo])) / (np.log10(grid[hi]) - np.log10(grid[lo]))
    return a + f * (on(grid[hi]) - a)


def weights(M1, M2, R1, R2, a, e, rot):
    fe = (1 + 1.5 * e ** 2 + 0.125 * e ** 4) / (1 - e ** 2) ** 5; ge = 1 / (1 - e ** 2) ** 2
    c1 = (R1 / a) ** 5 * (15 * fe * M2 / M1 + rot ** 2 * ge * (1 + M2 / M1))
    c2 = (R2 / a) ** 5 * (15 * fe * M1 / M2 + rot ** 2 * ge * (1 + M1 / M2))
    return c1, c2


par = {}
for f in ('binary_radii.csv', 'binary_radii_more.csv', 'binary_radii_paul.csv', 'binary_radii_ecc.csv'):
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            par[int(float(r['tic']))] = r

rows = []
print(f"radius sigma {args.sigR:.0%}, mass sigma {args.sigM:.0%}, {args.n} draws per system\n")
print(f"{'TIC':>10} | {'log(obs/theory) [16-84%]':>26} | {'Z=0.004':>7} {'Z=0.013':>7} {'Z=0.042':>7} | {'sig':>5} | note")
for o in csv.DictReader(open('consistency_modelfree.csv')):
    t = int(o['tic'])
    if t not in par:
        continue
    k_ob, lo95, hi95 = float(o['k2_mf']), float(o['k2_mf_lo']), float(o['k2_mf_hi'])
    if not (k_ob > 0 and lo95 > 0) or hi95 / lo95 > 10 or 'near-circular' in o['notes']:
        continue                                   # same well-constrained selection as theory_k2.py
    p = par[t]
    M1, M2, R1, R2, a = (float(p[k]) for k in ('M1', 'M2', 'R1', 'R2', 'a_Rsun'))
    P, e = float(o['P']), float(o['e_mf'])
    prot = float(p.get('prot') or 0); rot = P / prot if prot > 0 else 1.0
    if min(M1, M2) < 0.2 or max(M1, M2) > 8:
        continue
    c1_0, c2_0 = weights(M1, M2, R1, R2, a, e, rot)
    sig_obs = (np.log10(hi95) - np.log10(lo95)) / 3.92
    per_z = {}
    for z in GRIDS:
        l1, l2 = logk2_star(z, M1, R1), logk2_star(z, M2, R2)
        per_z[z] = np.log10(k_ob / ((c1_0 * 10 ** l1 + c2_0 * 10 ** l2) / (c1_0 + c2_0)))
    zs = rng.choice(list(GRIDS), size=args.n, p=[ZW[z] for z in GRIDS])
    d = []
    for z in zs:
        m1, m2 = M1 * np.exp(args.sigM * rng.standard_normal()), M2 * np.exp(args.sigM * rng.standard_normal())
        r1, r2 = R1 * np.exp(args.sigR * rng.standard_normal()), R2 * np.exp(args.sigR * rng.standard_normal())
        aa = a * ((m1 + m2) / (M1 + M2)) ** (1 / 3)
        c1, c2 = weights(m1, m2, r1, r2, aa, e, rot)
        l1, l2 = logk2_star(z, m1, r1), logk2_star(z, m2, r2)
        if not (np.isfinite(l1) and np.isfinite(l2)):
            continue
        k_th = (c1 * 10 ** l1 + c2 * 10 ** l2) / (c1 + c2)
        k_obs = k_ob * (c1_0 + c2_0) / (c1 + c2) * 10 ** (sig_obs * rng.standard_normal())
        d.append(np.log10(k_obs / k_th))
    d = np.array(d)
    q16, q50, q84 = np.percentile(d, [16, 50, 84])
    sig = q50 / ((q84 - q16) / 2)
    rows.append(dict(tic=t, P=P, M1=M1, M2=M2, R1=R1, R2=R2, e=e, dex_median=q50, dex_16=q16, dex_84=q84,
                     dex_Zlow=per_z[0.004237], dex_Zsolar=per_z[0.0134], dex_Zhigh=per_z[0.04237],
                     significance=sig, notes=o['notes']))
    flag = 'edge-on e sin w' if 'edge-on' in o['notes'] else ''
    print(f"{t:>10} | {q50:+8.2f} [{q16:+.2f}, {q84:+.2f}]{'':>7} | {per_z[0.004237]:+7.2f} {per_z[0.0134]:+7.2f} "
          f"{per_z[0.04237]:+7.2f} | {sig:+5.1f} | {flag}", flush=True)

with open('theory_k2_bands.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

med = np.array([r['dex_median'] for r in rows]); hw = np.array([(r['dex_84'] - r['dex_16']) / 2 for r in rows])
wts = 1 / hw ** 2
mean = np.sum(wts * med) / np.sum(wts); mean_err = 1 / np.sqrt(np.sum(wts))
chi2 = np.sum(((med - mean) / hw) ** 2)
within = np.sum(np.abs(med) < hw)
print(f"\n{len(rows)} systems: weighted mean log(obs/theory) = {mean:+.2f} +/- {mean_err:.2f} dex "
      f"({abs(mean) / mean_err:.1f} sigma from zero)")
print(f"  consistent with theory within 1 sigma: {within}/{len(rows)};  scatter about the mean: chi2 = {chi2:.1f} for {len(rows) - 1} dof")
noedge = [r for r in rows if 'edge-on' not in r['notes']]
if len(noedge) > 1:
    m2 = np.array([r['dex_median'] for r in noedge]); h2 = np.array([(r['dex_84'] - r['dex_16']) / 2 for r in noedge])
    w2 = 1 / h2 ** 2
    print(f"  excluding systems flagged 'edge-on e sin w' (n={len(noedge)}): weighted mean {np.sum(w2 * m2) / np.sum(w2):+.2f} "
          f"+/- {1 / np.sqrt(np.sum(w2)):.2f} dex")

fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(rows) + 1.5))
order = np.argsort(med)
for y, i in enumerate(order):
    r = rows[i]
    col = 'tab:orange' if 'edge-on' in r['notes'] else 'tab:blue'
    ax.errorbar(r['dex_median'], y, xerr=[[r['dex_median'] - r['dex_16']], [r['dex_84'] - r['dex_median']]],
                fmt='o', color=col, capsize=2, ms=5)
    ax.plot([r['dex_Zlow'], r['dex_Zhigh']], [y, y], '|', color='0.4', ms=8)
ax.axvline(0, color='k', lw=1); ax.axvspan(-0.3, 0.3, color='tab:green', alpha=0.12)
ax.axvline(mean, color='tab:red', ls='--', lw=1, label=f'weighted mean {mean:+.2f} ± {mean_err:.2f} dex')
ax.set_yticks(range(len(rows))); ax.set_yticklabels([str(rows[i]['tic']) for i in order], fontsize=8)
ax.set_xlabel('log$_{10}$(k$_2$ observed / k$_2$ theory)   [bars: 16–84%; grey ticks: metal-poor / metal-rich grids]')
ax.set_title(f'Observed vs Claret (2023) k$_2$, radius ±{args.sigR:.0%}, mass ±{args.sigM:.0%}\n'
             '(orange: e sin ω from edge-on duration formula, likely underestimated)', fontsize=10)
ax.legend(fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig('theory_k2_bands.png', dpi=150)
print('saved theory_k2_bands.csv and theory_k2_bands.png')
