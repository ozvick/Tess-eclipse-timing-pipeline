"""Observed vs theoretical apsidal-motion constant k2.

Theory: Claret (2023, A&A 674, A67) MESA tracks (solar composition, 0.2-8 Msun), which tabulate
log(k2) (apsidal-motion constant) along each star's evolution. For each star, the track at its mass
(interpolated between grid masses) is entered at the main-sequence/subgiant point where the model
radius equals the star's radius; the two stars' k2 are then combined with the same weights as in
the apsidal-motion formula:
    k2_theory = (c1 k2_1 + c2 k2_2)/(c1 + c2),
    c_i = (R_i/a)^5 [15 f(e) M_j/M_i + (P/P_rot)^2 g(e) (1 + M_j/M_i)].
Observed k2: consistency_modelfree.csv (model-free e cos w / e sin w). Masses, radii, separation and
rotation: Penev's fits (binary_radii*.csv).

The tracks are downloaded once from the VizieR mirror at Harvard and cached in claret2023_solar.csv.
Usage: python3 theory_k2.py        Writes theory_k2.csv and theory_k2.png.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MASSES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0,
          6.0, 7.0, 7.5, 8.0]
CACHE = 'claret2023_solar.csv'
LOGG_SUN = 4.438


def download_tracks():
    from astroquery.vizier import Vizier
    v = Vizier(columns=['**'], row_limit=-1, timeout=300)
    v.VIZIER_SERVER = 'vizier.cfa.harvard.edu'
    with open(CACHE, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['grid_mass', 'age', 'mass', 'logg', 'logk2'])
        for n, m in enumerate(MASSES, 1):
            t = v.get_catalogs(f'J/A+A/674/A67/table{n}')[0]
            for r in t:
                w.writerow([m, float(r['Age']), float(r['Mass']), float(r['log(g)']), float(r['log(k2)'])])
            print(f'  downloaded table{n} (M = {m} Msun, {len(t)} rows)', flush=True)


def load_tracks():
    if not os.path.exists(CACHE):
        print('downloading Claret (2023) solar-composition tracks from the Harvard VizieR mirror...')
        download_tracks()
    tr = {}
    for r in csv.DictReader(open(CACHE)):
        tr.setdefault(float(r['grid_mass']), []).append((float(r['age']), float(r['mass']), float(r['logg']), float(r['logk2'])))
    return {m: np.array(sorted(v)) for m, v in tr.items()}


def ms_segment(track):
    """Main sequence + subgiant part of a track. The ZAMS is the point of maximum surface gravity
    between the first time log g exceeds 3.8 (end of the pre-main-sequence contraction; robust to
    the small reversals real tracks show there) and the first later time it falls below 3.0 (giant)."""
    logg = track[:, 2]
    hi = np.where(logg >= 3.8)[0]
    start = int(hi[0]) if len(hi) else 0
    # stop at the giant branch (log g < 3.0) or, for stars that never become giants, before the
    # white-dwarf stage (log g > 5.5; the lowest-mass main-sequence stars reach log g ~ 5.1)
    leave = np.where((logg[start:] < 3.0) | (logg[start:] > 5.5))[0]
    end = start + (int(leave[0]) if len(leave) else len(logg) - start)
    i0 = start + int(np.argmax(logg[start:end]))
    seg = track[i0:max(end, i0 + 2)]
    logR = 0.5 * (np.log10(seg[:, 1]) - (seg[:, 2] - LOGG_SUN))
    return logR, seg[:, 3], seg[:, 0]


def k2_on_track(track, R):
    """log k2, age and a status flag where the track's radius first equals R after the ZAMS."""
    logR, logk2, age = ms_segment(track)
    lr = np.log10(R)
    if lr <= logR[0]:
        return logk2[0], age[0], 'at/below ZAMS radius'
    above = np.where(logR >= lr)[0]
    if not len(above):
        return logk2[-1], age[-1], 'larger than track (evolved)'
    j = above[0]
    f = (lr - logR[j - 1]) / (logR[j] - logR[j - 1]) if logR[j] != logR[j - 1] else 0.0
    return logk2[j - 1] + f * (logk2[j] - logk2[j - 1]), age[j - 1] + f * (age[j] - age[j - 1]), 'ok'


def theory_logk2(tracks, M, R):
    grid = np.array(sorted(tracks))
    if M < grid[0] or M > grid[-1]:
        return np.nan, np.nan, 'mass outside 0.2-8 Msun grid'
    hi = int(np.searchsorted(grid, M)); lo = max(hi - 1, 0)
    if grid[hi] == M:
        lo = hi
    a = k2_on_track(tracks[grid[lo]], R); b = k2_on_track(tracks[grid[hi]], R)
    if lo == hi:
        return a
    f = (np.log10(M) - np.log10(grid[lo])) / (np.log10(grid[hi]) - np.log10(grid[lo]))
    flag = 'ok' if a[2] == b[2] == 'ok' else f'{a[2]} / {b[2]}'
    return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), flag


if __name__ == '__main__':
    tracks = load_tracks()
    print("ZAMS found on each track (check: 1 Msun -> R ~ 0.9, k2 ~ 0.02; 2 Msun -> R ~ 1.6-1.7, k2 ~ 0.005):")
    print(f"  {'M':>5} {'age_ZAMS[yr]':>13} {'R_ZAMS':>7} {'logg':>5} {'k2_ZAMS':>8} | {'R_end':>7} {'k2_end':>8}")
    for m in sorted(tracks):
        lR, lk, ag = ms_segment(tracks[m])
        print(f"  {m:5.2f} {ag[0]:13.3e} {10 ** lR[0]:7.3f} {LOGG_SUN + np.log10(m) - 2 * lR[0]:5.2f} {10 ** lk[0]:8.4f} | "
              f"{10 ** lR[-1]:7.2f} {10 ** lk[-1]:8.5f}")
    print()
    par = {}
    for f in ('binary_radii.csv', 'binary_radii_more.csv', 'binary_radii_paul.csv', 'binary_radii_ecc.csv'):
        if os.path.exists(f):
            for r in csv.DictReader(open(f)):
                par[int(float(r['tic']))] = r
    obs = list(csv.DictReader(open('consistency_modelfree.csv')))
    rows = []
    print(f"{'TIC':>10} {'P':>7} {'M1':>5} {'M2':>5} {'R1':>5} {'R2':>5} | {'k2 obs [95%]':>26} | {'k2 theory':>9} "
          f"{'(star1':>7} {'star2)':>7} | {'dex':>6} | notes")
    for o in obs:
        t = int(o['tic'])
        if t not in par:
            continue
        p = par[t]
        M1, M2, R1, R2, a = (float(p[k]) for k in ('M1', 'M2', 'R1', 'R2', 'a_Rsun'))
        P = float(o['P']); e = float(o['e_mf'])
        prot = float(p.get('prot') or 0); rot = P / prot if prot > 0 else 1.0
        l1, age1, f1 = theory_logk2(tracks, M1, R1)
        l2, age2, f2 = theory_logk2(tracks, M2, R2)
        fe = (1 + 1.5 * e ** 2 + 0.125 * e ** 4) / (1 - e ** 2) ** 5; ge = 1 / (1 - e ** 2) ** 2
        c1 = (R1 / a) ** 5 * (15 * fe * M2 / M1 + rot ** 2 * ge * (1 + M2 / M1))
        c2 = (R2 / a) ** 5 * (15 * fe * M1 / M2 + rot ** 2 * ge * (1 + M1 / M2))
        k_th = (c1 * 10 ** l1 + c2 * 10 ** l2) / (c1 + c2) if np.isfinite(l1) and np.isfinite(l2) else np.nan
        k_ob, lo, hi = float(o['k2_mf']), float(o['k2_mf_lo']), float(o['k2_mf_hi'])
        dex = np.log10(k_ob / k_th) if k_ob > 0 and np.isfinite(k_th) else np.nan
        notes = '; '.join(x for x in (f1 if f1 != 'ok' else '', f2 if f2 != 'ok' and f2 != f1 else '', o.get('notes', '')) if x)
        rows.append(dict(tic=t, P=P, M1=M1, M2=M2, R1=R1, R2=R2, e=e, k2_obs=k_ob, k2_obs_lo=lo, k2_obs_hi=hi,
                         k2_theory=k_th, k2_star1=10 ** l1 if np.isfinite(l1) else np.nan,
                         k2_star2=10 ** l2 if np.isfinite(l2) else np.nan, age1_yr=age1, age2_yr=age2,
                         weight_star1=c1 / (c1 + c2), log_obs_over_theory=dex, notes=notes))
        print(f"{t:>10} {P:7.3f} {M1:5.2f} {M2:5.2f} {R1:5.2f} {R2:5.2f} | {k_ob:8.4f} [{lo:.4f}, {hi:.4f}] | {k_th:9.4f} "
              f"{10 ** l1 if np.isfinite(l1) else np.nan:7.4f} {10 ** l2 if np.isfinite(l2) else np.nan:7.4f} | "
              f"{dex:+6.2f} | {notes}", flush=True)

    with open('theory_k2.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    good = [r for r in rows if np.isfinite(r['log_obs_over_theory']) and r['k2_obs_lo'] > 0
            and r['k2_obs_hi'] / r['k2_obs_lo'] < 10 and 'near-circular' not in r['notes']]
    if good:
        d = np.array([r['log_obs_over_theory'] for r in good])
        print(f"\nwell-constrained systems (95% range narrower than a factor 10, not near-circular): n={len(good)}")
        print(f"  log10(k2 obs / k2 theory): median {np.median(d):+.2f} dex, scatter (MAD) {1.4826 * np.median(np.abs(d - np.median(d))):.2f} dex, "
              f"within 0.3 dex: {np.sum(np.abs(d) < 0.3)}/{len(d)}")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for r in rows:
        if not (np.isfinite(r['k2_theory']) and r['k2_obs'] > 0):
            continue
        ok = r in good
        lo, hi = max(r['k2_obs_lo'], 1e-5), r['k2_obs_hi']
        ax.errorbar(r['k2_theory'], r['k2_obs'], yerr=[[r['k2_obs'] - lo], [hi - r['k2_obs']]], fmt='o', ms=6 if ok else 4,
                    color='tab:blue' if ok else '0.7', capsize=2, lw=1, zorder=3 if ok else 1)
        if ok:
            ax.annotate(str(r['tic']), (r['k2_theory'], r['k2_obs']), fontsize=6, xytext=(4, 3), textcoords='offset points')
    lim = [5e-4, 0.3]
    ax.plot(lim, lim, 'k-', lw=1, label='observed = theory')
    ax.fill_between(lim, [x / 2 for x in lim], [x * 2 for x in lim], color='tab:green', alpha=0.12, label='within a factor of 2')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(1e-4, 1)
    ax.set_xlabel('theoretical k$_2$ (Claret 2023 tracks, weighted over both stars)')
    ax.set_ylabel('observed k$_2$ (tides + GR, model-free e)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Apsidal-motion constant: observed vs stellar models', fontsize=11)
    fig.tight_layout(); fig.savefig('theory_k2.png', dpi=150)
    print('saved theory_k2.csv and theory_k2.png')
